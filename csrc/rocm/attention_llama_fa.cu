// SPDX-License-Identifier: Apache-2.0
//
// Paged-KV port of llama.cpp's ggml flash-attention "tile" kernel:
//   ggml/src/ggml-cuda/fattn-tile.cuh -> flash_attn_tile<DKQ=256, DV=256, ...>
// which is the kernel llama.cpp selects on gfx90a (MI250) for head_size=256,
// GQA, small batch, non-quantized KV (BEST_FATTN_KERNEL_TILE).
//
// What is kept identical to llama.cpp:
//   - 256 threads/block (32 lanes x 8 warps), __launch_bounds__ occupancy 2
//   - tile sizes from llama.cpp's AMD config table:
//       ncols=2 (qlen 1): nbatch_fa=128, nbatch_K=64
//       ncols=4/8 (qlen 2/4): nbatch_fa=64, nbatch_K=128
//   - GQA packing: 2 Q heads sharing one KV head per block (ncols2 = 2);
//     qlen > 1 packs qlen Q tokens x 2 heads = ncols columns per block
//   - online softmax: running max/sum, VKQ rescale by exp(old_max - new_max)
//   - KV split over gridDim.y (parallel_blocks) with llama's occupancy/wave
//     efficiency heuristic + separate combine kernel (float partials + (max,sum))
// What is adapted for vLLM:
//   - KV cache uses vLLM/ROCm paged layouts (PagedAttention.split_kv_cache):
//       key_cache   [num_blocks, num_kv_heads, head_size/8, block_size, 8]
//       value_cache [num_blocks, num_kv_heads, head_size, block_size]
//     K keeps 8 dims contiguous per token (16B vector loads along dims,
//     threads remapped to run along tokens for coalescing); V is token-minor,
//     so it is staged through registers and transposed into shared memory
//     (block_size arbitrary, e.g. 400; the out-of-bounds check is always on)
//   - dtype is bf16 (vLLM cache) instead of fp16; VKQ accumulated in float2
//     (llama.cpp uses half2); KQ probabilities staged as fp16 like llama.cpp
//   - no mask tensor: causality for the qlen trailing tokens is computed
//     analytically (token j attends to KV [0, seq_len - qlen + j]); no ALiBi /
//     softcap / sinks

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>

namespace {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

constexpr int LLAMA_FA_D = 256;   // head size (DKQ = DV)
constexpr int LLAMA_FA_D2 = 128;  // bf162 elements per head row
constexpr int LLAMA_FA_NB_V = 32; // V rows staged per chunk (all configs)
constexpr int LLAMA_FA_THREADS = 256;

__device__ __forceinline__ float warp_reduce_max_32(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum_32(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32);
  }
  return v;
}

// llama.cpp flash_attn_tile<256, 256> decode/spec-verify specialization with
// paged KV. QLEN = query tokens per sequence (1 = decode, 4 = MTP with 3
// draft tokens, ...). Supported: 1, 2, 4.
// grid  : (num_seqs * (num_q_heads/2), parallel_blocks, 1)
// block : (32, 8, 1)
template <int QLEN>
__global__ __launch_bounds__(LLAMA_FA_THREADS, 2) void llama_fa_tile_kernel(
    const bf16_t* __restrict__ q,  // [num_seqs*QLEN, num_q_heads, D]
    const bf16_t* __restrict__ kc, // [num_blocks, num_kv_heads, D/8, block_size, 8]
    const bf16_t* __restrict__ vc, // [num_blocks, num_kv_heads, D, block_size]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks]
    const int* __restrict__ seq_lens,     // [num_seqs]  (includes new tokens)
    const int head_groups,   // num_q_heads / 2
    const int gqa_ratio,     // num_q_heads / num_kv_heads (must be even)
    const int num_kv_heads, const int block_size, const int max_blocks,
    const int num_blocks, // total KV blocks in the caches (bounds clamp)
    const int64_t kc_blk, const int64_t kc_h, // elem strides of key_cache
    const int64_t vc_blk, const int64_t vc_h, // elem strides of value_cache
    const float scale, const int pb,
    bf16_t* __restrict__ out,    // [num_seqs*QLEN, num_q_heads*D]
    float* __restrict__ parts,   // [num_seqs*QLEN*Hq, pb, D]  (pb > 1)
    float2* __restrict__ meta,   // [num_seqs*QLEN*Hq, pb]     (pb > 1)
    const int dbg) {
  static_assert(QLEN == 1 || QLEN == 2 || QLEN == 4, "bad QLEN");
  constexpr int NCOLS = 2 * QLEN;         // Q columns per block
  constexpr int NP = 8 / NCOLS;           // warps per Q column
  constexpr int NB_FA = QLEN == 1 ? 128 : 64; // KV rows per tile
  constexpr int NB_K = QLEN == 1 ? 64 : 128;  // K elements per chunk
  constexpr int NROWS = NB_FA / (NP * 32);    // KV rows per lane in KQ
  constexpr int K_RS = NB_K / 2 + 4;      // K staging row stride (bf162)
  constexpr int V_RS = LLAMA_FA_D2 + 2;   // V staging row stride (bf162, padded
                                          // against transpose-store conflicts)
  constexpr int NCHUNK_D = LLAMA_FA_D / NB_K; // K chunks per tile

  const int seq = blockIdx.x / head_groups;
  const int grp = blockIdx.x % head_groups;
  const int head0 = grp * 2; // first of the 2 Q heads of this block
  const int kvh = head0 / gqa_ratio;
  // clamp for padded rows under CUDA graph capture; also clamp against
  // garbage/stale seq_lens so the block-table index below can never run
  // past the [num_seqs, max_blocks] table
  int seq_len = max(seq_lens[seq], QLEN);
  const int seq_cap = max_blocks * block_size;
  if (seq_len > seq_cap) {
    if (dbg) {
      printf("LLAMA_FA OOB-SL: seq=%d seq_len=%d capped to %d\n", seq, seq_len,
             seq_cap);
    }
    seq_len = seq_cap;
  }

  // NOTE: block/head strides come from the tensor: vLLM's KV cache block
  // stride is 2x the contiguous size (K/V blocks interleaved in one storage).
  const int* bt_row = block_tables + (int64_t)seq * max_blocks;

  // Padded graph rows can carry arbitrary stale block ids; clamp every
  // block-table value into [0, num_blocks) so a bad id degrades the row's
  // (discarded) output instead of faulting the whole engine.
  auto load_bid = [&](int t) -> int {
    const int idx = t / block_size;
    const int b = bt_row[idx];
    if (dbg && (b < 0 || b >= num_blocks)) {
      printf("LLAMA_FA OOB-BT: seq=%d blk=%d t=%d sl=%d idx=%d/%d bid=%d/%d\n",
             seq, (int)blockIdx.y, t, seq_len, idx, max_blocks, b, num_blocks);
    }
    return (b < 0 || b >= num_blocks) ? 0 : b;
  };

  const int tx = threadIdx.x; // 0..31
  const int ty = threadIdx.y; // 0..7
  const int jc = ty / NP;     // Q column of this warp (0..NCOLS-1)
  const int t8 = ty * 32 + tx;

  if (dbg && blockIdx.x == 0 && blockIdx.y == 0 && tx == 0 && ty == 0) {
    const int b0 = load_bid(0);
    const int64_t koff = (int64_t)b0 * kc_blk + kvh * kc_h;
    printf("DBG tile: seq_len=%d bt0=%d kvh=%d k0=%f v0=%f q0=%f\n", seq_len,
           b0, kvh, __bfloat162float(kc[koff]),
           __bfloat162float(vc[(int64_t)b0 * vc_blk + kvh * vc_h]),
               __bfloat162float(q[(int64_t)(seq * QLEN) * (head_groups * 2) *
                                      LLAMA_FA_D]));
  }

  __shared__ __align__(16) bf162_t Q_tmp[8][LLAMA_FA_D2];
  __shared__ __align__(16) bf162_t KV_tmp[4608];
  __shared__ __half KQb[512]; // [NCOLS][NB_FA] flat

  // ---- load Q rows (with softmax scale applied), one column per warp ----
  {
    const bf16_t* qp = q + ((int64_t)(seq * QLEN + (jc >> 1)) *
                                (head_groups * 2) + head0 + (jc & 1)) *
                               LLAMA_FA_D;
    for (int idx = (ty % NP) * 32 + tx; idx < LLAMA_FA_D2; idx += NP * 32) {
      float2 f = __bfloat1622float2(*(const bf162_t*)(qp + 2 * idx));
      f.x *= scale;
      f.y *= scale;
      Q_tmp[jc][idx] = __float22bfloat162_rn(f);
    }
  }
  __syncthreads();

  float KQ_max = -FLT_MAX / 2.0f;
  float KQ_sum = 0.0f;
  float2 VKQ[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};

  // causal limit for this warp's column: token j attends KV [0, seq_len-QLEN+j]
  const int col_limit = seq_len - QLEN + 1 + (jc >> 1);

  // ---- main loop over KV tiles (llama.cpp: k_VKQ_0 loop) ----
  // NOTE: the loop bound must be uniform across the block (it contains
  // __syncthreads), so it uses seq_len; columns whose causal limit is already
  // exhausted contribute p=0 (no-op) in the extra tiles.
  for (int k0 = blockIdx.y * NB_FA; k0 < seq_len; k0 += pb * NB_FA) {
    const int sup = col_limit - k0; // valid rows in this tile
    float acc[NROWS] = {0.0f};

    // KQ = K @ Q over D, staged in chunks of NB_K elements
#pragma unroll
    for (int c = 0; c < NCHUNK_D; ++c) {
      // stage K chunk: NB_FA rows x NB_K halves, 16B vector loads.
      // K layout [nb, h, D/8, bs, 8]: 8 dims contiguous per token, so lanes
      // are mapped along tokens (16B stride per lane -> coalesced).
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int chunk_id = it * 256 + t8;
        const int r = chunk_id % NB_FA;   // token row
        const int ch = chunk_id / NB_FA;  // dim group (8 dims each)
        const int t = k0 + r;
        bf162_t val[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        if (t < seq_len) {
          const int b = load_bid(t);
          const int tl = t % block_size;
          const int dg = c * (NB_K / 8) + ch;
          const int64_t off = (int64_t)b * kc_blk + kvh * kc_h +
                              (int64_t)dg * block_size * 8 + tl * 8;
          *reinterpret_cast<int4*>(val) =
              *reinterpret_cast<const int4*>(kc + off);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) KV_tmp[r * K_RS + ch * 4 + l] = val[l];
      }
      __syncthreads();

      // dot products of my KV row(s) with my Q column over this chunk
#pragma unroll
      for (int n = 0; n < NROWS; ++n) {
        const int i_KQ = n * (NP * 32) + (ty % NP) * 32 + tx;
#pragma unroll
        for (int m = 0; m < NB_K / 2; m += 4) {
          int4 kraw = *reinterpret_cast<const int4*>(&KV_tmp[i_KQ * K_RS + m]);
          int4 qraw = *reinterpret_cast<const int4*>(
              &Q_tmp[jc][c * (NB_K / 2) + m]);
          const bf162_t* kk = reinterpret_cast<const bf162_t*>(&kraw);
          const bf162_t* qq = reinterpret_cast<const bf162_t*>(&qraw);
#pragma unroll
          for (int l = 0; l < 4; ++l) {
            const float2 kf = __bfloat1622float2(kk[l]);
            const float2 qf = __bfloat1622float2(qq[l]);
            acc[n] += qf.x * kf.x + qf.y * kf.y;
          }
        }
      }
      __syncthreads();
    }

    // ---- online softmax update (llama.cpp iter softmax) ----
    float m = KQ_max;
#pragma unroll
    for (int n = 0; n < NROWS; ++n) {
      const int i_KQ = n * (NP * 32) + (ty % NP) * 32 + tx;
      m = fmaxf(m, i_KQ < sup ? acc[n] : -FLT_MAX / 2.0f);
    }
    m = warp_reduce_max_32(m);
    if constexpr (NP > 1) {
      // combine the NP warps of this Q column
      __shared__ float red[8];
      if (tx == 0) red[ty] = m;
      __syncthreads();
      m = red[(ty & ~(NP - 1)) + (tx & (NP - 1))];
#pragma unroll
      for (int off = NP / 2; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffffffffffULL, m, off, NP));
      }
    }
    const float max_scale = expf(KQ_max - m);
    KQ_max = m;
    float psum = 0.0f;
#pragma unroll
    for (int n = 0; n < NROWS; ++n) {
      const int i_KQ = n * (NP * 32) + (ty % NP) * 32 + tx;
      const float p = i_KQ < sup ? expf(acc[n] - m) : 0.0f;
      KQb[jc * NB_FA + i_KQ] = __float2half(p);
      psum += p;
    }
    KQ_sum = KQ_sum * max_scale + psum;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      VKQ[v].x *= max_scale;
      VKQ[v].y *= max_scale;
    }
    __syncthreads();

    // ---- VKQ = V @ KQ, V staged in chunks of NB_V=32 rows ----
#pragma unroll
    for (int k0v = 0; k0v < NB_FA; k0v += LLAMA_FA_NB_V) {
      // stage V chunk: 32 tokens x 256 dims. V layout [nb, h, D, bs] is
      // token-minor: load 8 tokens x 1 dim per lane (16B, coalesced), then
      // transpose through registers into shared [token][dim].
      bf16_t* KV16 = reinterpret_cast<bf16_t*>(KV_tmp);
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int chunk_id = it * 256 + t8;
        const int d = chunk_id >> 2; // 0..255
        const int tg = chunk_id & 3; // token group of 8
        const int rl0 = tg * 8;      // local token base in this chunk
        const int t0 = k0 + k0v + rl0;
        bf16_t vreg[8];
        *reinterpret_cast<int4*>(vreg) = make_int4(0, 0, 0, 0);
        if (t0 < seq_len) {
          const int b0 = load_bid(t0);
          const int tl0 = t0 % block_size;
          const int64_t vbase =
              (int64_t)b0 * vc_blk + kvh * vc_h +
              (int64_t)d * block_size;
          if (tl0 + 8 <= block_size && t0 + 8 <= seq_len) {
            *reinterpret_cast<int4*>(vreg) =
                *reinterpret_cast<const int4*>(vc + vbase + tl0);
          } else {
            // rare: token group crosses a page boundary or the sequence end
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const int t = t0 + i;
              if (t < seq_len) {
                const int b = load_bid(t);
                vreg[i] = vc[(int64_t)b * vc_blk + kvh * vc_h +
                             (int64_t)d * block_size + t % block_size];
              }
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          KV16[(rl0 + i) * (2 * V_RS) + d] = vreg[i];
        }
      }
      __syncthreads();

#pragma unroll
      for (int k1 = 0; k1 < LLAMA_FA_NB_V; k1 += NP) {
        const int rl = k1 + (ty % NP); // V row handled by this warp
        const float pv = __half2float(KQb[jc * NB_FA + k0v + rl]);
        const bf162_t* vrow = &KV_tmp[rl * V_RS];
        int2 v01 = *reinterpret_cast<const int2*>(&vrow[2 * tx]);
        int2 v23 = *reinterpret_cast<const int2*>(&vrow[64 + 2 * tx]);
        const float2 f0 = __bfloat1622float2(((const bf162_t*)&v01)[0]);
        const float2 f1 = __bfloat1622float2(((const bf162_t*)&v01)[1]);
        const float2 f2 = __bfloat1622float2(((const bf162_t*)&v23)[0]);
        const float2 f3 = __bfloat1622float2(((const bf162_t*)&v23)[1]);
        VKQ[0].x += f0.x * pv;
        VKQ[0].y += f0.y * pv;
        VKQ[1].x += f1.x * pv;
        VKQ[1].y += f1.y * pv;
        VKQ[2].x += f2.x * pv;
        VKQ[2].y += f2.y * pv;
        VKQ[3].x += f3.x * pv;
        VKQ[3].y += f3.y * pv;
      }
      __syncthreads();
    }
  }

  // ---- epilogue: reduce the NP warps of each Q column ----
  KQ_sum = warp_reduce_sum_32(KQ_sum);
  if constexpr (NP > 1) {
    __syncthreads(); // KV_tmp / Q_tmp are reused as combine buffers
    float* comb = reinterpret_cast<float*>(&KV_tmp[0]);   // [8][32][8] floats
    float* sumc = reinterpret_cast<float*>(&Q_tmp[0][0]); // [8] floats
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      comb[(ty * 32 + tx) * 8 + 2 * v] = VKQ[v].x;
      comb[(ty * 32 + tx) * 8 + 2 * v + 1] = VKQ[v].y;
    }
    if (tx == 0) sumc[ty] = KQ_sum;
    __syncthreads();
    if ((ty % NP) != 0) return;
#pragma unroll
    for (int ip = 1; ip < NP; ++ip) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        VKQ[v].x += comb[((ty + ip) * 32 + tx) * 8 + 2 * v];
        VKQ[v].y += comb[((ty + ip) * 32 + tx) * 8 + 2 * v + 1];
      }
      KQ_sum += sumc[ty + ip];
    }
  }

  // ---- write back ----
  if (dbg && blockIdx.x == 0 && blockIdx.y == 0 && tx == 0 && ty == 0) {
    printf("DBG done: KQ_sum=%f KQ_max=%f VKQ0=%f\n", KQ_sum, KQ_max,
           VKQ[0].x);
  }
  const int64_t out_row = (int64_t)seq * QLEN + (jc >> 1);
  const int head = head0 + (jc & 1);
  if (pb == 1) {
    const float inv = 1.0f / KQ_sum;
    bf16_t* op =
        out + (out_row * (head_groups * 2) + head) * LLAMA_FA_D;
    *reinterpret_cast<bf162_t*>(op + 4 * tx) =
        __float22bfloat162_rn(make_float2(VKQ[0].x * inv, VKQ[0].y * inv));
    *reinterpret_cast<bf162_t*>(op + 4 * tx + 2) =
        __float22bfloat162_rn(make_float2(VKQ[1].x * inv, VKQ[1].y * inv));
    *reinterpret_cast<bf162_t*>(op + 128 + 4 * tx) =
        __float22bfloat162_rn(make_float2(VKQ[2].x * inv, VKQ[2].y * inv));
    *reinterpret_cast<bf162_t*>(op + 128 + 4 * tx + 2) =
        __float22bfloat162_rn(make_float2(VKQ[3].x * inv, VKQ[3].y * inv));
  } else {
    const int64_t zh = out_row * (head_groups * 2) + head;
    float* pp = parts + (zh * pb + blockIdx.y) * LLAMA_FA_D;
    pp[4 * tx + 0] = VKQ[0].x;
    pp[4 * tx + 1] = VKQ[0].y;
    pp[4 * tx + 2] = VKQ[1].x;
    pp[4 * tx + 3] = VKQ[1].y;
    pp[128 + 4 * tx + 0] = VKQ[2].x;
    pp[128 + 4 * tx + 1] = VKQ[2].y;
    pp[128 + 4 * tx + 2] = VKQ[3].x;
    pp[128 + 4 * tx + 3] = VKQ[3].y;
    if (tx == 0) meta[zh * pb + blockIdx.y] = make_float2(KQ_max, KQ_sum);
  }
}

// llama.cpp flash_attn_combine_results<D=256>, writes bf16 directly.
__global__ __launch_bounds__(256, 1) void llama_fa_combine_kernel(
    const float* __restrict__ parts, const float2* __restrict__ meta,
    bf16_t* __restrict__ out, const int pb) {
  const int64_t zh = blockIdx.x; // out_row*num_q_heads + head
  const int tid = threadIdx.x;   // 0..255
  parts += zh * pb * LLAMA_FA_D;
  meta += zh * pb;

  extern __shared__ float2 smeta[];
  for (int i = tid; i < pb; i += 256) smeta[i] = meta[i];
  __syncthreads();

  float kqmax = smeta[0].x;
  for (int l = 1; l < pb; ++l) kqmax = fmaxf(kqmax, smeta[l].x);

  float num = 0.0f, den = 0.0f;
  for (int l = 0; l < pb; ++l) {
    const float s = expf(smeta[l].x - kqmax);
    num += s * parts[l * LLAMA_FA_D + tid];
    den += s * smeta[l].y;
  }
  out[zh * LLAMA_FA_D + tid] = __float2bfloat16(num / den);
}

template <int QLEN>
void launch_llama_fa(const torch::Tensor& out, const torch::Tensor& query,
                     const torch::Tensor& key_cache,
                     const torch::Tensor& value_cache,
                     const torch::Tensor& block_tables,
                     const torch::Tensor& seq_lens, int num_seqs,
                     int num_q_heads, int num_kv_heads, int block_size,
                     int max_blocks, float scale) {
  const int gqa_ratio = num_q_heads / num_kv_heads;
  const int head_groups = num_q_heads / 2;
  const int ntiles_dst = num_seqs * head_groups;
  constexpr int NB_FA = QLEN == 1 ? 128 : 64;

  // llama.cpp launch_fattn heuristic (stream_k=false branch): parallel_blocks
  // from occupancy, grown while the wave efficiency improves. Computed from the
  // KV capacity (capture-stable under CUDA graphs), not the current seq_lens.
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const int occupancy = 2;
  const int blocks_per_wave = prop->multiProcessorCount * occupancy;
  const int ntiles_kv =
      ((int64_t)max_blocks * block_size + NB_FA - 1) / NB_FA;
  int pb = std::min(occupancy, ntiles_kv);
  int nwaves_best = 0, eff_best = 0;
  for (int t = pb; t <= ntiles_kv; ++t) {
    const int total = ntiles_dst * t;
    const int nwaves = (total + blocks_per_wave - 1) / blocks_per_wave;
    const int eff = 100 * total / (nwaves * blocks_per_wave);
    if (eff_best >= 95 && nwaves > nwaves_best) break;
    if (eff > eff_best) {
      nwaves_best = nwaves;
      eff_best = eff;
      pb = t;
    }
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(ntiles_dst, pb, 1);
  dim3 block(32, 8, 1);
  if (getenv("LLAMA_FA_DEBUG_CPP")) {
    fprintf(stderr,
            "llama_fa launch: grid=(%d,%d) pb=%d head_groups=%d gqa=%d "
            "out_ptr=%p out_numel=%ld kc_ptr=%p vc_ptr=%p q_ptr=%p\n",
            ntiles_dst, pb, pb, head_groups, gqa_ratio, out.data_ptr(),
            out.numel(), key_cache.data_ptr(), value_cache.data_ptr(),
            query.data_ptr());
  }

  bf16_t* out_ptr = reinterpret_cast<bf16_t*>(out.data_ptr());
  float* parts_ptr = nullptr;
  float2* meta_ptr = nullptr;
  torch::Tensor parts, meta;
  if (pb > 1) {
    parts = at::empty({(int64_t)num_seqs * QLEN * num_q_heads * pb * LLAMA_FA_D},
                      query.options().dtype(at::kFloat));
    meta = at::empty({(int64_t)num_seqs * QLEN * num_q_heads * pb * 2},
                     query.options().dtype(at::kFloat));
    parts_ptr = parts.data_ptr<float>();
    meta_ptr = reinterpret_cast<float2*>(meta.data_ptr<float>());
  }

  const int dbg = getenv("LLAMA_FA_DEBUG_CPP") ? 1 : 0;
  const int num_blocks = key_cache.size(0);
  llama_fa_tile_kernel<QLEN><<<grid, block, 0, stream>>>(
      reinterpret_cast<const bf16_t*>(query.data_ptr()),
      reinterpret_cast<const bf16_t*>(key_cache.data_ptr()),
      reinterpret_cast<const bf16_t*>(value_cache.data_ptr()),
      block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), head_groups,
      gqa_ratio, num_kv_heads, block_size, max_blocks, num_blocks,
      key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
      value_cache.stride(1), scale, pb, out_ptr,
      parts_ptr, meta_ptr, dbg);

  if (pb > 1) {
    llama_fa_combine_kernel<<<num_seqs * QLEN * num_q_heads, LLAMA_FA_D,
                              pb * sizeof(float2), stream>>>(
        parts_ptr, meta_ptr, out_ptr, pb);
  }
}

} // namespace

void paged_attention_llama_fa(torch::Tensor& out, torch::Tensor& query,
                              torch::Tensor& key_cache,
                              torch::Tensor& value_cache,
                              torch::Tensor& block_tables,
                              torch::Tensor& seq_lens, int64_t num_kv_heads,
                              double scale, int64_t qlen) {
  if (getenv("LLAMA_FA_DEBUG_CPP")) {
    int cur = -1;
    hipGetDevice(&cur);
    fprintf(stderr,
            "llama_fa enter: cur_dev=%d q_dev=%ld kc_dev=%ld out_dev=%ld "
            "bt_dev=%ld sl_dev=%ld\n",
            cur, query.get_device(), key_cache.get_device(),
            out.get_device(), block_tables.get_device(),
            seq_lens.get_device());
  }
  const int num_seqs = seq_lens.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int max_blocks = block_tables.size(1);

  TORCH_CHECK(head_size == LLAMA_FA_D, "llama_fa: head_size must be 256");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kBFloat16,
              "llama_fa: bf16 only");
  // vLLM/ROCm paged layouts from PagedAttention.split_kv_cache:
  //   K [nb, hkv, D/8, bs, 8], V [nb, hkv, D, bs]
  TORCH_CHECK(key_cache.dim() == 5 && key_cache.size(2) == LLAMA_FA_D / 8 &&
                  key_cache.size(4) == 8,
              "llama_fa: key_cache must be [nb, hkv, D/8, bs, 8]");
  TORCH_CHECK(value_cache.dim() == 4 &&
                  value_cache.size(2) == LLAMA_FA_D &&
                  value_cache.size(3) == key_cache.size(3),
              "llama_fa: value_cache must be [nb, hkv, D, bs]");
  const int block_size = key_cache.size(3);
  TORCH_CHECK(key_cache.stride(2) == (int64_t)block_size * 8 &&
                  key_cache.stride(3) == 8 && key_cache.stride(4) == 1,
              "llama_fa: key_cache inner layout must be [D/8, bs, 8] dense");
  TORCH_CHECK(value_cache.stride(2) == block_size &&
                  value_cache.stride(3) == 1,
              "llama_fa: value_cache inner layout must be [D, bs] dense");
  TORCH_CHECK(num_q_heads % 2 == 0, "llama_fa: odd num_q_heads");
  const int gqa_ratio = num_q_heads / (int)num_kv_heads;
  TORCH_CHECK(gqa_ratio >= 2 && gqa_ratio % 2 == 0,
              "llama_fa: gqa_ratio must be even (heads sharing a KV head are "
              "packed 2/block)");
  TORCH_CHECK(query.size(0) == (int64_t)num_seqs * qlen,
              "llama_fa: non-uniform query length across sequences");

  switch (qlen) {
    case 1:
      launch_llama_fa<1>(out, query, key_cache, value_cache, block_tables,
                         seq_lens, num_seqs, num_q_heads, (int)num_kv_heads,
                         block_size, max_blocks, (float)scale);
      break;
    case 2:
      launch_llama_fa<2>(out, query, key_cache, value_cache, block_tables,
                         seq_lens, num_seqs, num_q_heads, (int)num_kv_heads,
                         block_size, max_blocks, (float)scale);
      break;
    case 4:
      launch_llama_fa<4>(out, query, key_cache, value_cache, block_tables,
                         seq_lens, num_seqs, num_q_heads, (int)num_kv_heads,
                         block_size, max_blocks, (float)scale);
      break;
    default:
      TORCH_CHECK(false, "llama_fa: qlen must be 1, 2 or 4");
  }
  // NOTE: hipGetLastError/hipDeviceSynchronize are illegal during CUDA graph
  // capture; gate them on a separate flag so LLAMA_FA_DEBUG_CPP=1 (device-side
  // bounds printf, baked into captured kernels) stays capture-safe.
  if (getenv("LLAMA_FA_DEBUG_SYNC")) {
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      fprintf(stderr, "llama_fa launch err: %s\n", hipGetErrorString(e));
    } else {
      e = hipDeviceSynchronize();
      fprintf(stderr, "llama_fa sync: %s\n", hipGetErrorString(e));
      __hip_bfloat16 hbuf[8];
      if (hipMemcpy(hbuf, out.data_ptr(), sizeof(hbuf),
                    hipMemcpyDeviceToHost) == hipSuccess) {
        fprintf(stderr,
                "llama_fa out[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                __bfloat162float(hbuf[0]), __bfloat162float(hbuf[1]),
                __bfloat162float(hbuf[2]), __bfloat162float(hbuf[3]),
                __bfloat162float(hbuf[4]), __bfloat162float(hbuf[5]),
                __bfloat162float(hbuf[6]), __bfloat162float(hbuf[7]));
      }
    }
  }
}
