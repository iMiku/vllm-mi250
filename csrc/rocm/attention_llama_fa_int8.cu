// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.8-27B int8-KV decode attention：llama.cpp fattn-tile 结构 + int8 KV 路径。
//   K int8 [nb, hkv, D/16, bs, 16] / V int8 [nb, hkv, D, bs] / scales f32 [nb, bs, hkv]
// host 路径与 csrc/rocm/attention_llama_fa.cu 逐字同构（同一条 getCurrentCUDAStream）。
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

// 探针: 把第 q 个 32-bit 字原样重解释为 bf162（不转换）
__device__ __forceinline__ bf162_t make_bf162_from_words(unsigned int w0,
                                                         unsigned int w1,
                                                         unsigned int w2,
                                                         unsigned int w3, int q) {
  const unsigned int w = (q < 4) ? ((q == 0) ? w0 : (q == 1) ? w1
                                     : (q == 2) ? w2 : w3) : 0u;
  const int2 p = make_int2((int)w, (int)w);
  return *reinterpret_cast<const bf162_t*>(&p);
}

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
template <int QLEN, int PROBE = 0, bool I8DOT = false, bool HOIST = false>
__global__ __launch_bounds__(LLAMA_FA_THREADS, 1) void llama_fa_tile_kernel_int8(
    const bf16_t* __restrict__ q,  // [num_seqs*QLEN, num_q_heads, D]
    const int8_t* __restrict__ kc, // [num_blocks, num_kv_heads, D/16, block_size, 16]
    const int8_t* __restrict__ vc, // [num_blocks, num_kv_heads, D, block_size]
    const float* __restrict__ k_scale, // [num_blocks, block_size, num_kv_heads]
    const float* __restrict__ v_scale, // [num_blocks, block_size, num_kv_heads]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks]
    const int* __restrict__ seq_lens,     // [num_seqs]  (includes new tokens)
    const int head_groups,   // num_q_heads / 2
    const int gqa_ratio,     // num_q_heads / num_kv_heads (must be even)
    const int num_kv_heads, const int block_size, const int max_blocks,
    const int num_blocks, // total KV blocks in the caches (bounds clamp)
    const int64_t kc_blk, const int64_t kc_h, // elem strides of key_cache
    const int64_t vc_blk, const int64_t vc_h, // elem strides of value_cache
    const int64_t ks_blk, const int64_t ks_slot, const int64_t ks_h,
    const int64_t vs_blk, const int64_t vs_slot, const int64_t vs_h,
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
  __shared__ float KS_s[NB_FA]; // per-token K scale for this tile
  __shared__ float VS_s[NB_FA]; // per-token V scale for this tile
  // I8DOT: Q 量化成 int8 + K 以 int32 字存 smem（16 个 int8 / 16B 载入，零转换）
  __shared__ __align__(16) int8_t Q8[8][LLAMA_FA_D];
  __shared__ float Qscale[8];
  __shared__ float Qsm[8];
  constexpr int K_RS8 = NB_K / 4 + 4;   // K 行 stride（int32 字，带 pad）
  int32_t* K8 = reinterpret_cast<int32_t*>(&KV_tmp[0]);

  // ---- load Q rows ----
  const bf16_t* qp = q + ((int64_t)(seq * QLEN + (jc >> 1)) *
                              (head_groups * 2) + head0 + (jc & 1)) *
                             LLAMA_FA_D;
  if constexpr (I8DOT) {
    // 第一遍：per-(token,head) amax（柱内 NP 个 warp 归约），第二遍：量化成 int8
    float amax = 0.0f;
    for (int idx = (ty % NP) * 32 + tx; idx < LLAMA_FA_D; idx += NP * 32) {
      const float v = __bfloat162float(qp[idx]);
      amax = fmaxf(amax, fabsf(v));
    }
    amax = warp_reduce_max_32(amax);
    if (tx == 0) Qsm[ty] = amax;
    __syncthreads();
    if (ty < NCOLS) {
      float m = Qsm[ty * NP];
#pragma unroll
      for (int ip = 1; ip < NP; ++ip) m = fmaxf(m, Qsm[ty * NP + ip]);
      Qscale[ty] = (m > 0.0f) ? m / 127.0f : 1.0f;   // s_q（softmax scale 稍后一起乘）
    }
    __syncthreads();
    const float inv = 1.0f / Qscale[jc];
#pragma unroll
    for (int idx = (ty % NP) * 32 + tx; idx < LLAMA_FA_D; idx += NP * 32) {
      const float v = __bfloat162float(qp[idx]) * inv;
      int32_t qv = (int32_t)lrintf(v);
      qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
      Q8[jc][idx] = (int8_t)qv;
    }
  } else {
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

    // per-token-head scales for this KV tile (cooperative, 1 scalar/token/head)
#pragma unroll
    for (int i = t8; i < NB_FA; i += LLAMA_FA_THREADS) {
      const int t = k0 + i;
      float ks_ = 1.0f, vs_ = 1.0f;
      if (t < seq_len) {
        const int b = load_bid(t);
        const int tl = t % block_size;
        ks_ = k_scale[(int64_t)b * ks_blk + (int64_t)tl * ks_slot +
                      (int64_t)kvh * ks_h];
        vs_ = v_scale[(int64_t)b * vs_blk + (int64_t)tl * vs_slot +
                      (int64_t)kvh * vs_h];
      }
      KS_s[i] = ks_;
      VS_s[i] = vs_;
    }
    __syncthreads();

    // KQ = K @ Q over D, staged in chunks of NB_K elements
#pragma unroll
    for (int c = 0; c < NCHUNK_D; ++c) {
      // stage K chunk: NB_FA rows x NB_K int8, 16 dims (16B) per vector load.
      // K layout [nb, h, D/16, bs, 16]: 16 dims contiguous per token, lanes
      // run along tokens -> coalesced 16B loads.
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        const int chunk_id = it * 256 + t8;
        const int r = chunk_id % NB_FA;      // token row
        const int cg = chunk_id / NB_FA;     // dim group (16 dims each)
        const int t = k0 + r;
        unsigned int w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
        float s8v = 0.0f;
        if (t < seq_len) {
          const int b = load_bid(t);
          const int tl = t % block_size;
          const int dg = c * (NB_K / 16) + cg;
          const int64_t off = (int64_t)b * kc_blk + kvh * kc_h +
                              (int64_t)dg * block_size * 16 + tl * 16;
          const int4 raw = *reinterpret_cast<const int4*>(kc + off);
          w0 = (unsigned int)raw.x;
          w1 = (unsigned int)raw.y;
          w2 = (unsigned int)raw.z;
          w3 = (unsigned int)raw.w;
          s8v = KS_s[r];
        }
        if constexpr (I8DOT) {  // 原样 16B 存入 smem，零转换（K_I8DOT）
          // t >= seq_len 时 w0..w3 保持 0（上面已初始化），行为与原版一致
          K8[r * K_RS8 + cg * 4 + 0] = (int32_t)w0;
          K8[r * K_RS8 + cg * 4 + 1] = (int32_t)w1;
          K8[r * K_RS8 + cg * 4 + 2] = (int32_t)w2;
          K8[r * K_RS8 + cg * 4 + 3] = (int32_t)w3;
        } else if constexpr (PROBE & 1) {  // 探针: 不做 cvt/scale, 裸字节当 bf16 (数值错)
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            KV_tmp[r * K_RS + cg * 8 + q] =
                make_bf162_from_words(w0, w1, w2, w3, q);
          }
        } else {
        // 4 bytes -> 4 bf16 per word, no address-taken locals
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const unsigned int w = (q == 0) ? w0 : (q == 1) ? w1
                               : (q == 2) ? w2 : w3;
          const float a = (float)(int)(signed char)(w & 0xFFu) * (HOIST ? 1.0f : s8v);
          const float b_ = (float)(int)(signed char)((w >> 8) & 0xFFu) * (HOIST ? 1.0f : s8v);
          const float c_ = (float)(int)(signed char)((w >> 16) & 0xFFu) * (HOIST ? 1.0f : s8v);
          const float d_ = (float)(int)(signed char)((w >> 24) & 0xFFu) * (HOIST ? 1.0f : s8v);
          KV_tmp[r * K_RS + cg * 8 + 2 * q] =
              __float22bfloat162_rn(make_float2(a, b_));
          KV_tmp[r * K_RS + cg * 8 + 2 * q + 1] =
              __float22bfloat162_rn(make_float2(c_, d_));
        }
        }
      }
      __syncthreads();

      // dot products of my KV row(s) with my Q column over this chunk
#pragma unroll
      for (int n = 0; n < NROWS; ++n) {
        const int i_KQ = n * (NP * 32) + (ty % NP) * 32 + tx;
        if constexpr (I8DOT) {
          // int8 点积：4 个 MAC/指令，且 K 侧零转换；per-token scale 事后乘一次
          int a32 = 0;
          const int32_t* qrow =
              reinterpret_cast<const int32_t*>(&Q8[jc][c * NB_K]);
#pragma unroll
          for (int m = 0; m < NB_K / 4; ++m) {
            a32 = __builtin_amdgcn_sdot4(qrow[m], K8[i_KQ * K_RS8 + m], a32,
                                        false);
          }
          acc[n] = (float)a32 * (Qscale[jc] * scale) * KS_s[i_KQ];
          continue;
        }
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
    if constexpr (HOIST && !I8DOT) {
      // scale 提升：s_k 从"每元素乘"变成"每行乘一次"
#pragma unroll
      for (int n = 0; n < NROWS; ++n) {
        const int i_KQ = n * (NP * 32) + (ty % NP) * 32 + tx;
        acc[n] *= KS_s[i_KQ];
      }
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
      for (int it = 0; it < 2; ++it) {          // 32 rows / 16-token vectors
        const int chunk_id = it * 256 + t8;     // [0,512)
        const int d = chunk_id >> 1;            // 0..255
        const int tg = chunk_id & 1;            // which 16-token group
        const int rl0 = tg * 16;                // local token base
        const int t0 = k0 + k0v + rl0;
        unsigned int w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
        if (t0 < seq_len) {
          const int b0 = load_bid(t0);
          const int tl0 = t0 % block_size;
          const int64_t vbase =
              (int64_t)b0 * vc_blk + kvh * vc_h +
              (int64_t)d * block_size;
          if (tl0 + 16 <= block_size && t0 + 16 <= seq_len) {
            const int4 raw = *reinterpret_cast<const int4*>(vc + vbase + tl0);
            w0 = (unsigned int)raw.x;
            w1 = (unsigned int)raw.y;
            w2 = (unsigned int)raw.z;
            w3 = (unsigned int)raw.w;
          } else {
            // rare: token group crosses a page boundary or the sequence end
            const int8_t* vp = vc + vbase;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              const int t = t0 + i;
              unsigned int byte = 0u;
              if (t < seq_len) {
                const int b = load_bid(t);
                byte = (unsigned int)(unsigned char)
                           vc[(int64_t)b * vc_blk + kvh * vc_h +
                              (int64_t)d * block_size + t % block_size];
              }
              if (i < 4) {
                w0 |= byte << (8 * i);
              } else if (i < 8) {
                w1 |= byte << (8 * (i - 4));
              } else if (i < 12) {
                w2 |= byte << (8 * (i - 8));
              } else {
                w3 |= byte << (8 * (i - 12));
              }
            }
            (void)vp;
          }
        }
        if constexpr (PROBE & 2) {  // 探针: V 不做 cvt/scale
          const unsigned int ww[4] = {w0, w1, w2, w3};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int2 p2 = make_int2((int)ww[q], (int)ww[q]);
            const bf162_t* b2 = reinterpret_cast<const bf162_t*>(&p2);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              KV16[(rl0 + q * 4 + 2 * j) * (2 * V_RS) + d] =
                  reinterpret_cast<const bf16_t*>(b2)[j];
              KV16[(rl0 + q * 4 + 2 * j + 1) * (2 * V_RS) + d] = bf16_t(0.f);
            }
          }
        } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const unsigned int w = (q == 0) ? w0 : (q == 1) ? w1
                               : (q == 2) ? w2 : w3;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int i = q * 4 + j;
            const float val = (float)(int)(signed char)((w >> (8 * j)) & 0xFFu);
            KV16[(rl0 + i) * (2 * V_RS) + d] =
                __float2bfloat16(val * (HOIST ? 1.0f : VS_s[rl0 + i]));
          }
        }
        }
      }
      __syncthreads();

#pragma unroll
      for (int k1 = 0; k1 < LLAMA_FA_NB_V; k1 += NP) {
        const int rl = k1 + (ty % NP); // V row handled by this warp
        const float pv = __half2float(KQb[jc * NB_FA + k0v + rl]) *
                         (HOIST ? VS_s[k0v + rl] : 1.0f);
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


// ================= MFMA i8/f16 原生矩阵单元路径（qlen == 1）=================
// 与 llama_fa_tile_kernel_int8 的输出约定**完全一致**：parts（未归一化 Σ p·V，
// 已含 v_scale）+ meta（每切片 (m, l)），复用同一个 llama_fa_combine_kernel 合并。
// 精度：内核内无降低精度的量化 —— int8 来自既有量化缓存；V 的 int8->fp16 无损；
//       P 仅做 fp32->fp16 浮点舍入；累积全程 fp32。
typedef int v2i32 __attribute__((ext_vector_type(2)));
typedef int v4i32 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));
constexpr int MFMA_HQ = 16;  // Q 头数补齐到 16（12 真 + 4 零），M=16 满利用 75%

// KQ 用 f16 MFMA：A=K（int8->fp16 无损）、B=Q（bf16->fp16，等价无损，**不量化**）
__device__ __forceinline__ v4i32 mfma_i8_d(unsigned a, unsigned b, v4i32 c) {
  return __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ v4f32 mfma_f16_d(v2i32 a, v2i32 b, v4f32 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ uint2 cvt4_i8_f16_d(unsigned x) {
  __half2 h0 = __floats2half2_rn((float)(signed char)(x & 0xff),
                                 (float)(signed char)((x >> 8) & 0xff));
  __half2 h1 = __floats2half2_rn((float)(signed char)((x >> 16) & 0xff),
                                 (float)(signed char)(x >> 24));
  uint2 r;
  r.x = *reinterpret_cast<unsigned int*>(&h0);
  r.y = *reinterpret_cast<unsigned int*>(&h1);
  return r;
}

// 一 block = 一个 (seq, kv_head, pb 切片)：M=16 头（补齐）、K=256（16 步 MFMA）
// 交换操作数（A=K, B=Q）=> C[token][head] => P 天然落在 PV 的 A 片段布局
template <bool PACKED>
__global__ void __launch_bounds__(64, 4)
llama_fa_mfma_kernel(const signed char* __restrict__ kcache,
                     const signed char* __restrict__ vcache,
                     const float* __restrict__ k_scale,
                     const float* __restrict__ v_scale,
                     const bf16_t* __restrict__ query,       // [T][num_q_heads][256]
                     float* __restrict__ parts,              // [T*HQ*pb*D]
                     float2* __restrict__ meta,              // [T*HQ*pb]
                     const int* __restrict__ block_table,    // [T][max_blocks]
                     const int* __restrict__ seq_lens, const int max_blocks,
                     const int bs, const int hkv, const int num_q_heads,
                     const int pb, const float softmax_scale,
                     // 运行时 stride（元素单位）：允许 K/V 是打包 cache 的 as_strided 视图
                     const int64_t k_blk, const int64_t k_h, const int64_t v_blk,
                     const int64_t v_h, const int64_t k_slot,
                     const int64_t v_slot, const int64_t v_off,
                     const int64_t ss_blk, const int64_t ss_slot,
                     const int64_t ss_head) {
  const int L = threadIdx.x;
  const int seq = blockIdx.y;
  const int kvh = blockIdx.x % hkv;
  const int slice = blockIdx.x / hkv;
  const int mg = L / 16, mi = L % 16;
  const int seq_len = seq_lens[seq];
  const int* bt = block_table + (int64_t)seq * max_blocks;
  // ★ GQA 分组：一个 kv_head 只服务它自己那 gqa 个 Q 头（否则 hkv 个 block 会
  //   往同一片 parts 写同样的头 -> 竞态写坏）
  const int gqa = num_q_heads / hkv;
  const int hbase = kvh * gqa;   // 本 block 负责的 Q 头区间起点

  const int dg_stride = bs * 16;
  const int ks_blk = bs * hkv;
  (void)ks_blk;
  // K 的 A 片段：每步 4 字节。dim-major 下步进 = dg_stride（跨 16 维组）；
  // packed（token-major，K 区在 slot 内容偏移 0）下步进 = 16 字节（同 token 内相邻 16 维）
  const int64_t k_sstep = PACKED ? 16 : (int64_t)dg_stride;
  const int64_t k_soff = PACKED ? k_slot : (int64_t)16;   // (slot+mi) 的系数
  const int64_t k_goff = 4 * mg;

  // Q 作为 B 操作数（fp16，**无量化**）：lane L 的 4 个半字 ->
  //   B[dim = s*16+4*mg+p][head = mi]，即 Q[head=mi][dims s*16+4*mg .. +3]
  // lane 的 A 片段行 = mi -> 本 block 的 Q 头 hbase+mi；mi >= gqa 的为补齐项（结果废弃）
  const int qh = (mi < gqa) ? (hbase + mi) : hbase;
  const bf16_t* qrow = query + ((int64_t)seq * num_q_heads + qh) * LLAMA_FA_D;
  v2i32 qb[16];
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const bf16_t* qp = qrow + s * 16 + 4 * mg;
    __half2 h0 = __floats2half2_rn(__bfloat162float(qp[0]), __bfloat162float(qp[1]));
    __half2 h1 = __floats2half2_rn(__bfloat162float(qp[2]), __bfloat162float(qp[3]));
    qb[s][0] = *reinterpret_cast<int*>(&h0);
    qb[s][1] = *reinterpret_cast<int*>(&h1);
  }

  __shared__ __half Vs[LLAMA_FA_D][20];  // V16[dim][token]，行 40B -> 无 bank 冲突

  // ★ per 必须对齐到 16：tile 起点非 16 倍数时，K/V 的 16 字节向量载入会越过
  //   block 行尾（slot+15 >= bs），可能越界访问
  int per = (seq_len + pb - 1) / pb;
  per = ((per + 15) / 16) * 16;
  const int t0 = slice * per;
  const int t1 = min(seq_len, t0 + per);

  v4f32 acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = (v4f32){0.f, 0.f, 0.f, 0.f};
  float mrun = -1e30f, lrun = 0.f;

  for (int tt = t0; tt < t1; tt += 16) {
    const int blk = bt[tt / bs];
    const int slot = tt % bs;      // bs % 16 == 0 => tile 不跨块
    const int tok = tt + mi;
    const int inr = (tok < seq_len);
    // A 片段：lane L 的字节 b -> K[token = tt+mi][dim = s*16+4*mg+b]
    const signed char* kb = kcache + (int64_t)blk * k_blk + (int64_t)kvh * k_h +
                            (int64_t)(slot + mi) * k_soff + k_goff;
    v4f32 qk = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      const unsigned int raw =
          *reinterpret_cast<const unsigned int*>(kb + (int64_t)s * k_sstep);
      uint2 k16 = cvt4_i8_f16_d(inr ? raw : 0u);   // int8 -> fp16（无损）
      qk = mfma_f16_d(*reinterpret_cast<const v2i32*>(&k16), qb[s], qk);
    }

    float ks[4], vs_[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int sl = slot + 4 * mg + j;
      const bool ok = (tt + 4 * mg + j) < seq_len;
      const int64_t si = (int64_t)blk * ss_blk + (int64_t)sl * ss_slot +
                         (int64_t)kvh * ss_head;
      ks[j] = ok ? k_scale[si] * softmax_scale : 0.f;
      vs_[j] = ok ? v_scale[si] : 0.f;
    }
    float sc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = (tt + 4 * mg + j) < seq_len;
      sc[j] = ok ? qk[j] * ks[j] : -1e30f;
    }

    // online softmax：沿 token（4 寄存器 + shfl 16/32）
    float tmax = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffffffffffULL, tmax, 16));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffffffffffULL, tmax, 32));
    const float mn = fmaxf(mrun, tmax);
    const float alpha = __expf(mrun - mn);
    mrun = mn;
    float al[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      al[j] = __shfl_sync(0xffffffffffffffffULL, alpha, 4 * mg + j);
    if (al[0] != 1.f || al[1] != 1.f || al[2] != 1.f || al[3] != 1.f) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i][0] *= al[0];
        acc[i][1] *= al[1];
        acc[i][2] *= al[2];
        acc[i][3] *= al[3];
      }
    }
    float p[4], ssum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __expf(sc[j] - mrun);
      ssum += p[j];
    }
    ssum += __shfl_xor_sync(0xffffffffffffffffULL, ssum, 16);
    ssum += __shfl_xor_sync(0xffffffffffffffffULL, ssum, 32);
    lrun = lrun * alpha + ssum;

    // PV 的 A 片段：P16[head=mi][token=4*mg+j]（寄存器直出）
    v2i32 afrag;
    {
      __half2 h0 = __floats2half2_rn(p[0] * vs_[0], p[1] * vs_[1]);
      __half2 h1 = __floats2half2_rn(p[2] * vs_[2], p[3] * vs_[3]);
      afrag[0] = *reinterpret_cast<int*>(&h0);
      afrag[1] = *reinterpret_cast<int*>(&h1);
    }
    // V: int8 -> fp16（无损）
    if constexpr (PACKED) {
      // V 区：packed 布局下每 (head,slot) 的 V 是 256 连续字节（token-major，dim 连续）。
      // 每 lane 取 4 个 16 字节块（= 某 token 的 16 个 dim），无损转 fp16 后按
      // Vs[dim][token]（dim-major smem）散写 -> B 片段仍是 1×LDS.64（与 dim-major 版一致）
      const signed char* vb = vcache + (int64_t)blk * v_blk + (int64_t)kvh * v_h +
                              (int64_t)slot * v_slot + v_off;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int q = i * 64 + L;
        const int tk = q >> 4;        // tile 内 token
        const int d16 = (q & 15) << 4;  // 该 token 的 16 个 dim 起点
        const uint4 raw = *reinterpret_cast<const uint4*>(
            vb + (int64_t)tk * v_slot + d16);
        __half h[16];
        unsigned int w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          __half2 a = *reinterpret_cast<__half2*>(&w[c]);
          (void)a;
        }
        // 无损 int8->fp16（拆字节）
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int b0 = (int)(signed char)(w[c] & 0xff);
          const int b1 = (int)(signed char)((w[c] >> 8) & 0xff);
          const int b2 = (int)(signed char)((w[c] >> 16) & 0xff);
          const int b3 = (int)(signed char)(w[c] >> 24);
          h[4 * c + 0] = __float2half_rn((float)b0);
          h[4 * c + 1] = __float2half_rn((float)b1);
          h[4 * c + 2] = __float2half_rn((float)b2);
          h[4 * c + 3] = __float2half_rn((float)b3);
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) Vs[d16 + k][tk] = h[k];
      }
    } else {
      const signed char* vb =
          vcache + (int64_t)blk * v_blk + (int64_t)kvh * v_h + slot;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = i * 64 + L;
        const uint4 raw = *reinterpret_cast<const uint4*>(vb + (int64_t)d * bs);
        *reinterpret_cast<uint2*>(&Vs[d][0]) = cvt4_i8_f16_d(raw.x);
        *reinterpret_cast<uint2*>(&Vs[d][4]) = cvt4_i8_f16_d(raw.y);
        *reinterpret_cast<uint2*>(&Vs[d][8]) = cvt4_i8_f16_d(raw.z);
        *reinterpret_cast<uint2*>(&Vs[d][12]) = cvt4_i8_f16_d(raw.w);
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const v2i32 bfrag = *reinterpret_cast<const v2i32*>(&Vs[16 * i + mi][4 * mg]);
      acc[i] = mfma_f16_d(afrag, bfrag, acc[i]);
    }
    __syncthreads();
  }

  // 写 parts（未归一化，含 v_scale）+ meta(m,l)——与 combine 内核约定一致
  float lr[4], mr[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    lr[j] = __shfl_sync(0xffffffffffffffffULL, lrun, 4 * mg + j);
    mr[j] = __shfl_sync(0xffffffffffffffffULL, mrun, 4 * mg + j);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int h = hbase + 4 * mg + j;   // 全局 Q 头号
    if ((4 * mg + j) < gqa && h < num_q_heads) {
      const int64_t zh = (int64_t)seq * num_q_heads + h;
      float* po = parts + (zh * pb + slice) * LLAMA_FA_D;
#pragma unroll
      for (int i = 0; i < 16; ++i) po[16 * i + mi] = acc[i][j];
      if (mi == 0) meta[zh * pb + slice] = make_float2(mr[j], lr[j]);
    }
  }
}


// ============ 多组内核：256 线程 = NG 个 64-lane 组，每组一个查询 token ============
// 关键收益：K/V tile 只从 global 读一次并暂存 smem，被 NG 个查询 token 共享
//           => 每查询 token 的 K/V 访存量降到 1/NG（verify 步 NG=4 即 4×）
// 支持两种 cache 布局（PACKED=1 为 triton 的 [K|kscale|V|vscale]，0 为 dim-major 微基准布局）
// 因果：查询 token i 位置 = seq_len - qlen + i，仅对与查询区间重叠的 KV tile 施加掩码
template <bool PACKED, int NG>
__global__ void __launch_bounds__(64 * NG, 2)
llama_fa_mfma_multi_kernel(
    const signed char* __restrict__ kcache, const signed char* __restrict__ vcache,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const bf16_t* __restrict__ query, float* __restrict__ parts,
    float2* __restrict__ meta, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, const int max_blocks, const int bs,
    const int hkv, const int num_q_heads, const int pb, const int qlen,
    const int nqt, const int q_off, const float softmax_scale, const int64_t k_blk,
    const int64_t k_h, const int64_t v_blk, const int64_t v_h,
    const int64_t k_slot, const int64_t v_slot, const int64_t v_off,
    const int64_t ss_blk, const int64_t ss_slot, const int64_t ss_head) {
  const int tid = threadIdx.x;
  const int L = tid & 63;
  // blockIdx.y = seq * nqt + 查询 tile 序号（预填：一个 block 只吃 NG 个查询行）
  const int seq = blockIdx.y / nqt;
  const int qt = blockIdx.y % nqt;
  const int g = qt * NG + (tid >> 6);   // 全局查询行号
  const bool g_ok = (g < qlen);         // 末 tile 不满时仍参与 barrier，只跳过写出
  const int kvh = blockIdx.x % hkv;
  const int slice = blockIdx.x / hkv;
  const int mg = L / 16, mi = L % 16;
  const int seq_len = seq_lens[seq];
  const int* bt = block_table + (int64_t)seq * max_blocks;
  const int gqa = num_q_heads / hkv;
  const int hbase = kvh * gqa;
  const int qh = (mi < gqa) ? (hbase + mi) : hbase;
  // 本组查询 token 的绝对位置。q_off 由 host 侧决定（见 launcher）：
  //   0    => 引擎 seq_lens 已含当前 query 块（含 chunked-prefill 的块）
  //   qlen => 引擎 seq_lens 只含已缓存 KV（只读 query 块自身不在 cache 里）
  const int q_pos = seq_len - qlen + g + q_off;

  // Q 片段（B 操作数，fp16，无量化）
  // ★ 末尾不满的查询 tile 里 g 可能 >=query + (((int64_t)seq * qlen + gq) * num_q_heads + qh) * LLAMA_FA_D;预填即挂）。越界组的结果本来
  //   就被 g_ok 丢弃，读到的数据无所谓。
  const int gq = g_ok ? g : (qlen - 1);
    const bf16_t* qrow =
      query + (((int64_t)seq * qlen + gq) * num_q_heads + qh) * LLAMA_FA_D;
  // 空切片（t0 >= seq_len）跳过下面的 Q 片段载入：那是 16 次散列全局读的昂贵
  // 前导，而计算本来不会执行（tile 循环零次）。零值 parts/meta 仍照写（combine
  // 靠 l=0 忽略该切片）。pb 被图捕获固化为常量后短上下文会有大量空切片
  // （1k 上下文 + pb=128 => 每片仅 1 个 tile），这笔跳过很关键。
  int per_e = (seq_len + pb - 1) / pb;
  per_e = ((per_e + 15) / 16) * 16;
  const bool slice_empty = (slice * per_e) >= seq_len;
  v2i32 qb[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (slice_empty) break;
    const bf16_t* qp = qrow + i * 16 + 4 * mg;
    __half2 h0 = __floats2half2_rn(__bfloat162float(qp[0]), __bfloat162float(qp[1]));
    __half2 h1 = __floats2half2_rn(__bfloat162float(qp[2]), __bfloat162float(qp[3]));
    qb[i][0] = *reinterpret_cast<int*>(&h0);
    qb[i][1] = *reinterpret_cast<int*>(&h1);
  }

  __shared__ signed char Ks[16][260];        // K tile: [token][dim]（行 260B -> 无 bank 冲突）
  __shared__ __half Vs[LLAMA_FA_D][20];      // V tile: [dim][token]（行 40B -> 无 bank 冲突）

  int per = (seq_len + pb - 1) / pb;
  per = ((per + 15) / 16) * 16;
  // ★ 因果上界：本 tile 内最大 q_pos 之后的 KV 块用不到，不循环（预填省近一半）
  const int gmax = min(qlen - 1, qt * NG + NG - 1);
  const int cmax = min(seq_len, seq_len - qlen + gmax + 1 + q_off);
  const int t0 = slice * per;
  int t1 = min(seq_len, t0 + per);
  if (cmax > 0 && cmax < t1) t1 = ((cmax + 15) / 16) * 16;

  v4f32 acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = (v4f32){0.f, 0.f, 0.f, 0.f};
  float mrun = -1e30f, lrun = 0.f;

  for (int tt = t0; tt < t1; tt += 16) {
    const int blk = bt[tt / bs];
    const int slot = tt % bs;
    // ---------- 协作暂存 K tile（256B/lane 均分：每线程 1 个 16B 块）----------
    {
      // 256 个 16B 块 = 16 token × 16 dim-group；按线程数跨步覆盖
      for (int chunk = tid; chunk < 256; chunk += 64 * NG) {
        const int tout = chunk >> 4, dgt = chunk & 15;
        const signed char* src;
        if (PACKED)
          src = kcache + (int64_t)blk * k_blk + (int64_t)kvh * k_h +
                (int64_t)(slot + tout) * k_slot + dgt * 16;
        else
          src = kcache + (int64_t)blk * k_blk + (int64_t)kvh * k_h +
                (int64_t)dgt * (bs * 16) + (int64_t)(slot + tout) * 16;
        *reinterpret_cast<uint4*>(&Ks[tout][dgt * 16]) =
            *reinterpret_cast<const uint4*>(src);
      }
    }
    // ---------- 协作暂存 V tile（int8 -> fp16，dim-major smem）----------
    if (PACKED) {
      for (int chunk = tid; chunk < 256; chunk += 64 * NG) {
        const int tk = chunk >> 4, d16 = (chunk & 15) << 4;
        const signed char* src = vcache + (int64_t)blk * v_blk +
                                 (int64_t)kvh * v_h +
                                 (int64_t)(slot + tk) * v_slot + v_off + d16;
        const uint4 raw = *reinterpret_cast<const uint4*>(src);
        unsigned int w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int byte = (int)(signed char)((w[c] >> (8 * b)) & 0xff);
            Vs[d16 + 4 * c + b][tk] = __float2half_rn((float)byte);
          }
        }
      }
    } else {
      for (int d = tid; d < LLAMA_FA_D; d += 64 * NG) {
        const signed char* src = vcache + (int64_t)blk * v_blk +
                                 (int64_t)kvh * v_h + (int64_t)d * bs + slot;
        const uint4 raw = *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint2*>(&Vs[d][0]) = cvt4_i8_f16_d(raw.x);
        *reinterpret_cast<uint2*>(&Vs[d][4]) = cvt4_i8_f16_d(raw.y);
        *reinterpret_cast<uint2*>(&Vs[d][8]) = cvt4_i8_f16_d(raw.z);
        *reinterpret_cast<uint2*>(&Vs[d][12]) = cvt4_i8_f16_d(raw.w);
      }
    }
    __syncthreads();

    // ---------- KQ（A=K from smem, B=Q）----------
    v4f32 qk = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned int a = *reinterpret_cast<const unsigned int*>(&Ks[mi][4 * mg + i * 16]);
      uint2 k16 = cvt4_i8_f16_d(a);
      qk = mfma_f16_d(*reinterpret_cast<const v2i32*>(&k16), qb[i], qk);
    }
    // ---------- scale + 因果掩码 + online softmax ----------
    // ★ 整块都落在因果边界之外（tt > q_pos）时必须整块跳过：否则该块 16 个 token
    //   全被写成 -1e30，exp(0)=1 => 贡献假的 l+=16 与一堆 p*V，直接算错输出。
    //   微基准的 seq_len 恒为 16 对齐、q_pos 恒大于末块起点，故永不复现；引擎 seq_len
    //   任意，必然复现（表现为首调用正常、后续调用全错）。
    if (tt > q_pos) continue;
    const bool need_mask = (tt + 15) >= q_pos;      // 只有与查询区间重叠的 tile 需要
    float sc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int token = tt + 4 * mg + j;
      float v = (float)qk[j];
      if (need_mask && token > q_pos) v = -1e30f;
      // ★ 尾部越界 token 必须硬掩码：它们的 k_scale 被置 0 => score=0，而真分数为负时
      //   0 分会抢走 softmax 质量（系统性 10~70% 误差）。qlen=1 内核在 sc[j] 处已这么做。
      if (token >= seq_len) v = -1e30f;
      sc[j] = v;
    }
    float ks[4], vs_[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int sl = slot + 4 * mg + j;
      const int64_t si = (int64_t)blk * ss_blk + (int64_t)sl * ss_slot +
                         (int64_t)kvh * ss_head;
      ks[j] = (tt + 4 * mg + j) < seq_len ? k_scale[si] * softmax_scale : 0.f;
      vs_[j] = (tt + 4 * mg + j) < seq_len ? v_scale[si] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j] *= ks[j];
    float tmax = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffffffffffULL, tmax, 16));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffffffffffULL, tmax, 32));
    const float mn = fmaxf(mrun, tmax);
    const float alpha = __expf(mrun - mn);
    mrun = mn;
    float al[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      al[j] = __shfl_sync(0xffffffffffffffffULL, alpha, 4 * mg + j);
    if (al[0] != 1.f || al[1] != 1.f || al[2] != 1.f || al[3] != 1.f) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i][0] *= al[0];
        acc[i][1] *= al[1];
        acc[i][2] *= al[2];
        acc[i][3] *= al[3];
      }
    }
    float p[4], ssum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __expf(sc[j] - mrun);
      ssum += p[j];
    }
    ssum += __shfl_xor_sync(0xffffffffffffffffULL, ssum, 16);
    ssum += __shfl_xor_sync(0xffffffffffffffffULL, ssum, 32);
    lrun = lrun * alpha + ssum;
    v2i32 afrag;
    {
      __half2 h0 = __floats2half2_rn(p[0] * vs_[0], p[1] * vs_[1]);
      __half2 h1 = __floats2half2_rn(p[2] * vs_[2], p[3] * vs_[3]);
      afrag[0] = *reinterpret_cast<int*>(&h0);
      afrag[1] = *reinterpret_cast<int*>(&h1);
    }
    // ---------- PV ----------
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const v2i32 bfrag = *reinterpret_cast<const v2i32*>(&Vs[16 * i + mi][4 * mg]);
      acc[i] = mfma_f16_d(afrag, bfrag, acc[i]);
    }
    __syncthreads();
  }

  float lr[4], mr[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    lr[j] = __shfl_sync(0xffffffffffffffffULL, lrun, 4 * mg + j);
    mr[j] = __shfl_sync(0xffffffffffffffffULL, mrun, 4 * mg + j);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int h = hbase + 4 * mg + j;
    if (g_ok && (4 * mg + j) < gqa && h < num_q_heads) {
      const int64_t zh = ((int64_t)seq * qlen + g) * num_q_heads + h;
      float* po = parts + (zh * pb + slice) * LLAMA_FA_D;
#pragma unroll
      for (int i = 0; i < 16; ++i) po[16 * i + mi] = acc[i][j];
      if (mi == 0) meta[zh * pb + slice] = make_float2(mr[j], lr[j]);
    }
  }
}

void llama_fa_mfma_run(bf16_t* out_ptr, const bf16_t* q_ptr,
                       const int8_t* kc_ptr, const int8_t* vc_ptr,
                       const float* ks_ptr, const float* vs_ptr, const int* bt_ptr,
                       const int* sl_ptr, int num_seqs, int num_q_heads,
                       int num_kv_heads, int block_size, int max_blocks, int pb,
                       float scale, int64_t k_blk, int64_t k_h, int64_t v_blk,
                       int64_t v_h, int64_t k_slot, int64_t v_slot, int64_t v_off,
                       int64_t ss_blk, int64_t ss_slot, int64_t ss_head, bool packed,
                       int qlen, cudaStream_t stream) {
  auto opts_f = torch::TensorOptions().dtype(at::kFloat).device(torch::kCUDA);
  torch::Tensor parts = at::empty(
      {(int64_t)num_seqs * qlen * num_q_heads * pb * LLAMA_FA_D}, opts_f);
  torch::Tensor meta =
      at::empty({(int64_t)num_seqs * qlen * num_q_heads * pb * 2}, opts_f);
  dim3 grid(num_kv_heads * pb, num_seqs, 1);
  if (qlen >= 2) {   // 多组内核：NG = qlen 个查询 token 共享 K/V tile
    const int ng = (qlen >= 4) ? 4 : 2;   // NG=8/16 寄存器不可行（VGPR×线程 > 65536/CU）
    const int nqt = (qlen + ng - 1) / ng;              // 每序列查询 tile 数
    // 预填：查询 tile 维进 grid.y，K/V tile 在 NG 个查询行间共享（KV 流量/NG）
    grid = dim3(num_kv_heads * pb, num_seqs * nqt, 1);
    // 在线可切换的查询位置语义（免重启对比两种约定，便于定位/锁定）：
    //   /tmp/fa_int8_qoff 首字符 '1' => q_off = qlen；否则 0。无文件时读 env。
    int qoff_mode = 0;
    {
      FILE* f = fopen("/tmp/fa_int8_qoff", "r");
      if (f) {
        int c = getc(f);
        fclose(f);
        qoff_mode = (c == '1') ? 1 : 0;
      } else {
        const char* e = getenv("FA_INT8_QOFF");
        qoff_mode = (e && atoi(e) != 0) ? 1 : 0;
      }
    }
    const int q_off = qoff_mode ? qlen : 0;
#define MFMA_MULTI_LAUNCH(PK, N)                                              \
  llama_fa_mfma_multi_kernel<PK, N><<<grid, 64 * N, 0, stream>>>(              \
      (const signed char*)kc_ptr, (const signed char*)vc_ptr, ks_ptr, vs_ptr,  \
      q_ptr, parts.data_ptr<float>(),                                           \
      reinterpret_cast<float2*>(meta.data_ptr<float>()), bt_ptr, sl_ptr,        \
      max_blocks, block_size, num_kv_heads, num_q_heads, pb, qlen, nqt, q_off, \
      scale, k_blk, k_h, v_blk, v_h, k_slot, v_slot, v_off, ss_blk, ss_slot, \
      ss_head);
    if (ng == 4) {
      if (packed) MFMA_MULTI_LAUNCH(true, 4) else MFMA_MULTI_LAUNCH(false, 4)
    } else {
      if (packed) MFMA_MULTI_LAUNCH(true, 2) else MFMA_MULTI_LAUNCH(false, 2)
    }
#undef MFMA_MULTI_LAUNCH
    llama_fa_combine_kernel<<<num_seqs * qlen * num_q_heads, LLAMA_FA_D,
                              pb * sizeof(float2), stream>>>(
        parts.data_ptr<float>(),
        reinterpret_cast<const float2*>(meta.data_ptr<float>()), out_ptr, pb);
    return;
  }
  if (packed)
    llama_fa_mfma_kernel<true><<<grid, 64, 0, stream>>>(
        (const signed char*)kc_ptr, (const signed char*)vc_ptr, ks_ptr, vs_ptr,
        q_ptr, parts.data_ptr<float>(),
        reinterpret_cast<float2*>(meta.data_ptr<float>()), bt_ptr, sl_ptr, max_blocks,
        block_size, num_kv_heads, num_q_heads, pb, scale, k_blk, k_h, v_blk, v_h,
        k_slot, v_slot, v_off, ss_blk, ss_slot, ss_head);
  else
    llama_fa_mfma_kernel<false><<<grid, 64, 0, stream>>>(
        (const signed char*)kc_ptr, (const signed char*)vc_ptr, ks_ptr, vs_ptr,
        q_ptr, parts.data_ptr<float>(),
        reinterpret_cast<float2*>(meta.data_ptr<float>()), bt_ptr, sl_ptr, max_blocks,
        block_size, num_kv_heads, num_q_heads, pb, scale, k_blk, k_h, v_blk, v_h,
        k_slot, v_slot, v_off, ss_blk, ss_slot, ss_head);
  llama_fa_combine_kernel<<<num_seqs * num_q_heads, LLAMA_FA_D, pb * sizeof(float2),
                            stream>>>(
      parts.data_ptr<float>(),
      reinterpret_cast<const float2*>(meta.data_ptr<float>()), out_ptr, pb);
}

}  // namespace

// ---------------------------------------------------------------- host entry
// 与 csrc/rocm/attention_llama_fa.cu 的 host 路径逐字同构：同一个
// at::cuda::getCurrentCUDAStream() -> `<<<..., stream>>>` 启动方式。
// int8 KV: K [nb,hkv,D/16,bs,16] / V [nb,hkv,D,bs] / scales f32 [nb,bs,hkv]
void paged_attention_llama_fa_int8(torch::Tensor& out, torch::Tensor& query,
                                   torch::Tensor& key_cache,
                                   torch::Tensor& value_cache,
                                   torch::Tensor& k_scale,
                                   torch::Tensor& v_scale,
                                   torch::Tensor& block_tables,
                                   torch::Tensor& seq_lens,
                                   int64_t num_kv_heads, double scale,
                                   int64_t qlen, int64_t max_seq_len) {
  const int num_seqs = seq_lens.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int max_blocks = block_tables.size(1);
  TORCH_CHECK(head_size == LLAMA_FA_D, "llama_fa_int8: head_size must be 256");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kChar &&
                  value_cache.scalar_type() == at::kChar,
              "llama_fa_int8: query bf16 + int8 KV only");
  TORCH_CHECK(key_cache.dim() == 5 && key_cache.size(2) == LLAMA_FA_D / 16 &&
                  key_cache.size(4) == 16,
              "llama_fa_int8: key_cache must be [nb, hkv, D/16, bs, 16]");
  TORCH_CHECK(value_cache.dim() == 4 &&
                  value_cache.size(2) == LLAMA_FA_D &&
                  value_cache.size(3) == key_cache.size(3),
              "llama_fa_int8: value_cache must be [nb, hkv, D, bs]");
  const int block_size = key_cache.size(3);
  // 布局判定（自动，不依赖环境变量）：
  //   dim-major : K [D/16,bs,16] 稠密 + V [D,bs] 稠密   （rocm_attn 约定 / 微基准用）
  //   packed    : 每 (head,slot) 内容 = [K(D)|K_scale(4)|V(D)|V_scale(4)]（triton 后端用）
  const bool k_dim_major = key_cache.stride(3) == 16 && key_cache.stride(4) == 1;
  const bool v_dim_major = value_cache.stride(2) == (int64_t)block_size &&
                           value_cache.stride(3) == 1;
  const bool layout_dim_major = k_dim_major && v_dim_major;
  const bool k_packed = key_cache.stride(2) == 16 && key_cache.stride(4) == 1;
  const bool v_packed = value_cache.stride(3) == 1;
  TORCH_CHECK(layout_dim_major || (k_packed && v_packed),
              "llama_fa_int8: unsupported K/V layout (need dim-major or packed)");
  // qlen>=1 全走 MFMA：1=64线程内核；>=2=多组内核（NG=4 查询行共享 K/V tile）；
  // 预填 qlen>4 同样如此（grid.y 带查询 tile 维，且 pb 已强制为 1）
  const bool mfma_path = (qlen >= 1 && qlen <= 16384);
  if (!mfma_path) {
    // 其余 qlen 走旧的 VALU 分支：其寻址硬编码 dim-major，喂 packed 缓存会静默读错
    TORCH_CHECK(layout_dim_major,
                "llama_fa_int8: qlen>4 (VALU path) requires the dim-major "
                "[D/16,bs,16] cache layout");
    TORCH_CHECK(k_scale.is_contiguous() && v_scale.is_contiguous(),
                "llama_fa_int8: qlen>4 requires contiguous per-token-head scales");
  }
  TORCH_CHECK(k_scale.dim() == 3 && k_scale.size(0) == key_cache.size(0) &&
                  k_scale.size(1) == block_size &&
                  k_scale.size(2) == key_cache.size(1),
              "llama_fa_int8: k_scale must be contiguous f32 [nb, bs, hkv]");
  TORCH_CHECK(num_q_heads % 2 == 0, "llama_fa_int8: odd num_q_heads");
  const int gqa_ratio = num_q_heads / (int)num_kv_heads;
  TORCH_CHECK(gqa_ratio >= 2 && gqa_ratio % 2 == 0,
              "llama_fa_int8: gqa_ratio must be even");

  const int head_groups = num_q_heads / 2;
  const int ntiles_dst = num_seqs * head_groups;
  const int occupancy = 2;
  const int NB_FA = (qlen == 1) ? 128 : 64;
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const int blocks_per_wave = prop->multiProcessorCount * occupancy;
  const int ntiles_kv =
      (int)(((int64_t)max_blocks * block_size + NB_FA - 1) / NB_FA);
  int pb = std::min(occupancy, std::max(1, ntiles_kv));
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

  // 长上下文优化钩子：覆盖 llama.cpp 的 pb 启发式（并可按需打印实际 pb）
  if (const char* e = getenv("FA_INT8_FORCE_PB")) {
    pb = std::max(1, atoi(e));
  }
  if (getenv("FA_INT8_DBG_PB")) {
    fprintf(stderr, "int8 pb=%d ntiles_dst=%d ntiles_kv=%d sm=%d\n", pb,
            ntiles_dst, ntiles_kv, prop->multiProcessorCount);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(ntiles_dst, pb, 1);
  dim3 block(32, 8, 1);
  bf16_t* out_ptr = reinterpret_cast<bf16_t*>(out.data_ptr());
  float* parts_ptr = nullptr;
  float2* meta_ptr = nullptr;
  torch::Tensor parts, meta;
  if (pb > 1) {
    parts = at::empty({(int64_t)num_seqs * qlen * num_q_heads * pb * LLAMA_FA_D},
                      query.options().dtype(at::kFloat));
    meta = at::empty({(int64_t)num_seqs * qlen * num_q_heads * pb * 2},
                     query.options().dtype(at::kFloat));
    parts_ptr = parts.data_ptr<float>();
    meta_ptr = reinterpret_cast<float2*>(meta.data_ptr<float>());
  }
  const int dbg = 0;
  const int num_blocks = key_cache.size(0);
  const int8_t* kc_ptr = reinterpret_cast<const int8_t*>(key_cache.data_ptr());
  const int8_t* vc_ptr = reinterpret_cast<const int8_t*>(value_cache.data_ptr());
  const bf16_t* q_ptr = reinterpret_cast<const bf16_t*>(query.data_ptr());
  const float* ks_ptr = k_scale.data_ptr<float>();
  const float* vs_ptr = v_scale.data_ptr<float>();
  const int* bt_ptr = block_tables.data_ptr<int>();
  const int* sl_ptr = seq_lens.data_ptr<int>();

  // ---- MFMA 原生矩阵单元快速路径（qlen==1；FA_INT8_MFMA=0 可关闭）----
  {
    static int mfma_on = -1;
    if (mfma_on < 0) {
      const char* e = getenv("FA_INT8_MFMA");
      mfma_on = (e && atoi(e) == 0) ? 0 : 1;
    }
    if (mfma_on && mfma_path && num_q_heads <= MFMA_HQ && block_size % 16 == 0) {
      // MFMA 路径的 pb 策略与 VALU 内核不同：它一个 block 覆盖 16 个头，
      // 需要几百个 block 才能填满机器（实测 128k 最优 ~pb=200/2 头=400 blocks）。
      // 目标：每切片 ~512 token（32 个 tile），即 pb ≈ tokens/512，下限 24。
      // ★ 必须用**真实序列长度**算 pb：max_blocks 在引擎里是"每序列可用 block 数"
      //   （≈KV 池容量），拿它当 tokens 会把 pb 顶到上限 512 -> 每切片只剩 1~2 个
      //   tile，per-block 固定开销（Q 片段载入/parts/meta/combine）淹没 decode。
      const int tokens = (max_seq_len > 0)
                             ? (int)std::min<int64_t>(max_seq_len, (int64_t)1 << 30)
                             : max_blocks * block_size;
      // 目标：每切片 ~256 token（16 个 tile），即 pb ≈ tokens/256；
      // 下限 8（保底并行度），上限受 tokens/16（per≥16）与总块数 2048 约束
      // 实测最优切片数 ≈ sqrt(tokens)（2048->45 / 8192->91 / 32768->181 / 128k->362）
      const int grid_cap =
          std::max(1, 2048 / std::max(1, num_seqs * (int)num_kv_heads));
      // ★ CUDA Graph 兼容：pb 会在**图捕获**时被固化，重放不会再算一次。vLLM 捕获
      //   时传的是 max_model_len，若直接用它 => pb=362 被固化，而短上下文每片只剩
      //   1~2 个 tile，per-block 固定开销淹没 decode（即"和预填一样慢"的形态）。
      //   故把 pb 的输入封顶到 16384 => 无论捕获还是重放都得到同一个 pb=128。
      const int pb_tokens = std::min(tokens, 16384);
      int mpb = 1;
      while ((int64_t)mpb * mpb < (int64_t)pb_tokens && mpb < 512) ++mpb;
      mpb = std::min(512, std::max(8, mpb));
      mpb = std::min(mpb, std::min(std::max(1, (pb_tokens + 15) / 16), grid_cap));
      if (const char* e2 = getenv("FA_INT8_FORCE_PB")) mpb = std::max(1, atoi(e2));
      // ★ 预填（qlen>4）：不切 KV。否则 parts = nseq*qlen*hq*pb*D ≈ 10GB 直接爆分配；
      //   且切片会让 nqt 个查询 tile 各自重读整段 KV，抵消查询 tile 共享的收益。
      if (qlen > 4) mpb = 1;
      if (getenv("FA_INT8_DBG_PB")) {
        fprintf(stderr, "mfma pb=%d tokens=%d ntiles_kv=%d\n", mpb, tokens, ntiles_kv);
      }
      // K 视图 [nb,hkv,D/16,bs,16]、V 视图 [nb,hkv,D,bs]：块 stride 由打包 cache 决定
      const int64_t k_blk_p = (int64_t)key_cache.stride(0);
      const int64_t k_h_p = (int64_t)key_cache.stride(1);
      const int64_t v_blk_p = (int64_t)value_cache.stride(0);
      const int64_t v_h_p = (int64_t)value_cache.stride(1);
      // 自动探测：dim-major 时 K 的 stride(3)==16；包布局的 token 跨步 = 内容跨步
      const bool packed = !(k_dim_major && v_dim_major);
      // packed 布局：每 (head,slot) 内容 = [K(D) | K_scale(4B) | V(D) | V_scale(4B)]
      // packed: K 视图 [nb,hkv,D/16,bs,16]，stride(3) = 内容(slot)跨步
      const int64_t slot_stride =
          packed ? (int64_t)key_cache.stride(3) : (int64_t)16;
      // scale: packed 下是 buffer 内 f32 视图，跨步 (blk, slot, head)（元素单位）
      const int64_t ss_blk = packed ? (k_scale.stride(0)) : (int64_t)block_size * num_kv_heads;
      const int64_t ss_slot = packed ? (k_scale.stride(1)) : (int64_t)num_kv_heads;
      const int64_t ss_head = packed ? (k_scale.stride(2)) : (int64_t)1;
      llama_fa_mfma_run(out_ptr, q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr,
                        sl_ptr, num_seqs, num_q_heads, (int)num_kv_heads,
                        block_size, max_blocks, mpb, (float)scale, k_blk_p, k_h_p,
                        v_blk_p, v_h_p, slot_stride, slot_stride,
                        // 注意：packed 时调用方传入的 V 视图 data_ptr 已含 +（D+4）偏移
                        // （as_strided storage_offset），这里必须为 0，否则偏移量双算
                        (int64_t)0, ss_blk, ss_slot, ss_head, packed, (int)qlen,
                        stream);
      return;
    }
  }

  const int probe = getenv("FA_INT8_PROBE") ? atoi(getenv("FA_INT8_PROBE")) : 0;
  const bool i8dot = getenv("FA_INT8_I8DOT") && atoi(getenv("FA_INT8_I8DOT"));
  const bool hoist = getenv("FA_INT8_HOIST") && atoi(getenv("FA_INT8_HOIST"));
  if (getenv("FA_INT8_DBG_ATTRS")) {
    auto dump = [](const char* nm, const void* fn) {
      hipFuncAttributes at;
      hipFuncGetAttributes(&at, fn);
      fprintf(stderr, "attrs %-10s regs=%d smem=%zu maxthr=%d\n", nm,
              at.numRegs, at.sharedSizeBytes, at.maxThreadsPerBlock);
    };
    dump("conv", (const void*)llama_fa_tile_kernel_int8<1, 0, false, false>);
    dump("i8dot", (const void*)llama_fa_tile_kernel_int8<1, 0, true, false>);
    dump("hoist", (const void*)llama_fa_tile_kernel_int8<1, 0, false, true>);
  }
  if (getenv("FA_INT8_DBG_PB")) {
    fprintf(stderr, "int8 i8dot=%d\n", (int)i8dot);
  }
  if (getenv("FA_INT8_DBG_PB")) {
    fprintf(stderr, "int8 probe=%d\n", probe);
  }
  switch (qlen) {
    case 1:
     if (hoist) {
      llama_fa_tile_kernel_int8<1, 0, false, true><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
     }
     if (i8dot) {
      llama_fa_tile_kernel_int8<1, 0, true><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
     }
     if (probe == 1) {
      llama_fa_tile_kernel_int8<1, 1><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
     }
     if (probe == 2) {
      llama_fa_tile_kernel_int8<1, 2><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
     }
     if (probe == 3) {
      llama_fa_tile_kernel_int8<1, 3><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
     }
      llama_fa_tile_kernel_int8<1><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
    case 2:
      llama_fa_tile_kernel_int8<2><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
    case 4:
      llama_fa_tile_kernel_int8<4><<<grid, block, 0, stream>>>(
          q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, bt_ptr, sl_ptr, head_groups,
          gqa_ratio, (int)num_kv_heads, block_size, max_blocks, num_blocks,
          key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
          value_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
          k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
          v_scale.stride(2), (float)scale, pb, out_ptr, parts_ptr, meta_ptr,
          dbg);
      break;
    default:
      TORCH_CHECK(false, "llama_fa_int8: qlen must be 1, 2 or 4");
  }
  if (pb > 1) {
    llama_fa_combine_kernel<<<num_seqs * qlen * num_q_heads, LLAMA_FA_D,
                              pb * sizeof(float2), stream>>>(
        parts_ptr, meta_ptr, out_ptr, pb);
  }
}
