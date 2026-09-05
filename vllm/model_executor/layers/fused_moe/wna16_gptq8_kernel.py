"""8-pack block GEMM Triton kernel for W4A16 MoE (prefill-focused).

Consumes repacked weights (E, N, K//8) int32 (wna16_repack.repack_2pack_to_8pack)
plus the unchanged (E, N, K//group_size) scale. Same calling convention as the
production fused_moe_kernel_gptq_awq (sorted_token_ids / expert_ids /
num_tokens_post_padded / C strided (M, topk, N) / MUL_ROUTED_WEIGHT for w2).

Correctness rules from gfx90a ablation (see §24.6): BLOCK_M=64 blocks,
no k-range mask when K % BLOCK_K == 0, advancing pointer math only.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def gptq8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    ALIGN_M: tl.constexpr,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # Newer triton (>=3.6) requires tl.dtype, not torch.dtype, in tl.zeros/tl.dot.
    if compute_type == torch.bfloat16:
        compute_type = tl.bfloat16
    elif compute_type == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    npp = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= npp:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    offs_token = tl.load(sorted_token_ids_ptr + rm).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    if tl.min(offs_token) >= num_valid_tokens:
        tl.store(c_ptr + stride_cm * offs_token[:, None] + stride_cn * rn[None, :],
                 tl.zeros((BLOCK_M, BLOCK_N), dtype=compute_type),
                 mask=token_mask[:, None] & (rn[None, :] < N))
        return
    expert = tl.load(expert_ids_ptr + (pid_m * BLOCK_M) // ALIGN_M).to(tl.int64)
    if expert == -1:
        tl.store(c_ptr + stride_cm * offs_token[:, None] + stride_cn * rn[None, :],
                 tl.zeros((BLOCK_M, BLOCK_N), dtype=compute_type),
                 mask=token_mask[:, None] & (rn[None, :] < N))
        return
    offs_bn = rn % N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + rk[None, :] * stride_ak
    b_ptrs = (b_ptr + expert * stride_be + (rk[:, None] // 8) * stride_bk
              + offs_bn[None, :] * stride_bn)
    bs_ptrs = (b_scale_ptr + expert * stride_bse + offs_bn[None, :] * stride_bsn
               + (rk[:, None] // group_size) * stride_bsk)
    shifter = (rk[:, None] % 8) * 4
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if block_k_diviable:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            sc = tl.load(bs_ptrs).to(tl.float32)
        else:
            km = rk[:, None] < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (rk[None, :] < K - k * BLOCK_K), other=0.0)
            b = tl.load(b_ptrs, mask=km, other=0.0)
            sc = tl.load(bs_ptrs, mask=km, other=0.0).to(tl.float32)
        nib = (b >> shifter) & 0xF
        b = ((nib.to(tl.float32) - 8.0) * sc).to(compute_type)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk
        bs_ptrs += (BLOCK_K // group_size) * stride_bsk
    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        acc = acc * w[:, None]
    tl.store(c_ptr + stride_cm * offs_token[:, None] + stride_cn * rn[None, :],
             acc.to(compute_type), mask=token_mask[:, None] & (rn[None, :] < N))


@torch.compiler.disable
def run_gptq8(
    A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
    num_tokens_post_padded, mul_routed_weight, top_k, compute_type,
    block_shape, block_size_m_align,
    block_m=64, block_n=64, block_k=64, num_warps=4,
):
    """B arrives as int32 (E, N, K//8); block_shape[1] = effective group size."""
    N = C.size(-1)
    K = A.size(1)
    M = A.size(0)
    num_valid_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < block_size_m_align:
        EM = min(EM, M * top_k * block_size_m_align)
    grid = (triton.cdiv(EM, block_m), triton.cdiv(N, block_n))
    grid = (triton.cdiv(EM, block_m), triton.cdiv(N, block_n))
    if os.environ.get("VLLM_W4A16_GPTQ8_DEBUG", "0") == "1":
        print(f"[gptq8dbg] A={tuple(A.shape)} B={tuple(B.shape)} "
              f"C={tuple(C.shape)} scale={tuple(B_scale.shape)} "
              f"st={tuple(sorted_token_ids.shape)} "
              f"npp={int(num_tokens_post_padded.cpu().item())} "
              f"A_rows={M} topk={top_k} EM={EM} gs={block_shape[1]} "
              f"st_min={int(sorted_token_ids.min())} "
              f"st_max={int(sorted_token_ids.max())} "
              f"st_head={sorted_token_ids[:8].cpu().tolist()} "
              f"st_tail={sorted_token_ids[-8:].cpu().tolist()} "
              f"eids={tuple(expert_ids.shape)} "
              f"eids_max={int(expert_ids.max())}", flush=True)

    gptq8_gemm_kernel[grid](
        A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        block_size_m_align,
        K % block_k == 0,
        block_shape[1],
        block_m, block_n, block_k,
        mul_routed_weight, top_k,
        compute_type,
        num_warps=num_warps,
    )