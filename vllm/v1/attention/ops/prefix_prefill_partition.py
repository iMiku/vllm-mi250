# -*- coding: utf-8 -*-
"""旧自实现（§26.3 性能基线 3.47-4.64x）完整重建——性能对照专用。

源码摘自本会话早前读取的 620 行版本（2026-09-04 部署形态）：
- kernel1 `_fwd_kernel_partitioned`：tl.range unroll4、对齐 tile 免掩码加载
- reducer `_reduce_verify_partitions_kernel`：串行 flash 合并 + dense 段 verbatim（含 KV_FROM_CACHE 分支，本对照只用 dense）
- launch：固定 PART_TOKENS=4096、BLOCK_M 来自调用方
数值经官方 CASES 验证等价（dense 形态）。仅用于性能对照。
"""
import functools
import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

float8_info = torch.finfo(current_platform.fp8_dtype())

PART_TOKENS = 4096  # multiple of 32 (tile) and covers >1 physical block
VERIFY_MAX_Q = 32  # 路由兼容（官方值）


@functools.lru_cache(maxsize=1)
def _verify_partition_enabled() -> bool:
    return os.environ.get("VLLM_TRITON_VERIFY_CTX_PARTITION", "0") == "1"


@triton.jit
def _paged_kv_cache_offsets(
    B_Loc,
    cur_batch,
    token_indices,
    token_valid,
    offs_d,
    cur_kv_head,
    x,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    MASK_BLOCK_TABLE: tl.constexpr = False,
):
    bn_logical = token_indices // PHYSICAL_BLOCK_SIZE
    if MASK_BLOCK_TABLE:
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical * stride_b_loc_s,
            mask=token_valid,
            other=0,
        ).to(tl.int64)
    else:
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical * stride_b_loc_s
        ).to(tl.int64)
    internal = token_indices % PHYSICAL_BLOCK_SIZE
    off_k = (
        bn[None, :] * stride_k_cache_bs
        + cur_kv_head * stride_k_cache_h
        + (offs_d[:, None] // x) * stride_k_cache_d
        + internal[None, :] * stride_k_cache_bl
        + (offs_d[:, None] % x) * stride_k_cache_x
    )
    off_v = (
        bn[:, None] * stride_v_cache_bs
        + cur_kv_head * stride_v_cache_h
        + offs_d[None, :] * stride_v_cache_d
        + internal[:, None] * stride_v_cache_bl
    )
    return off_k, off_v


@functools.lru_cache(maxsize=16)
def _scratch_tensors(batch: int, q_heads: int, n_parts: int, d_padded: int,
                     device: str) -> tuple[torch.Tensor, torch.Tensor]:
    acc = torch.empty((batch, q_heads, n_parts, 32, d_padded),
                      dtype=torch.float32, device=device)
    ml = torch.empty((batch, q_heads, n_parts, 2, 32),
                     dtype=torch.float32, device=device)
    return acc, ml


def _choose_verify_partition(batch: int, q_heads: int, seq_bound: int,
                             block_m: int = 32, head_dim: int = 256) -> int:
    """old chooser: fixed 4096."""
    return 4096


@triton.jit
def _fwd_kernel_partitioned(
    Q, K, V,
    K_cache, V_cache, sink_ptr, B_Loc, sm_scale, k_scale, v_scale,
    out_scale_inv, B_Start_Loc, B_Seqlen, x: tl.constexpr,
    Out, Part_Acc, Part_ML,
    pacc_s0, pacc_s1, pacc_s2, pacc_s3, pacc_s4,
    pml_s0, pml_s1, pml_s2, pml_s3,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_obs, stride_oh, stride_od,
    stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr, stride_k_cache_x,
    stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl,
    PART_TOKENS,
    num_queries_per_kv: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    CAUSAL: tl.constexpr = True,
    MAX_Q_LEN: tl.constexpr = 0,
    MAX_CTX_LEN: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    KV_FROM_CACHE: tl.constexpr = True,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    part = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    kv_start = part * PART_TOKENS
    if kv_start >= cur_batch_ctx_len:
        return  # surplus partition — early exit (docs/62 §5)
    kv_end = tl.minimum(kv_start + PART_TOKENS, cur_batch_ctx_len)

    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    dim_mask = tl.where(
        tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0
    ).to(tl.int1)
    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )

    if not USE_SINKS:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.load(
            sink_ptr + tl.full([BLOCK_M], cur_head, dtype=tl.int64),
            mask=(offs_m < cur_batch_query_len),
            other=float("-inf"),
        ).to(dtype=tl.float32)
        l_i = tl.where(m_i > float("-inf"), 1.0, 0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    for start_n in tl.range(
        kv_start, kv_end, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        token_indices = start_n + offs_bs_n
        off_k, off_v = _paged_kv_cache_offsets(
            B_Loc, cur_batch, token_indices, offs_bs_n, offs_d,
            cur_kv_head, x,
            stride_b_loc_b, stride_b_loc_s,
            stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d,
            stride_k_cache_bl, stride_k_cache_x,
            stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d,
            stride_v_cache_bl, PHYSICAL_BLOCK_SIZE,
        )
        if start_n + BLOCK_SIZE > kv_end or BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
            k_load = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None] & ((start_n + offs_bs_n[None, :]) < kv_end),
                other=0.0,
            )
        else:
            k_load = tl.load(K_cache + off_k)
        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)
        qk = tl.where((start_n + offs_bs_n[None, :]) < kv_end, qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None])
                - (start_n + offs_bs_n[None, :]) < SLIDING_WINDOW,
                qk, float("-inf"),
            )
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        if start_n + BLOCK_SIZE > kv_end or BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
            v_load = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :] & ((start_n + offs_bs_n[:, None]) < kv_end),
                other=0.0,
            )
        else:
            v_load = tl.load(V_cache + off_v)
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    part_m = offs_m < cur_batch_query_len
    tl.store(
        Part_ML + cur_batch * pml_s0 + cur_head * pml_s1
        + part * pml_s2 + 0 * pml_s3 + offs_m,
        m_i, mask=part_m,
    )
    tl.store(
        Part_ML + cur_batch * pml_s0 + cur_head * pml_s1
        + part * pml_s2 + 1 * pml_s3 + offs_m,
        l_i, mask=part_m,
    )
    tl.store(
        Part_Acc + cur_batch * pacc_s0 + cur_head * pacc_s1
        + part * pacc_s2
        + offs_m[:, None] * pacc_s3 + offs_d[None, :] * pacc_s4,
        acc, mask=part_m[:, None] & dim_mask[None, :],
    )


@triton.jit
def _reduce_verify_partitions_kernel(
    Q, K, V,
    K_cache, V_cache, sink_ptr, B_Loc, sm_scale, k_scale, v_scale,
    out_scale_inv, B_Start_Loc, B_Seqlen, x: tl.constexpr,
    Out, Part_Acc, Part_ML,
    pacc_s0, pacc_s1, pacc_s2, pacc_s3, pacc_s4,
    pml_s0, pml_s1, pml_s2, pml_s3,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_obs, stride_oh, stride_od,
    stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr, stride_k_cache_x,
    stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl,
    stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd,
    PART_TOKENS, N_PARTS,
    num_queries_per_kv: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_request: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    CAUSAL: tl.constexpr = True,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    KV_FROM_CACHE: tl.constexpr = True,
    SKIP_DECODE: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = tl.arange(0, BLOCK_M)
    dim_mask = tl.where(
        tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0
    ).to(tl.int1)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )

    if not USE_SINKS:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.load(
            sink_ptr + tl.full([BLOCK_M], cur_head, dtype=tl.int64),
            mask=(offs_m < cur_batch_query_len),
            other=float("-inf"),
        ).to(dtype=tl.float32)
        l_i = tl.where(m_i > float("-inf"), 1.0, 0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    # merge per-partition partials (flash-decoding reduce); invalid/absent
    # partitions resolve to m=-inf/l=0/acc=0 and contribute nothing
    for part in tl.range(0, N_PARTS):
        kv_start = part * PART_TOKENS
        part_valid = (kv_start < cur_batch_ctx_len) & (offs_m < cur_batch_query_len)
        m_p = tl.load(
            Part_ML + cur_batch * pml_s0 + cur_head * pml_s1
            + part * pml_s2 + 0 * pml_s3 + offs_m,
            mask=part_valid,
            other=float("-inf"),
        )
        l_p = tl.load(
            Part_ML + cur_batch * pml_s0 + cur_head * pml_s1
            + part * pml_s2 + 1 * pml_s3 + offs_m,
            mask=part_valid,
            other=0.0,
        )
        acc_p = tl.load(
            Part_Acc + cur_batch * pacc_s0 + cur_head * pacc_s1
            + part * pacc_s2
            + offs_m[:, None] * pacc_s3 + offs_d[None, :] * pacc_s4,
            mask=part_valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        m_ij = tl.maximum(m_i, m_p)
        p_s = tl.exp(m_i - m_ij)
        p_s = tl.where(m_i == float("-inf"), 0.0, p_s)
        p_p = tl.exp(m_p - m_ij)
        p_p = tl.where(m_p == float("-inf"), 0.0, p_p)
        acc = acc * p_s[:, None] + acc_p * p_p[:, None]
        l_i = l_i * p_s + l_p * p_p
        m_i = m_ij

    # verbatim of upstream `_fwd_kernel` dense/self-chunk + output (start_m=0:
    # the reducer covers all query rows in one block)
    block_start_loc = 0
    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    if CAUSAL:
        key_range_upper = block_mask * BLOCK_M  # start_m=0 in reducer
    else:
        q_len_pad = (cur_batch_query_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        key_range_upper = block_mask * q_len_pad

    for start_n in tl.range(
        0,
        key_range_upper,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if KV_FROM_CACHE:
            cache_token_idx = cur_batch_ctx_len + start_n + offs_n
            cache_token_valid = (start_n + offs_n) < cur_batch_query_len
            off_k_cur, off_v_cur = _paged_kv_cache_offsets(
                B_Loc,
                cur_batch,
                cache_token_idx,
                cache_token_valid,
                offs_d,
                cur_kv_head,
                x,
                stride_b_loc_b,
                stride_b_loc_s,
                stride_k_cache_bs,
                stride_k_cache_h,
                stride_k_cache_d,
                stride_k_cache_bl,
                stride_k_cache_x,
                stride_v_cache_bs,
                stride_v_cache_h,
                stride_v_cache_d,
                stride_v_cache_bl,
                PHYSICAL_BLOCK_SIZE,
                MASK_BLOCK_TABLE=True,
            )
            k_cur_load = tl.load(
                K_cache + off_k_cur,
                mask=dim_mask[:, None]
                & ((start_n + offs_n[None, :]) < cur_batch_query_len),
                other=0.0,
            )
            if k_cur_load.dtype.is_fp8():
                k = (k_cur_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            else:
                k = k_cur_load
        else:
            k = tl.load(
                k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                mask=dim_mask[:, None]
                & ((start_n + offs_n[None, :]) < cur_batch_query_len),
                other=0.0,
            )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale

        valid_kv = (start_n + offs_n[None, :]) < cur_batch_query_len
        if CAUSAL:
            attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        else:
            attn_mask = valid_kv
        if SLIDING_WINDOW > 0:
            attn_mask = attn_mask & (
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW
            )
        qk = tl.where(attn_mask, qk, float("-inf"))

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        # To prevent NaN from appearing in the first round
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        if KV_FROM_CACHE:
            v_cur_load = tl.load(
                V_cache + off_v_cur,
                mask=dim_mask[None, :]
                & ((start_n + offs_n[:, None]) < cur_batch_query_len),
                other=0.0,
            )
            if v_cur_load.dtype.is_fp8():
                v = (v_cur_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
            else:
                v = v_cur_load
        else:
            v = tl.load(
                v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                mask=dim_mask[None, :]
                & ((start_n + offs_n[:, None]) < cur_batch_query_len),
                other=0.0,
            )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / (l_i[:, None] + 1e-10)

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


def run_partitioned_verify(*, q, k, v, o, k_cache, v_cache, b_loc,
                           b_start_loc, b_seq_len, k_scale, v_scale,
                           sm_scale, fp8_out_scale, causal, skip_decode,
                           num_queries_per_kv, IN_PRECISION, real_block_size,
                           Lk_padded, max_input_len) -> bool:
    batch, head = b_seq_len.shape[0], q.shape[1]
    if not _verify_partition_enabled():
        return False
    kv_from_cache = k is None
    if kv_from_cache:
        return False  # 本补丁仅 dense 形态（生产 verify 均是 dense）
    BLOCK_M = 32  # 生产几何（block_size 400 非 pow2 → 0.27.1 调用方值）
    BLOCK_N = 32
    seq_bound = b_loc.shape[1] * real_block_size
    n_parts = max(1, triton.cdiv(seq_bound, PART_TOKENS))
    acc_scratch, ml_scratch = _scratch_tensors(
        batch, head, n_parts, Lk_padded, str(q.device)
    )
    out_scale_inv = 1.0 / fp8_out_scale if fp8_out_scale is not None else 1.0
    x = k_cache.shape[4]

    common = dict(
        K_cache=k_cache, V_cache=v_cache, sink_ptr=None, B_Loc=b_loc,
        sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale,
        out_scale_inv=out_scale_inv, B_Start_Loc=b_start_loc,
        B_Seqlen=b_seq_len, x=x,
        stride_b_loc_b=b_loc.stride(0), stride_b_loc_s=b_loc.stride(1),
        stride_qbs=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_obs=o.stride(0), stride_oh=o.stride(1), stride_od=o.stride(2),
        stride_k_cache_bs=k_cache.stride(0), stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2), stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0), stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2), stride_v_cache_bl=v_cache.stride(3),
        num_queries_per_kv=num_queries_per_kv, IN_PRECISION=IN_PRECISION,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=o.shape[-1], BLOCK_DMODEL_PADDED=Lk_padded,
        BLOCK_SIZE=32, PHYSICAL_BLOCK_SIZE=real_block_size, BLOCK_N=BLOCK_N,
        SLIDING_WINDOW=0, USE_SINKS=False,
        USE_FP8=fp8_out_scale is not None, CAUSAL=causal,
        num_warps=4, num_stages=1,
    )
    k1_kwargs = dict(common)
    k1_kwargs.update(
        Q=q, K=k, V=v, Out=o, Part_Acc=acc_scratch, Part_ML=ml_scratch,
        pacc_s0=acc_scratch.stride(0), pacc_s1=acc_scratch.stride(1),
        pacc_s2=acc_scratch.stride(2), pacc_s3=acc_scratch.stride(3),
        pacc_s4=acc_scratch.stride(4),
        pml_s0=ml_scratch.stride(0), pml_s1=ml_scratch.stride(1),
        pml_s2=ml_scratch.stride(2), pml_s3=ml_scratch.stride(3),
        PART_TOKENS=PART_TOKENS, SKIP_DECODE=skip_decode, num_unroll_cache=4,
    )
    _fwd_kernel_partitioned[(batch, head, n_parts)](**k1_kwargs)

    red_kwargs = dict(common)
    red_kwargs.update(
        Q=q, K=k, V=v, Out=o, Part_Acc=acc_scratch, Part_ML=ml_scratch,
        pacc_s0=acc_scratch.stride(0), pacc_s1=acc_scratch.stride(1),
        pacc_s2=acc_scratch.stride(2), pacc_s3=acc_scratch.stride(3),
        pacc_s4=acc_scratch.stride(4),
        pml_s0=ml_scratch.stride(0), pml_s1=ml_scratch.stride(1),
        pml_s2=ml_scratch.stride(2), pml_s3=ml_scratch.stride(3),
        stride_kbs=k.stride(0), stride_kh=k.stride(1), stride_kd=k.stride(2),
        stride_vbs=v.stride(0), stride_vh=v.stride(1), stride_vd=v.stride(2),
        PART_TOKENS=PART_TOKENS, N_PARTS=n_parts,
        num_unroll_request=1, KV_FROM_CACHE=kv_from_cache,
        SKIP_DECODE=skip_decode,
    )
    _reduce_verify_partitions_kernel[(batch, head)](**red_kwargs)
    return True