"""Offline repack: vLLM gptq 2-pack (E, N, K//2) uint8 -> 8-pack (E, N, K//8) int32.

8-pack layout (marlin-style, matches wna16_gptq8 kernel / §23.1.2 micro):
  word w covers k = 8w..8w+7; int4_i sits in bits [4i, 4i+4) of int32 word.
2-pack input: byte b covers k = 2b..2b+1; lo nibble = k even, hi nibble = k odd.
"""
import torch


def repack_2pack_to_8pack(w_2pack: torch.Tensor) -> torch.Tensor:
    """2-pack (E, N, K//2) -> 8-pack (E, N, K//8) int32, low nibble first.

    Tolerates int32 storage (viewed as uint8 bytes) from checkpoints whose
    loader kept the K-first packed words; byte layout is identical.
    """
    import os
    _dbg = os.environ.get("VLLM_W4A16_GPTQ8_DEBUG", "0") == "1"
    if _dbg:
        print(f"[repack] in: dtype={w_2pack.dtype} shape={tuple(w_2pack.shape)} "
              f"contig={w_2pack.is_contiguous()}", flush=True)
    if w_2pack.dtype == torch.int32:
        # loader fast path stores the checkpoint 8-pack (E, N, K//8) int32
        # directly — already in gptq8 kernel layout, nothing to do.
        return w_2pack.contiguous()
    assert w_2pack.dtype == torch.uint8, f"unexpected dtype {w_2pack.dtype}"
    w8 = w_2pack
    assert w8.dim() == 3, f"w8 dim {w8.dim()}"
    E, N, K2 = w8.shape
    K = K2 * 2
    assert K % 8 == 0, f"K={K}"
    lo = (w8 & 0x0F).to(torch.uint8)  # k even
    hi = ((w8 >> 4) & 0x0F).to(torch.uint8)  # k odd
    w4 = torch.stack([lo, hi], dim=-1).reshape(E, N, K).to(torch.int32)  # (E,N,K)
    w4 = w4.reshape(E, N, K // 8, 8)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          device=w4.device, dtype=torch.int32)
    packed = (w4 << shifts).sum(dim=-1).to(torch.int32)  # (E,N,K/8) int32
    return packed.contiguous()


def dequant_8pack(packed: torch.Tensor, scale: torch.Tensor,
                  group_size: int) -> torch.Tensor:
    """(E,N,K/8) int32 + (E,N,K/G) -> (E,N,K) fp16."""
    E, N, K8 = packed.shape
    K = K8 * 8
    sph = torch.arange(0, 8, device=packed.device, dtype=torch.int32) * 4
    nib = (packed.unsqueeze(-1) >> sph) & 0xF  # (E,N,K/8,8)
    w = nib.reshape(E, N, K).float()
    sc = scale.repeat_interleave(group_size, dim=-1).float()
    return ((w - 8.0) * sc).half()


def dequant_2pack(w_2pack: torch.Tensor, scale: torch.Tensor,
                  group_size: int) -> torch.Tensor:
    E, N, K2 = w_2pack.shape
    K = K2 * 2
    lo = (w_2pack & 0x0F).float()
    hi = ((w_2pack >> 4) & 0x0F).float()
    w = torch.stack([lo, hi], dim=-1).reshape(E, N, K)
    sc = scale.repeat_interleave(group_size, dim=-1).float()
    return ((w - 8.0) * sc).half()


def verify(seed=0):
    torch.manual_seed(seed)
    for E, N, K in [(512, 160, 2560), (512, 2560, 80)]:
        G = 32 if K == 2560 else 16
        w2 = torch.randint(0, 16, (E, N, K), dtype=torch.uint8)
        even = w2[..., 0::2]
        odd = w2[..., 1::2]
        packed2 = (even | (odd << 4).to(torch.uint8)).contiguous()  # (E,N,K/2)
        sc = (torch.rand(E, N, K // G) * 0.1 + 0.05).half()
        ref = dequant_2pack(packed2, sc, G)
        packed8 = repack_2pack_to_8pack(packed2)
        out = dequant_8pack(packed8, sc, G)
        err = (ref.float() - out.float()).abs().max().item()
        print(f"E={E} N={N} K={K}: repack maxerr={err:.6f} shape={tuple(packed8.shape)}")
        assert err < 1e-6, "repack mismatch"


if __name__ == "__main__":
    verify()