import torch
import vllm._rocm_C  # noqa: F401  registers torch.ops._rocm_C

torch.manual_seed(0)
dev = "cuda"

Hq, Hkv, D, BS = 24, 4, 256, 800
GQA = Hq // Hkv
scale = 1.0 / (D ** 0.5)


def run_case(seq_lens_list, qlen, seed=0):
    """seq_lens include the qlen new tokens; token j attends [0, L-qlen+j]."""
    g = torch.Generator(device=dev).manual_seed(seed)
    S = len(seq_lens_list)
    L = max(seq_lens_list)
    max_blocks = (L + BS - 1) // BS
    NB = max_blocks * S + 8
    kc_flat = torch.randn(NB, BS, Hkv, D, dtype=torch.bfloat16, device=dev, generator=g)
    vc_flat = torch.randn(NB, BS, Hkv, D, dtype=torch.bfloat16, device=dev, generator=g)
    # op contract (csrc/rocm/attention_llama_fa.cu):
    #   key_cache   [nb, hkv, D/8, bs, 8]  (head_dim split into 8-wide groups)
    #   value_cache [nb, hkv, D, bs]
    kc = kc_flat.reshape(NB, BS, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    vc = vc_flat.permute(0, 2, 3, 1).contiguous()
    q = torch.randn(S * qlen, Hq, D, dtype=torch.bfloat16, device=dev, generator=g)
    bt = torch.zeros(S, max_blocks, dtype=torch.int32, device=dev)
    for s in range(S):
        nb = (seq_lens_list[s] + BS - 1) // BS
        bt[s, :nb] = torch.arange(s * max_blocks, s * max_blocks + nb,
                                  dtype=torch.int32, device=dev)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)

    out = torch.empty(S * qlen, Hq * D, dtype=torch.bfloat16, device=dev)
    torch.ops._rocm_C.paged_attention_llama_fa(
        out, q, kc, vc, bt, seq_lens, Hkv, scale, qlen)

    # fp32 reference with analytic causal mask
    refs = []
    for s in range(S):
        n = seq_lens_list[s]
        idx = torch.arange(n, device=dev)
        blk = bt[s].long()
        K = kc_flat[blk[idx // BS], idx % BS].float().repeat_interleave(GQA, dim=1)
        V = vc_flat[blk[idx // BS], idx % BS].float().repeat_interleave(GQA, dim=1)
        for j in range(qlen):
            lim = n - qlen + j + 1
            att = torch.softmax(
                torch.einsum("hd,nhd->hn", q[s * qlen + j].float(), K[:lim]) * scale,
                dim=-1)
            refs.append(torch.einsum("hn,nhd->hd", att, V[:lim]).reshape(-1))
    ref = torch.stack(refs)

    err = (out.float() - ref).abs()
    rel = err / ref.abs().clamp_min(1e-3)
    print(f"qlen={qlen} seq_lens={seq_lens_list}: max_abs={err.max().item():.5f} "
          f"mean_abs={err.mean().item():.6f} max_rel={rel.max().item():.4f} "
          f"ref_absmax={ref.abs().max().item():.3f}")
    ok = err.max().item() < 0.03 * max(1.0, ref.abs().max().item())
    return ok


ok = True
ok &= run_case([1], 1)
ok &= run_case([5, 800, 2500], 1)
ok &= run_case([128, 128 * 17, 128 * 17 + 1], 1)
ok &= run_case([8192], 1)
ok &= run_case([333] * 32, 1)
ok &= run_case([2, 801, 2502], 2)          # spec qlen=2, page crossing
ok &= run_case([1024] * 8, 2)
ok &= run_case([4, 803, 2504], 4)          # MTP verify qlen=4
ok &= run_case([8192], 4)
ok &= run_case([4096] * 16, 4)
print("ALL OK" if ok else "MISMATCH!")
