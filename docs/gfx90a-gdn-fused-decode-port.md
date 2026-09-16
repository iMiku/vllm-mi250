# Porting a CUDA-only op to gfx90a: the fused GDN decode kernel

**Symptom.** `VLLM_GDN_DECODE_KERNEL=cuda` — the only value that makes
`enable_fused_gdn_decode` true — aborts at startup with
`ValueError: ... torch.ops._C.fused_gdn_decode_post_conv_mtp is not built`.
The 48 GDN layers then silently fall back to the in-tree triton decode kernel
(the engine logs `GDN decode kernel: triton`).

**Root cause (build-time, not hardware).** The implementation has always been in the
tree — `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`, 597 lines, and it already
includes `cuda_compat.h` — and it is registered under
`STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)`, so `torch.ops._C.<op>` is the right lookup.
What excluded it was the macro: `#ifdef VLLM_ENABLE_FUSED_GDN_DECODE` is defined only
inside the CUDA branch, whose arch intersection list is
`"8.0;8.6;8.9;9.0a;10.0f;12.0f"` — note that **`9.0a` here is CUDA `sm_90a`, not
`gfx90a`**. On ROCm neither the source nor the macro reached the build, so the op was
absent from every namespace (`_C`, `libtorch_stable`, `_rocm_C` all reported false).

**The port.** Mirror the HIP branch that already existed for KDA in the same CMakeLists:

1. `CMakeLists.txt`: in the HIP section, add the source to `VLLM_STABLE_EXT_SRC` with an
   arch filter of `gfx90a|gfx942|gfx950`, and add
   `target_compile_definitions(_C_stable_libtorch PRIVATE VLLM_ENABLE_FUSED_GDN_DECODE=1)`.
2. Three mechanical CUDA→HIP fixes in the kernel, each behind `#if defined(USE_ROCM)` so
   the CUDA path is untouched:

   | Compile error | Cause | Fix |
   |---|---|---|
   | `invalid input constraint 'l' in asm`, `__cvta_generic_to_shared` undeclared | `cp.async.cg.shared.global` is SM80+ PTX | synchronous 16 B copy on ROCm; `commit` becomes a no-op and `wait_all` a `__syncthreads()` |
   | `__floats2bfloat162_rn` undeclared | CUDA-only intrinsic | `__float22bfloat162_rn(make_float2(x, y))` |
   | `mask must be a 64-bit integer` | wave64: `__shfl_*_sync` with 32-bit masks | masks become `0xffffffffffffffffULL` |

**Verification** (no server needed): the in-tree reference test
`tests/kernels/test_fused_gdn_post_conv.py`, driven with the model's real dimensions
(H=16, HV=48, K=V=128) at L=1 (decode) and L=16 (MTP), across both `apply_l2norm` and
`output_g_exp` — **8/8 PASS**, plus `test_fused_post_conv_l0`.

**Lesson.** In vLLM, "this op does not exist on ROCm" must be separated into "the hardware
cannot do it" versus "the build excluded it". This case was the latter: the source was
present, the reference test was present, and only a CMake branch plus three HIP-compat
lines stood between the card and the kernel. The same shape of exclusion produced the
earlier AITER findings (`on_mi3xx()` as the single upstream predicate), so the check to
run first is always *what gated the build or the registration*, not *what the ISA allows*.

## Outcome: the port is correct but worth ~0% (measured)

After the op existed and `VLLM_GDN_DECODE_KERNEL=cuda` was accepted (the engine logs
`GDN decode kernel: cuda`), a like-for-like measurement gave:

| | before (triton decode) | after (fused decode) |
|---|---|---|
| decode @9k, 500 tokens | 76.4 tok/s | **77.1 tok/s** (+0.9%, inside noise) |
| prefill @2k | 1665 tok/s | 1651 tok/s (flat) |
| quality | correct | correct |

**So GDN is not the bottleneck**, and the earlier inference "the per-token-scaling part of
decode must be GDN" was wrong — the roofline decomposition only ever excluded *weight
bandwidth*; naming a module from it was a guess. GDN decode is a small per-layer state
update (48 layers x ~1.5 MB of state ~ 72 MB/step ~ 60 us at 1.2 TB/s): neither
bandwidth-bound nor many kernels, so fusing it could not pay.

Keep the port (it is correct and costs nothing), but do not expect a speedup from it, and
do not spend further effort on the GDN path: CuteDSL's GDN is hard-gated to Blackwell, the
AITER GDN triton kernels are already in use, and this fused op is now enabled and measured.
