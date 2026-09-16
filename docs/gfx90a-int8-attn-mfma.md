# gfx90a: INT8-KV attention (MFMA) + the engine-level wins it depends on

Measured on 4×MI250X (8 GCD, gfx90a), Qwen3.8-27B-W8A8, TP2, MTP n3, ROCm 7.2.
This note records the custom attention kernel, the four bugs that only the engine
exposed, and the three silently-broken engine settings worth 3.1× decode.

## 1. What the kernel is

`csrc/rocm/attention_llama_fa_int8.cu` — `paged_attention_llama_fa_int8`, serving
`qlen ∈ [1, 16384]` on the **int8 per-token-head** KV cache:

* **KQ**: f16 MFMA `16x16x16` with **A=K, B=Q** (Q untouched — bf16→fp16 is lossless;
  no quantization is introduced inside the kernel). Swapping the operands makes
  `C[token][head]`, which puts
  - the softmax row reduction on 4 registers + 2 shuffle rounds,
  - **P directly in the PV A-fragment layout** (no P smem transpose, one less barrier).
* **PV**: f16 MFMA, `A=P16` (register-direct), `B=V16` staged in smem as `[dim][token]`
  (row stride 20 halves = 40 B ⇒ conflict-free), `C=out[head][dim]`.
  `head = 4*mg+j` differs from P's `mi` ⇒ alpha/l are exchanged with 4 `__shfl_sync`.
* **qlen == 1** → 64-thread kernel; **qlen ≥ 2** → `llama_fa_mfma_multi_kernel<PACKED,NG>`
  (256 threads = NG=4 groups of 64 lanes). Every group is one query token; the K/V tile
  is staged in smem **once and shared by NG query rows** ⇒ KV traffic per query token ÷ NG.
  *Precision*: int8→fp16 conversion is exact, P is fp32→fp16 rounding, accumulation fp32.
  Measured error vs a torch reference: 1e-5 … 2.6e-4 (≈100× better than the old VALU path).
* **Prefill / context (qlen > 4)**: same kernel, `grid.y = seq*nqt + qt`
  (one block = NG query rows) plus a **causal loop bound** (`cmax` = the largest q_pos in
  the tile) so KV tiles beyond the causal horizon are never touched. `pb` is forced to 1
  for this path (see §3).
* Cache-layout **auto-detection** from strides; PACKED (triton) and dim-major
  (micro-benchmark) are bit-identical in an equivalence test (`max|Δ| = 0`).

## 2. Four bugs that only the engine exposes

All four were invisible to the micro-benchmark (which used a fresh, offset-0 buffer,
16-aligned `seq_len`, and one layer per test).

| # | Root cause | Symptom | Fix |
|---|---|---|---|
| 1 | K/V views ignored **`kv_cache.storage_offset()`** — the pool is one shared storage sliced per layer, and `empty(0).set_(kv_cache.untyped_storage())` zeroes the offset ⇒ **15 of 16 layers read layer 0's cache** | output **orthogonal** to the reference (row self-match 0.25, diag cos ≈ 0) | build views with `kv_cache.as_strided(...)` (inherits the offset); V at `storage_offset = kv_off + hs + 4` |
| 2 | Multi-group kernel **did not hard-mask tokens beyond `seq_len`** (their `k_scale` is 0 ⇒ score 0, and 0 steals softmax mass when the true max is negative) | 10–70 % systematic error | `if (token >= seq_len) score = -1e30f;` (the qlen=1 kernel already did this) |
| 3 | A KV tile **entirely beyond the causal horizon** still accumulated: all 16 scores become -1e30 ⇒ `exp(0)=1` ⇒ fake `l += 16` | first call fine, later calls 100 % wrong | `if (tt > q_pos) continue;` |
| 4 | In a **partial last query tile** `g ≥ qlen` ⇒ Q row `(seq*qlen + g)` read out of bounds | GPU illegal access ⇒ engine death on a 2222-token prefill | clamp: `const int gq = g_ok ? g : (qlen - 1);` |

**Causal convention (settled)**: the engine's `seq_lens` **already contains the current
query block** (the draft forward writes the draft tokens' K/V into the cache), so
`q_pos = seq_len - qlen + g`. `/tmp/fa_int8_qoff=1` switches to `seq_len+g`, which is
equivalent to **disabling the causal mask** and measurably degrades quality — debug only.

## 3. `pb` must not be derived from dynamic data

* `pb` (KV-slice count) must come from the **real sequence length**. Using
  `block_tables.size(1)` (the *pool* blocks per sequence) pins pb at the 512 cap, leaving
  1–2 KV tiles per slice, and per-block fixed cost then swamps decode — this reproduced
  the "decode is as slow as prefill" symptom exactly.
* Under **CUDA-graph capture pb is frozen**: capture passes `max_model_len`, so the slice
  decomposition is baked in. Cap the input (`pb_tokens = min(tokens, 16384)` ⇒ pb = 128)
  so capture and replay agree; otherwise a 131072 max freezes pb = 362 and short contexts
  fall back to the 1-tile-per-slice pathology.
* Prefill (`qlen > 4`) forces `pb = 1`: otherwise `parts = nseq*qlen*hq*pb*D ≈ 10 GB`,
  and slicing makes every query tile re-read the whole KV, cancelling the tile sharing.

## 4. Engine settings that were silently costing us 3.1×

| Setting | Before | After | Effect |
|---|---|---|---|
| CUDA graphs | `cudagraph_mode = NONE` (**`--enforce-eager`**) | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, verify the `Capturing decode CUDA graphs (FULL)` log line | **+45 %** decode |
| `--max-model-len` | 262144 | **131072** | required with graphs on gfx9 (the attention gate is evaluated at capture against the configured max) |
| int8 GEMM | `Selected TritonInt8ScaledMMLinearKernel` | `Selected AiterInt8ScaledMMLinearKernel` (`VLLM_ROCM_USE_AITER=1` + `..._LINEAR=1`) | **+92 %** decode |
| AITER sub-ops | main toggle enables everything | **`_MHA=0`** etc. — `aiter.ops.mha` imports `flydsl`'s **gfx1201** FA kernel, which fails (`cannot import name 'buffer_ops'`) and kills startup | startup fix |

`_RMSNORM=1` is a **no-op** (the launcher already had `rms_norm=['aiter','native']`).
`--load-format sharded_state` requires a **pre-sharded** checkpoint
(`model-rank-N-part-*.safetensors`); it is not a runtime win by itself.
`a8w8_tuned_gemm.csv` has no gfx90a rows (the log says so) and tuning it is a documented
null result, so it is not worth pursuing.

## 5. Measured result

| Metric | Baseline (bf16 KV, triton) | Start of session | Now |
|---|---|---|---|
| Pure decode @9k | 15.7 tok/s | 24.6 | **76.4 tok/s** |
| Prefill @2.2k | 1313 | 1280 | **1665 tok/s** |
| 8-way aggregate | ~130 | 127 | **176 tok/s** |
| KV pool | 1.17 M tokens | 2.21 M | 1.94 M |

Quality: unchanged (factual QA correct, counting complete); MTP mean acceptance
3.45–3.98 of a maximum of 4.

## 6. Roofline

Per decode step at batch 1: ≈14 GB of weight traffic (27 B params ÷ TP2, W8A8) over 52 ms
⇒ **~260 GB/s = 22 %** of the measured achievable HBM bandwidth (1199 GB/s copy).
Theoretical ceiling ≈ 340 tok/s. At batch 8 the step time grows 3.5× while the weight
traffic is unchanged ⇒ the remaining cost is **not** weight bandwidth; it scales with
token count (GDN layers 48/64, TP all-reduce, attention).

Remaining levers, in order: (1) stage K through smem in the qlen=1 kernel (it currently
reads K fragments straight from global: 4 B of every 32 B sector ⇒ 12.5 % efficiency,
whereas the multi-group kernel already stages cooperatively); (2) GDN fusion path;
(3) `--enable-custom-all-reduce`; (4) MTP n sweep; (5) A=Q/B=K 16-row query-tile kernel
(NG=4 is register-limited: 204 VGPR × 512 threads exceeds the 65536 VGPR/CU budget, so
NG=8/16 are not viable).
