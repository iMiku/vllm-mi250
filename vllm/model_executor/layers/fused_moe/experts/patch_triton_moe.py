#!/usr/bin/env python3
"""Skip 2-pack layout asserts in triton_moe.py when GPTQ8 repack is on."""
TM = "/home/mi250/flash-next/vllm-src/vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
src = open(TM).read()

old1 = """        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1) // 2} == {w1.size(2)}"
            )
        else:"""
new1 = """        import os as _os
        if self.quant_config.use_int4_w4a16 and _os.environ.get("VLLM_W4A16_GPTQ8", "0") != "1":
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1) // 2} == {w1.size(2)}"
            )
        elif not self.quant_config.use_int4_w4a16:"""
assert src.count(old1) == 1, "anchor1"
src = src.replace(old1, new1, 1)

old2 = """            assert hidden_states.size(-1) // 2 == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1) // 2} == {w1.size(2)}"
            )"""
new2 = """            assert hidden_states.size(-1) // (
                8 if _os.environ.get("VLLM_W4A16_GPTQ8", "0") == "1" else 2
            ) == w1.size(2), (
                f"Hidden size mismatch"
            )"""
assert src.count(old2) == 1, "anchor2"
src = src.replace(old2, new2, 1)
open(TM, "w").write(src)
print("triton_moe.py patched")