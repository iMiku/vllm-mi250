# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for PLE layers that support cross-process CPU offload.

Synchronization between the offload process and the GPU main stream is
handled by :class:`CpuGpuSemaphore`, which uses CUDA's
``cuStreamWaitValue32`` / ``cuStreamWriteValue32`` stream memory operations.

Typical decode-step timeline
----------------------------
Offload process (CPU)                  GPU main stream (forward)
-----------------------------          ------------------------------
forward_impl() on CPU                  ... other GPU ops ...
pinned -> GPU async H2D (copy_stream)   WaitValue32(flag==1)  <- in Graph
WriteValue32(flag=1, copy_stream)       return _gpu_output_buffer[:n]
                                       ... model consumes output ...
                                       WriteValue32(flag=0)  <- after forward
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from vllm.platforms import current_platform

if current_platform.is_rocm():
    # gfx90a: HIP exposes the same stream memory-ops via libamdhip64.so.
    # Flag values follow the HIP header: hipStreamWaitValue_eq = 0x2.
    import ctypes

    _hip = ctypes.CDLL("libamdhip64.so")
    _HIP_STREAM_WAIT_VALUE_EQ = 0x2
    _hip.hipStreamWriteValue32.restype = ctypes.c_int
    _hip.hipStreamWriteValue32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint,
    ]
    _hip.hipStreamWaitValue32.restype = ctypes.c_int
    _hip.hipStreamWaitValue32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint,
        ctypes.c_uint,
    ]

    def _hip_check(rc: int, operation: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{operation} failed: hipError_t={rc}")
else:
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings.driver import CUstreamWaitValue_flags

import vllm.envs as envs
from vllm.utils.torch_utils import direct_register_custom_op

# Module-level flag set to True inside the offload subprocess.
# Because the offload process and GPU worker processes are separate OS
# processes (spawned via multiprocessing), each has its own memory space.
# A plain module-level bool is sufficient -- no thread-local storage needed.
_offload_worker_flag = False


def is_offload_process() -> bool:
    """Return True inside the dedicated PLE CPU-offload subprocess."""
    return _offload_worker_flag


def mark_as_offload_worker() -> None:
    """Mark the current process as the dedicated PLE offload worker."""
    global _offload_worker_flag
    _offload_worker_flag = True


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class CpuGpuSemaphore:
    """Cross-process semaphore backed by a one-element int32 CUDA tensor.

    The flag is stored in a regular GPU tensor that is shared through
    PyTorch's CUDA IPC mechanism. Both processes obtain device pointers that
    map to the same physical GPU memory, so a stream-memory wait in the GPU
    worker observes the write issued by the offload process.
    """

    RESET_VALUE = 0
    DONE_VALUE = 1

    def __init__(self, device: torch.device) -> None:
        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)

    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        """Construct a semaphore from a CUDA tensor received through IPC."""
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        return semaphore

    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the CUDA tensor used to share the semaphore through IPC."""
        return self._flag_tensor

    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        if current_platform.is_rocm():
            _hip_check(
                _hip.hipStreamWriteValue32(
                    stream.cuda_stream,
                    self._flag_tensor.data_ptr(),
                    self.RESET_VALUE,
                    0,
                ),
                "CpuGpuSemaphore.reset",
            )
            return
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                0,
            ),
            "CpuGpuSemaphore.reset",
        )

    def signal_value(
        self,
        stream: torch.cuda.Stream | None = None,
        value: int = 1,
    ) -> None:
        """Monotonic write: enqueue ``flag = value`` (value must increase)."""
        if stream is None:
            stream = torch.cuda.current_stream()
        if current_platform.is_rocm():
            _hip_check(
                _hip.hipStreamWriteValue32(
                    stream.cuda_stream,
                    self._flag_tensor.data_ptr(),
                    value,
                    0,
                ),
                "CpuGpuSemaphore.signal_value",
            )
            return
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                value,
                0,
            ),
            "CpuGpuSemaphore.signal_value",
        )

    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        if current_platform.is_rocm():
            _hip_check(
                _hip.hipStreamWriteValue32(
                    stream.cuda_stream,
                    self._flag_tensor.data_ptr(),
                    self.DONE_VALUE,
                    0,
                ),
                "CpuGpuSemaphore.signal",
            )
            return
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.DONE_VALUE,
                0,
            ),
            "CpuGpuSemaphore.signal",
        )

    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        if current_platform.is_rocm():
            _hip_check(
                _hip.hipStreamWaitValue32(
                    stream.cuda_stream,
                    self._flag_tensor.data_ptr(),
                    self.RESET_VALUE,
                    _HIP_STREAM_WAIT_VALUE_EQ,
                    0xFFFFFFFF,
                ),
                "CpuGpuSemaphore.wait_reset",
            )
            return
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "CpuGpuSemaphore.wait_reset",
        )


# ---------------------------------------------------------------------------
# Custom op: vllm::ple_offload_wait
# ---------------------------------------------------------------------------
# Wraps cuStreamWaitValue32 as a torch custom op so that
# torch.compile(fullgraph=True) treats the CUDA Driver call as an opaque node.
# The hidden_states argument creates a dynamo data-dependency edge that prevents
# the wait from being reordered before preceding GPU work.
# ---------------------------------------------------------------------------


def _ple_offload_wait_impl(
    sem_flag_tensor: torch.Tensor,
    expected_seq: int,
    hidden_states: torch.Tensor,
) -> None:
    """Wait (device-side, GTE) until the offload process finishes `expected_seq`."""
    stream = torch.cuda.current_stream()
    if current_platform.is_rocm():
        _hip_check(
            _hip.hipStreamWaitValue32(
                stream.cuda_stream,
                sem_flag_tensor.data_ptr(),
                ctypes.c_int(expected_seq),
                0x1,  # hipStreamWaitValue_gte
                0xFFFFFFFF,
            ),
            "hipStreamWaitValue32(gte)",
        )
        return
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    dev_ptr = cuda_driver.CUdeviceptr(sem_flag_tensor.data_ptr())
    _cuda_check(
        cuda_driver.cuStreamWaitValue32(
            cuda_stream,
            dev_ptr,
            expected_seq,
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GTE.value,
        ),
        "cuStreamWaitValue32(gte)",
    )


def _ple_offload_wait_fake(
    sem_flag_tensor: torch.Tensor,
    expected_seq: int,
    hidden_states: torch.Tensor,
) -> None:
    """Represent the side-effect-only wait during dynamo tracing."""
    pass


direct_register_custom_op(
    op_name="ple_offload_wait",
    op_func=_ple_offload_wait_impl,
    mutates_args=set(),
    fake_impl=_ple_offload_wait_fake,
)


class PleOffloadLayer(nn.Module, ABC):
    """Base class for embedding-like PLE layers that can run on CPU.

    Subclasses implement :meth:`forward_impl`, which contains the regular
    on-device computation. In a GPU worker with PLE offload enabled, subclass
    constructors are skipped so their large weights are never allocated. The
    CPU process constructs a complete copy and owns those weights instead.
    """

    _is_cpu_offloaded = False
    _gpu_output_buffer: torch.Tensor
    _sem: CpuGpuSemaphore

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Skip subclass initialization in cross-process GPU-worker mode."""
        super().__init_subclass__(**kwargs)
        original_init_obj = cls.__dict__.get("__init__")
        if original_init_obj is None or not callable(original_init_obj):
            return
        original_init = cast(Callable[..., None], original_init_obj)

        @functools.wraps(original_init)
        def guarded_init(
            self: "PleOffloadLayer", *args: object, **kwargs: object
        ) -> None:
            if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
                nn.Module.__init__(self)
                return
            original_init(self, *args, **kwargs)

        cls.__init__ = guarded_init  # type: ignore[method-assign, assignment]

    @classmethod
    def get_target_device(cls) -> torch.device:
        """Return CPU for the offload process and the active GPU otherwise."""
        if envs.VLLM_PLE_CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device("cuda", torch.accelerator.current_device_index())

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the actual embedding computation on the owning device."""
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used by cross-process output buffers."""
        return default_dtype

    def get_offload_output_dim(self, default_dim: int) -> int:
        """Return the last dimension used by cross-process output buffers."""
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Initialize metadata needed by dummy offload execution."""

    def setup_cross_process_offload(
        self,
        gpu_output_buffer: torch.Tensor,
        semaphore: CpuGpuSemaphore,
    ) -> None:
        """Configure the GPU-worker placeholder with its IPC resources.

        gpu_output_buffer has two slots (seq % 2); the flag is a monotonic
        sequence counter written by the offload process after filling slot
        (seq-1) % 2. The GPU waits for flag >= seq (GTE) -- no reset, so a
        slow consumer can never lose a wakeup.
        """
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._sem = semaphore
        self._ple_seq = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Wait for an offloaded result or delegate to ``forward_impl``."""
        if self._is_cpu_offloaded:
            self._ple_seq += 1
            seq = self._ple_seq
            torch.ops.vllm.ple_offload_wait(
                self._sem.flag_tensor,
                seq,
                hidden_states,
            )
            return self._gpu_output_buffer[(seq - 1) % 2, : input_ids.shape[0]]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """No-op: monotonic sequencing needs no flag reset."""
        del stream
