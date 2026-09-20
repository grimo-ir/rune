# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rune GPU: the GPU execution subset, namespaced as rune.gpu.

Covers: context + buffers + launch indexing + layout + shared memory + sync.
Excludes: device_graph, tracing, tensormap, nn/linalg/shmem kernels,
and all of max python (pipelines/serve/engine/graph/nn).

Upstream pin (pixi.toml): mojo ==1.2.0.dev2026092005,
  max-core ==26.7.0.dev2026092005. Bump together; coverage follows MAX targets.
Runtime closure (see README ## Verified GPUs for the 6-GPU matrix):
  linked: libAsyncRTMojoBindings + libKGENCompilerRTShared.
  NOT opened (strace-verified, 0 opens): libmax.so, libMGPRT, libNVPTX.
License note: the .so files behind this re-export are still
  Modular binaries (conda-meta: LicenseRef-Modular-Proprietary).
  v0 thins the dependency surface, it does not cut the tail.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, DeviceEvent, DeviceStream, Dim, HostBuffer
from max.gpu import WARP_SIZE, block_dim, block_idx, global_idx, grid_dim, lane_id, sm_id, thread_idx, warp_id
from max.gpu.primitives.warp import broadcast, max, min, reduce, shuffle_down, shuffle_idx, shuffle_up, shuffle_xor, sum, vote
from max.gpu.sync import Semaphore, barrier, named_barrier, syncwarp
from max.gpu.memory import AddressSpace, CacheEviction, CacheOperation, Consistency, Fill, async_copy, async_copy_commit_group, async_copy_wait_all, async_copy_wait_group, external_memory, load
from max.gpu.host import Attribute, FuncAttribute, LaunchAttribute
from layout import Idx, TensorLayout, TileTensor, row_major, stack_allocation
from std.atomic import Atomic
from max.benchmark import bencher_iter_custom
from rune.gpu.copy import device_to_host, host_to_device
from rune.gpu.bench import bench_kernel
