# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host<->device copy helpers. Pin the synchronize-after-copy invariant."""

from max.gpu.host import DeviceBuffer, DeviceContext


def host_to_device[dtype: DType](ctx: DeviceContext, dst: DeviceBuffer[dtype], src: Span[mut=False, Scalar[dtype], _]) raises:
    ctx.enqueue_copy(dst, src)
    ctx.synchronize()


def device_to_host[dtype: DType](ctx: DeviceContext, dst: Span[mut=True, Scalar[dtype], _], src: DeviceBuffer[dtype]) raises:
    ctx.enqueue_copy(dst, src)
    ctx.synchronize()
