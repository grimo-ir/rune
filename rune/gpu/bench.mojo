# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel timing helper over DeviceContext.execution_time."""

from max.gpu.host import DeviceContext


def bench_kernel[LaunchT: def(DeviceContext) raises -> None](name: String, ctx: DeviceContext, launch: LaunchT, iters: Int = 100) raises -> Int:
    return ctx.execution_time(launch, iters)
