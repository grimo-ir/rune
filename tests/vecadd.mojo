# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rune regression: GPU vector-add through the rune.gpu wrapper.

Covers context + buffers + kernel launch through the `rune.gpu` facade.
"""

from std.math import ceildiv
from std.sys import has_accelerator

from rune.gpu import (
    DeviceContext,
    TensorLayout,
    TileTensor,
    global_idx,
    row_major,
)

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime layout = row_major[N]()


def add_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    size: Int32,
):
    var tid = Int(global_idx.x)
    if tid < Int(size):
        c[tid] = a[tid] + b[tid]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    ctx.enqueue_function[add_kernel](
        a, b, c, Int32(N),
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()

    with c_buf.map_to_host() as host:
        var result = TileTensor(host, layout)
        var ok = result[0] == 3.0 and result[N - 1] == 3.0
        print("c[0] =", result[0], "c[1023] =", result[N - 1])
        if not ok:
            raise Error("vecadd mismatch")
        print("PASS rune-gpu-vecadd")
