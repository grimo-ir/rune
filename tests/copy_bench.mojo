# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rune regression: copy helpers + pinned path + bench timing."""

from std.math import ceildiv
from std.sys import has_accelerator

from rune.gpu import (
    DeviceContext,
    TileTensor,
    bench_kernel,
    device_to_host,
    global_idx,
    host_to_device,
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

    var src: List[Float32] = List[Float32](length=N, fill=2.0)
    var dev = ctx.enqueue_create_buffer[dtype](N)
    host_to_device(ctx, dev, Span(src))
    var dst = List[Float32](length=N, fill=0.0)
    device_to_host(ctx, Span(dst), dev)
    var bad = 0
    for i in range(N):
        if dst[i] != 2.0:
            bad += 1
    print("copy bad:", bad)
    if bad != 0:
        raise Error("copy mismatch")

    var pinned = ctx.enqueue_create_host_buffer[dtype](N)
    dev.enqueue_copy_to(pinned)
    ctx.synchronize()
    var via_pinned = List[Float32](length=N, fill=0.0)
    for i in range(N):
        via_pinned[i] = pinned[i]
    var pinned_bad = 0
    for i in range(N):
        if via_pinned[i] != 2.0:
            pinned_bad += 1
    print("pinned bad:", pinned_bad)
    if pinned_bad != 0:
        raise Error("pinned copy mismatch")

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)
    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    def launch(cctx: DeviceContext) raises {imm a, imm b, imm c}:
        cctx.enqueue_function[add_kernel](a, b, c, Int32(N), grid_dim=ceildiv(N, BLOCK), block_dim=BLOCK)

    var ns = bench_kernel("vecadd", ctx, launch, iters=10)
    print("bench ns:", ns)
    if ns <= 0:
        raise Error("bench returned non-positive")
    print("PASS rune-gpu-copy-bench")
