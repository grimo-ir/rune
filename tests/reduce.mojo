# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rune regression: per-block reduce, two-launch form (no float atomics)."""

from std.math import ceildiv
from std.sys import has_accelerator

from rune.gpu import (
    AddressSpace,
    DeviceContext,
    TensorLayout,
    TileTensor,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    row_major,
    stack_allocation,
    thread_idx,
)

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime in_layout = row_major[N]()
comptime part_layout = row_major[4]()


def reduce_kernel(
    inp: TileTensor[dtype, type_of(in_layout), MutAnyOrigin],
    partials: TileTensor[dtype, type_of(part_layout), MutAnyOrigin],
):
    comptime assert inp.flat_rank == 1 and partials.flat_rank == 1
    var shared = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var tid = thread_idx.x
    var gid = Int(block_idx.x * block_dim.x + tid)
    if gid < N:
        shared[tid] = inp[gid]
    else:
        shared[tid] = 0.0
    barrier()
    var stride = BLOCK // 2
    while stride > 0:
        if Int(tid) < stride:
            shared[tid] = shared[tid] + shared[tid + stride]
        barrier()
        stride //= 2
    if Int(tid) == 0:
        partials[Int(block_idx.x)] = shared[0]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var in_buf = ctx.enqueue_create_buffer[dtype](N)
    var part_buf = ctx.enqueue_create_buffer[dtype](4)
    in_buf.enqueue_fill(1.0)
    part_buf.enqueue_fill(0.0)
    var inp = TileTensor(in_buf, in_layout)
    var partials = TileTensor(part_buf, part_layout)
    ctx.enqueue_function[reduce_kernel](inp, partials, grid_dim=4, block_dim=BLOCK)
    ctx.synchronize()
    with part_buf.map_to_host() as host:
        var parts = TileTensor(host, part_layout)
        var total: Float32 = 0.0
        for i in range(4):
            total += parts[i]
        print("partials:", parts[0], parts[1], parts[2], parts[3], "total:", total)
        if total != 1024.0:
            raise Error("reduce mismatch")
        print("PASS rune-gpu-reduce")
