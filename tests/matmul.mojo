# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rune regression: 64x64 tiled matmul with shared memory, full sweep check."""

from std.math import ceildiv
from std.sys import has_accelerator

from rune.gpu import (
    AddressSpace,
    DeviceContext,
    TensorLayout,
    TileTensor,
    barrier,
    block_idx,
    row_major,
    stack_allocation,
    thread_idx,
)

comptime dtype = DType.float32
comptime M = 64
comptime N = 64
comptime K = 64
comptime TILE = 16
comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()


def matmul_kernel[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout,
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE + ty
    var col = block_idx.x * TILE + tx

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[TILE, TILE]())
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[TILE, TILE]())

    var acc: C.ElementType = 0.0
    comptime for k_tile in range(0, K, TILE):
        if row < M and k_tile + Int(tx) < K:
            sa[ty, tx] = A[row, k_tile + Int(tx)]
        else:
            sa[ty, tx] = 0.0
        if k_tile + Int(ty) < K and col < N:
            sb[ty, tx] = B[k_tile + Int(ty), col]
        else:
            sb[ty, tx] = 0.0
        barrier()
        comptime for k in range(TILE):
            acc += sa[ty, k] * sb[k, tx]
        barrier()

    if row < M and col < N:
        C[row, col] = acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var a_buf = ctx.enqueue_create_buffer[dtype](M * K)
    var b_buf = ctx.enqueue_create_buffer[dtype](K * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](M * N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)
    c_buf.enqueue_fill(0.0)
    var A = TileTensor(a_buf, a_layout)
    var B = TileTensor(b_buf, b_layout)
    var C = TileTensor(c_buf, c_layout)
    comptime kernel = matmul_kernel[type_of(a_layout), type_of(b_layout), type_of(c_layout)]
    ctx.enqueue_function[kernel](
        A, B, C,
        grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()
    with c_buf.map_to_host() as host:
        var result = TileTensor(host, c_layout)
        var bad = 0
        for r in range(M):
            for c in range(N):
                if result[r, c] != 128.0:
                    bad += 1
        print("C[0,0] =", result[0, 0], "C[63,63] =", result[M - 1, N - 1], "bad:", bad)
        if bad != 0:
            raise Error("matmul mismatch")
        print("PASS rune-gpu-matmul")
