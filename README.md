# ᚱ Rune

**Thin GPU execution layer for Mojo.** One import surface (`rune.gpu`),
zero inference stack. Write kernels in Mojo, run them on NVIDIA GPUs —
without `pipelines`, `serve`, graphs, or Python.

```mojo
from rune.gpu import DeviceContext, TileTensor, global_idx, row_major

comptime layout = row_major[1024]()

def add_kernel(
    a: TileTensor[DType.float32, type_of(layout), MutAnyOrigin],
    b: TileTensor[DType.float32, type_of(layout), MutAnyOrigin],
    c: TileTensor[DType.float32, type_of(layout), MutAnyOrigin],
    size: Int32,
):
    var tid = Int(global_idx.x)
    if tid < Int(size):
        c[tid] = a[tid] + b[tid]
```

## ᛊ Why Rune

|                                              | MAX full                                 | Rune v0                   |
| -------------------------------------------- | ---------------------------------------- | ------------------------- |
| GPU kernels                                  | ✅                                       | ✅                        |
| `serve` / `pipelines` / `graph` / `nn`       | 45M+ Python, `libmax.so` 161M            | ❌ stripped               |
| Runtime linked                               | KGEN + Bindings + MGPRT + NVPTX + libmax | KGEN + Bindings only      |
| `libmax.so` / `libMGPRT` / `libNVPTX` opened | yes                                      | **0** (`strace`-verified) |

Same kernels, same `DeviceContext`, same PTX — minus the inference stack you never call.

## ᛞ Quickstart

```bash
pixi install
pixi run test-gpu     # vecadd · reduce · matmul · copy+bench → 4× PASS
```

```bash
pixi run test-vecadd      # single shots
pixi run test-reduce
pixi run test-matmul
pixi run test-copy-bench
```

> `mojo run` needs `-I .` (repo root isn't auto-searched); all tasks embed it.

## ᚷ Verified GPUs

`pixi run test-gpu` (vecadd · reduce · matmul · copy+bench) passes 4/4 on each.
2026-09-21, pins `mojo ==1.2.0.dev2026092005` · `max-core ==26.7.0.dev2026092005`.

| GPU | Arch | Target | Driver | Where | Result |
| --- | ---- | ------ | ------ | ----- | ------ |
| RTX 4070 SUPER | Ada | sm_89 | 616.92 | local (WSL2) | ✅ 4/4 |
| L40S 48GB | Ada | sm_89 | 580.178 | Verda FIN-02 | ✅ 4/4 |
| RTX A6000 48GB | Ampere | sm_86 | 595.71 | Verda FIN-01 | ✅ 4/4 |
| A100 80GB SXM4 | Ampere | sm_80 | 595.71 | Verda FIN-02 | ✅ 4/4 |
| RTX PRO 6000 Blackwell | Blackwell | sm_120 | 595.71 | Verda FIN-03 | ✅ 4/4 |
| H100 80GB HBM3 | Hopper | sm_90 | 595.71 | Verda FIN-02 | ✅ 4/4 |

Coverage: Ampere → Ada → Hopper → Blackwell, four generations.

## ᛟ Layout

- `rune/gpu/__init__.mojo` — the facade: context, buffers, launch indexing,
  warp ops, sync, memory, atomics, bench.
- `rune/gpu/copy.mojo` — `host_to_device` / `device_to_host`
  (Span overloads + `synchronize()` pinned, so callers can't forget).
- `rune/gpu/bench.mojo` — `bench_kernel(name, ctx, launch, iters)` → ns.
- `tests/` — `vecadd`, `reduce`, `matmul`, `copy_bench`.
  Each prints `PASS rune-gpu-*` or raises.

## ᚦ Nightly constraints (all measured, not guessed)

- Kernel scalar args must be `Int32`, not `Int` (`DevicePassable` violation).
- Host `List` for copy helpers must be typed `Float32`
  (literals default to `Float64` → Span conversion fails).
- Launch closures capturing tensors need an explicit capture list:
  `{imm a, imm b, imm c}`.
- Parametric kernels (`[LT: TensorLayout]`) must be bound before launch:
  `comptime kernel = f[type_of(layout)]`.

## ᚾ Why MAX is a dependency

Mojo alone cannot touch the GPU. The compiler emits PTX, but allocating
device memory, loading modules, launching kernels, and synchronizing
all go through `DeviceContext` → `libAsyncRTMojoBindings.so`,
both owned by the `max-core` conda package. There is no Mojo-native
driver path; hand-rolled `libcuda.so.1` calls remain a future option.
The second reason is legal, not technical: reimplementing that runtime
yourself trips the MAX Community License's AI-input clause (1.3.4 —
no MAX-derived substitute runtimes). Keeping MAX as the dependency
keeps Rune on the permitted side (use + interoperate, with attribution).

What that costs, measured: at runtime only `libKGENCompilerRTShared.so`
+ `libAsyncRTMojoBindings.so` link (`readelf -d` on built test binaries);
`strace` shows zero opens of `libmax.so` / `libMGPRT.so` / `libNVPTX.so`,
  and the `max` Python tree is never opened either — it isn't even installed
  (`max-core` replaces `max`). The dependency is the GPU runtime
  (~3M of `.so` + `max.mojoc` slice), not the inference stack.

Bump policy: track GPU-path changes only (`gpu/host/*`, `layout`,
`AsyncRT_DeviceContext_*`, `get_gpu_target`). Ignore
`pipelines` / `serve` / `graph` / `nn` churn.

## ᛏ License

Rune's own code: **Apache 2.0 with LLVM Exceptions** — see [`LICENSE`](LICENSE).

Upstream binaries (`mojo` toolchain, `max-core` `.so` / `.mojoc`) are
Modular binaries under the [Modular MAX Community License](https://www.modular.com/legal/community)
(`conda-meta` records `LicenseRef-Modular-Proprietary`).
Required attribution for any distribution lives in [`NOTICE`](NOTICE).
v0 thins the dependency surface; it does not cut that tail.
