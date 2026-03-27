# Tysor Project Reference

This document is the quick operational reference for working in the repository.

## Repository Layout

- `Cargo.toml`: crate manifest
- `src/`: Rust implementation
- `native/`: Objective-C++ Metal bridge
- `tests/`: DSL fixtures used by the harness
- `examples/`: sample programs
- `docs/`: language and implementation docs

## Source Layout

- `src/main.rs`: CLI entrypoint
- `src/lib.rs`: crate module root
- `src/compiler/`: lexer, parser, semantic analysis, frontend lowering
- `src/ir/`: graph construction
- `src/backend/core/`: backend kinds and execution-plan lowering
- `src/backend/cuda/`: CUDA backend scaffold and future runtime/codegen slot
- `src/backend/local.rs`: local backend entrypoints
- `src/backend/metal/`: Metal codegen and runtime integration
- `src/backend/pytorch/`: PyTorch export and runtime integration
- `src/runtime/`: tensors, layers, graph execution, runtime helpers
- `src/training/`: backward pass and training executor
- `src/ops/`: builtin op metadata and lowering helpers
- `src/bin/run_tests.rs`: compiled fixture runner used by the Python wrapper

## Common Commands

Build:

```sh
cargo build
```

Run the CLI:

```sh
cargo run -- tests/test_model_matmul_relu.ty
```

Run a model:

```sh
cargo run -- tests/test_model_matmul_relu.ty --run --shape x=2x3 --shape w=3x4
```

Run with Metal:

```sh
cargo run -- tests/test_model_matmul_relu.ty --run --backend metal --shape x=2x3 --shape w=3x4
```

Run with PyTorch:

```sh
cargo run -- tests/test_model_matmul_relu.ty --run --backend pytorch --shape x=2x3 --shape w=3x4
```

Train:

```sh
cargo run -- tests/test_train_small_mlp.ty --train --shape x=2x3 --shape target=2x3
```

Emit Metal:

```sh
cargo run -- tests/test_model_matmul_relu.ty --emit-metal
```

Emit PyTorch:

```sh
cargo run -- tests/test_model_matmul_relu.ty --emit-pytorch
```

## Tests

Run the default fixture sweep:

```sh
cargo run --bin run_tests --
```

Run focused runtime fixtures:

```sh
cargo run --bin run_tests -- --run test_run_two_layer_mlp.ty test_model_matmul_relu.ty
```

Run Metal comparisons:

```sh
cargo run --bin run_tests -- --run --backend metal --compare-cpu
```

Run backward comparisons:

```sh
cargo run --bin run_tests -- --backward --backend metal --compare-cpu
```

Run training fixtures:

```sh
cargo run --bin run_tests -- --train test_train_small_mlp.ty test_train_silu_mlp.ty
```

## Backends

Supported backends:

- `local`
- `cuda`
- `metal`
- `pytorch`

`local` is the default. Backend parsing lives in `src/backend/core/kind.rs`.
The current CUDA backend is scaffolded only and returns an unsupported-host/build error until it
is implemented and validated on a CUDA-capable machine.

## Related Docs

- `docs/LANGUAGE.md`
- `docs/CODEBASE.md`
- `docs/RUST_MIGRATION_STATUS.md`
