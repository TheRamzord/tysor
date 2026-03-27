# tysor

`tysor` is a research-oriented deep-learning DSL compiler and runtime implemented in Rust.
The repository is organized as a standard Cargo project with a small amount of native code for
the Metal bridge.

## What It Does

- tokenizes and parses the DSL
- runs semantic analysis
- lowers programs into a frontend IR and graph form
- executes supported programs on the local backend
- executes supported programs on the Metal backend
- exports and runs supported programs through the PyTorch backend
- supports backward execution and small-model training for the current subset

## Repository Layout

```text
.
├── src/           Rust crate sources
├── native/        Objective-C++ Metal bridge
├── tests/         DSL fixtures
├── examples/      Example Tysor programs
├── scripts/       Thin developer helpers
└── docs/          Language and implementation docs
```

## Build

```sh
cargo build
```

Run the CLI directly:

```sh
cargo run -- tests/test_model_matmul_relu.ty
```

Or use the built binary:

```sh
./target/debug/tysor tests/test_model_matmul_relu.ty
```

## Common Commands

Emit standalone PyTorch:

```sh
cargo run -- tests/test_model_matmul_relu.ty --emit-pytorch
cargo run -- tests/test_train_small_mlp.ty --emit-pytorch
```

Run with the built-in runtime:

```sh
cargo run -- tests/test_model_matmul_relu.ty --run --shape x=2x3 --shape w=3x4
cargo run -- tests/test_run_two_layer_mlp.ty --run --shape x=2x3
cargo run -- tests/test_model_matmul_relu.ty --run --backend metal --shape x=2x3 --shape w=3x4
```

Train with the current supported subset:

```sh
cargo run -- tests/test_train_small_mlp.ty --train --shape x=2x3 --shape target=2x3
cargo run -- tests/test_train_small_mlp.ty --train --backend pytorch --shape x=2x3 --shape target=2x3
```

`--run` defaults to `local`. Supported backends are `local`, `metal`, and `pytorch`.

## Tests

Run the full fixture suite:

```sh
cargo run --bin run_tests --
```

Run a focused runtime slice:

```sh
cargo run --bin run_tests -- --run test_run_two_layer_mlp.ty test_model_matmul_relu.ty
```

Run PyTorch export checks:

```sh
cargo run --bin run_tests -- --emit-pytorch \
  test_model_matmul_relu.ty test_train_small_mlp.ty test_run_rms_norm.ty
```

Run train fixtures:

```sh
cargo run --bin run_tests -- --train test_train_small_mlp.ty test_train_silu_mlp.ty
```

## Docs

- [`docs/LANGUAGE.md`](docs/LANGUAGE.md): language semantics
- [`docs/CODEBASE.md`](docs/CODEBASE.md): codebase tour and implementation responsibilities
- [`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md): daily-use commands and layout reference
