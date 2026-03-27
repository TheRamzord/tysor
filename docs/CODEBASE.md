# Tysor Codebase Guide

This document describes the current Rust codebase.

## Pipeline

The main flow is:

1. source text
2. tokenization
3. AST parsing
4. semantic analysis
5. frontend IR lowering
6. graph construction
7. backend execution-plan lowering
8. execution, export, backward, or training

The CLI entrypoint is `src/main.rs`.

## Main Areas

### Frontend

- `src/compiler/lexer.rs`
- `src/compiler/parser.rs`
- `src/compiler/semantic_analyzer.rs`
- `src/compiler/frontend_ir.rs`
- `src/compiler/builtins/`

Responsibilities:

- tokenize and parse the DSL
- validate `layer`, `fn`, `config`, `train`, and global declarations
- lower the parsed program into frontend IR

### IR

- `src/ir/builder.rs`
- `src/ir/graph.rs`

Responsibilities:

- convert frontend IR into graph form
- prepare lowered programs for backend planning and execution

### Backend Planning

- `src/backend/core/execution_plan.rs`
- `src/backend/core/kind.rs`

Responsibilities:

- map lowered graph operations to backend-specific execution plans
- keep backend selection centralized and explicit

### Runtime Backends

- local: `src/backend/local.rs`
- metal: `src/backend/metal/`
- pytorch: `src/backend/pytorch/`

Responsibilities:

- local: reference CPU execution and training path
- metal: Metal code generation and runtime integration
- pytorch: standalone export and runtime-backed execution for the supported subset

### Runtime Support

- `src/runtime/interpreter.rs`
- `src/runtime/graph_executor.rs`
- `src/runtime/tensor.rs`
- `src/runtime/layers.rs`

Responsibilities:

- tensor storage and helper operations
- graph execution
- runtime-facing layer helpers

### Training

- `src/training/backward.rs`
- `src/training/executor.rs`

Responsibilities:

- gradient computation for the supported subset
- training loop orchestration

### Operation Metadata

- `src/ops/library.rs`
- `src/ops/linear_support.rs`
- `src/ops/model.rs`

Responsibilities:

- builtin signatures and runtime metadata
- helper logic for higher-level operations such as `linear`

### Native Support

- `native/metal_bridge.mm`
- `build.rs`

Responsibilities:

- compile the Objective-C++ Metal bridge on macOS
- link the bridge into the Rust crate

## Conventions

- `src/` is the authoritative implementation root.
- `tests/` contains DSL fixtures, not Rust unit tests.
- generated artifacts belong in `target/` and should not be committed.
