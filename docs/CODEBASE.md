# Tysor Codebase Guide

This document is the contributor map for the current `tysor` codebase. It is intentionally
practical: it explains what the system does, where each responsibility lives, how data moves
through the pipeline, and what each major file is doing today.

The repository is a single Rust crate that combines:

- a frontend compiler for the Tysor DSL
- a graph-building layer
- backend-specific execution and export paths
- a small tensor runtime
- a backward and training stack
- a fixture-based integration harness

## What The Project Does

At a high level, `tysor` takes a `.ty` source file and turns it into one of several outcomes:

- a validated program model
- a lowered frontend IR
- a graph representation
- a backend execution plan
- a local execution result
- a Metal execution result
- a PyTorch export/runtime bridge
- a backward pass over the graph
- a simple training loop

The important point is that this is not just a parser. It is an end-to-end language and execution
stack.

## End-To-End Pipeline

The main pipeline is:

1. source text is read from a `.ty` file
2. `lexer.rs` tokenizes it
3. `parser.rs` builds an AST
4. `semantic_analyzer.rs` validates names, types, and call rules
5. `frontend_ir.rs` lowers the AST into a more regular frontend IR
6. `ir/builder.rs` converts lowered functions into graph form
7. `backend/core/execution_plan.rs` turns the graph into a backend-oriented plan
8. one of the backend/runtime paths executes, exports, differentiates, or trains the result

That phase split is reflected directly in the source tree, so the codebase is easier to reason
about if you follow it in that order.

## Top-Level Layout

```text
.
├── Cargo.toml
├── build.rs
├── native/
├── src/
├── tests/
├── examples/
└── docs/
```

Top-level responsibilities:

- `Cargo.toml`
  Crate manifest, dependencies, and binary targets.
- `build.rs`
  Compiles the Objective-C++ Metal bridge on macOS and links the required Apple frameworks.
- `native/`
  Native bridge code used by the Rust Metal runtime.
- `src/`
  The implementation of the compiler, IR, backends, runtime, ops metadata, and training code.
- `tests/`
  DSL fixture files consumed by the custom Rust test runner.
- `examples/`
  Sample Tysor programs.
- `docs/`
  Language, project, and architecture documentation.

## Main Entrypoints

### `src/main.rs`

This is the main `tysor` CLI. It is a thin orchestrator, not the implementation core.

Its responsibilities are:

- parse CLI flags such as `--run`, `--backward`, `--train`, `--emit-metal`, `--emit-pytorch`
- parse inspection flags such as `--tokens`, `--ast`, `--semantics`, `--ir`
- read the source file
- run the frontend pipeline once
- optionally print intermediate compiler stages
- dispatch to the requested backend or execution mode

In other words, `main.rs` is the best file to read first if you want the top-down control flow.

### `src/bin/run_tests.rs`

This is the project’s real integration harness. It is not a thin wrapper around `cargo test`.

Its responsibilities are:

- discover `.ty` files under `tests/`
- invoke the compiled `tysor` binary
- classify expected failures
- support runtime, backward, train, and export workflows
- pass per-fixture shape arguments when runtime execution needs tensor shapes
- compare Metal and PyTorch results against the local backend
- limit backend test sweeps to the currently supported fixture subset
- surface compiler-stage dumps for tokens, AST, semantics, and lowered IR

This file is where backend support status is effectively encoded today. The allowlists here show
which fixtures are considered supported for Metal and PyTorch comparison runs.

## Crate Root

### `src/lib.rs`

`lib.rs` is intentionally small. It exposes the major subsystem modules:

- `backend`
- `compiler`
- `ir`
- `ops`
- `runtime`
- `training`

This file exists mainly to define the crate’s public structure cleanly.

## Compiler Layer

The compiler layer lives under `src/compiler/`. This is where source text becomes a validated,
lowered program.

### `src/compiler/lexer.rs`

Responsibility:

- convert source text into tokens

What it owns:

- token kinds
- token values
- token formatting
- keyword/operator recognition
- indentation-sensitive token production
- numeric and string literal lexing

Why it matters:

- the language is indentation-sensitive, so the lexer emits indentation structure the parser relies
  on
- if the parser starts failing around layout or punctuation, the lexer is usually the first thing
  to inspect

### `src/compiler/parser.rs`

Responsibility:

- convert the token stream into the AST

What it defines:

- source spans
- declared types
- expressions
- statements
- function declarations
- layer declarations
- config declarations
- train declarations
- the top-level `Program`

Important language behavior here:

- parses explicit-width integer types such as `int16`, `int32`, `int64`
- parses explicit-width float types such as `float16`, `float32`, `float64`
- parses tensor dtype annotations
- parses the `->` arrow operator
- enforces syntax-shape rules for arrow stages

What it does not do:

- it does not decide the full semantic legality of a construct
- for example, parser-level acceptance of an arrow form does not mean it is valid in every
  callable context

### `src/compiler/semantic_analyzer.rs`

Responsibility:

- validate names, declarations, types, and language rules

What it does:

- collects configs, layers, functions, and train declarations
- detects duplicate declarations and cross-kind name conflicts
- validates declared types
- checks expressions and statements
- enforces return-type compatibility
- tracks callable context such as whether analysis is happening inside a `fn` or `layer`
- validates tensor dtypes against the supported set
- applies type compatibility and promotion rules
- validates train-model structure

Important current rules enforced here:

- `fn` and `layer` have different calling restrictions
- `fn` may use `->`, but inside `fn` the arrow chain must remain function-only
- tensor dtype compatibility is checked when both sides are concrete
- builtin signatures are registered before user functions so the analyzer can type-check builtin
  calls like regular callsites

If you want to change the language policy, this file is usually the first implementation point.

### `src/compiler/frontend_ir.rs`

Responsibility:

- lower the AST into a normalized frontend IR

Why this layer exists:

- the AST is source-shaped and rich in syntax detail
- later stages want something smaller and more execution-oriented

What it defines:

- `FeType` and `FeTypeKind`
- `FeValue`
- lowered expressions and statements
- lowered functions and layers
- lowered config/train structures
- `LoweredModule`
- lowering logic from AST to frontend IR

What it does in practice:

- rewrites parsed expressions into a more regular representation
- preserves tensor dtype and callable metadata needed by later stages
- lowers layer constructors and apply sites into explicit IR forms
- prepares execution-plan metadata for `train model` flows

This file is the bridge between “language frontend” and “execution system”.

### `src/compiler/builtins/`

This directory centralizes builtin-language metadata and lowering support.

Files:

- `mod.rs`
  Module wiring.
- `registry.rs`
  Builds the builtin signature list used by semantic analysis and lowering.
- `type_rules.rs`
  Encodes builtin return-type inference logic.
- `lowering.rs`
  Shared lowering helpers for builtins.

This folder exists so builtin behavior is not spread ad hoc across the parser, analyzer, and
lowerer.

## IR Layer

The IR layer lives under `src/ir/`. Its job is to turn normalized frontend functions into graph
form.

### `src/ir/graph.rs`

Responsibility:

- define the graph data structures

Main data:

- `GraphValue`
  A typed value in the graph, including whether it is a parameter and whether it requires grads.
- `GraphNode`
  A producer operation, including its kind, output value, op name, and input value ids.
- `GraphFunction`
  A graph-form function with values, nodes, outputs, and name lookup tables.

This is the backend-facing intermediate representation.

### `src/ir/builder.rs`

Responsibility:

- build a `GraphFunction` from a lowered frontend function

What it does:

- creates graph values for parameters and intermediates
- converts lowered expressions into graph nodes
- records outputs for returned values
- marks tensor parameters as gradient-bearing

Important limitation:

- the current builder expects the straight-line subset of the language
- it rejects control-flow-heavy lowered forms that are not yet represented in graph form

That limitation is important because many runtime and training features assume the graph builder
can reduce the entry function to a simple dataflow graph.

## Backend Layer

The backend layer lives under `src/backend/`. It splits into:

- `backend/core/`
  backend-independent plan construction
- `backend/local.rs`
  direct local execution wrappers
- `backend/metal/`
  Metal codegen and runtime
- `backend/pytorch/`
  PyTorch export/runtime bridge

### `src/backend/core/kind.rs`

Responsibility:

- define the backend enum and string conversions

This is the shared backend selector used by the CLI, execution plan compilation, runtime dispatch,
and training dispatch.

### `src/backend/core/execution_plan.rs`

Responsibility:

- lower graph functions into backend-oriented execution plans

What it defines:

- value placement (`Host` vs `Device`)
- plan operations
- step kinds such as allocation, upload, dispatch, download, and output materialization
- `ExecutionPlan`

What it does:

- builds a local execution plan that keeps values on host and executes ops in order
- builds a Metal execution plan that makes host/device placement explicit
- emits upload/download steps for backend interaction

This file is the scheduling boundary between graph semantics and concrete runtime behavior.

### `src/backend/local.rs`

Responsibility:

- provide local backend wrappers for forward, backward, and training execution

This file is mostly an integration layer that routes lowered modules into the local runtime and
training logic.

Conceptually, `local` is the reference backend. Other backends are compared against it.

## Metal Backend

The Metal backend lives under `src/backend/metal/`.

It currently has two execution styles:

- a native path through the Objective-C++ bridge
- a Rust simulated fallback that mirrors the execution-plan categories

### `src/backend/metal/codegen.rs`

Responsibility:

- generate Metal shader source from an execution plan

What it contains:

- kernel metadata structures
- kernel-name generation
- helper emitters for matmul, elementwise ops, binary ops, linear, embedding, RMS norm, softmax,
  cross entropy, rope, causal mask, and related operations
- plan-to-kernel mapping logic

This file answers the question: “if a graph op runs on Metal, what shader source should represent
it?”

Important current characteristic:

- scalar codegen is still effectively float-oriented
- this backend is focused on working coverage first, not aggressive optimization

### `src/backend/metal/native.rs`

Responsibility:

- own the FFI bridge between Rust and `native/metal_bridge.mm`

It loads compiled Metal source, manages native buffers, and dispatches native kernels when the host
has a usable Metal device and toolchain.

### `src/backend/metal/runtime.rs`

Responsibility:

- provide the main Metal runtime entrypoints

What it does:

- finds the selected entry function
- compiles a Metal execution plan
- tries the native Metal runtime first
- falls back to the Rust-side simulated Metal path when the native path is unavailable in allowed
  situations
- executes plan ops against host values and device-style tensor maps
- materializes the result back into the same `GraphExecutionResult` shape used elsewhere

Important environment controls:

- `TYSOR_METAL_ALLOW_FALLBACK`
  lets native-Metal failures drop to the simulated path
- `TYSOR_METAL_REQUIRE_NATIVE`
  forces failure if native Metal is not available

Practical reading:

- if you want correctness and behavior, read `runtime.rs`
- if you want GPU kernel emission, read `codegen.rs`
- if you want FFI and actual Metal device calls, read `native.rs` and `native/metal_bridge.mm`

## PyTorch Backend

The PyTorch backend lives under `src/backend/pytorch/`.

This backend is not a libtorch embedding. Instead it uses generated Python and runs `python3`.

### `src/backend/pytorch/codegen.rs`

Responsibility:

- generate standalone PyTorch module/program output from lowered Tysor programs

This is the export-focused side of the PyTorch backend.

### `src/backend/pytorch/runtime.rs`

Responsibility:

- run lowered graphs through a generated Python/PyTorch script

What it does:

- builds a temporary Python script from the lowered graph
- synthesizes inputs and parameters when needed
- executes the script with `python3`
- parses the emitted tensor/scalar values back into Rust runtime structures
- supports forward execution and integrates with the shared Rust training path

This backend is useful as:

- an export bridge
- a reference implementation for certain ops
- a runtime comparison path for the subset currently supported by the harness

## Runtime Layer

The runtime layer lives under `src/runtime/`. It is the small execution substrate used by the local
backend and reused by parts of the Metal and training code.

### `src/runtime/tensor.rs`

Responsibility:

- define the simple tensor container and tensor utilities

Main data:

- `SimpleTensor`
  shape + data + dtype string

What it provides:

- tensor printing
- synthetic tensor creation for fixture runs
- elementwise binary operations
- scalar/tensor binary operations
- matmul
- transpose
- ReLU, SiLU, softmax
- zero/one helpers
- in-place accumulation

Important current limitation:

- tensor storage is still `Vec<f32>`
- dtype strings are tracked, but execution is still fundamentally float32-backed

That is a key architectural fact when reading the current backend and training behavior.

### `src/runtime/layers.rs`

Responsibility:

- implement layer-like runtime behavior and many forward/backward helpers

What it contains:

- closure/spec structures for linear and embedding layers
- parameter initialization helpers
- forward implementations for linear, dropout, embedding, reshape, repeat-kv, flatten-heads,
  causal mask, rope, and more
- backward helpers used by training and backward execution

This file is where many of the library-level semantics become actual tensor math.

### `src/runtime/graph_executor.rs`

Responsibility:

- execute local execution plans and graph functions

What it does:

- materializes runtime values from plan values
- synthesizes tensor inputs from `--shape`
- interprets primitive ops, library calls, constructors, and apply nodes
- executes local graph plans in order
- returns both all computed values and final outputs

This is the local reference executor. When backend behavior is ambiguous, this is the baseline
implementation to compare against.

### `src/runtime/interpreter.rs`

Responsibility:

- provide the top-level local runtime entrypoint for lowered modules

What it does:

- selects the entry function
- compiles a local execution plan
- runs it through `graph_executor.rs`
- prints the final tensor result

This file is intentionally thin. It is the local equivalent of the higher-level backend runtime
entrypoints.

## Training Layer

The training layer lives under `src/training/`.

It is a shared Rust implementation that can use different forward backends while keeping backward
and SGD logic centralized.

### `src/training/backward.rs`

Responsibility:

- compute backward gradients over graph executions

What it does:

- rebuilds the graph for the entry function
- runs the forward pass through the chosen backend
- resolves the training objective or output value
- walks graph nodes in reverse order
- accumulates gradients into graph values
- uses runtime backward helpers for supported operations

This is the core reverse-mode differentiation implementation in the current codebase.

### `src/training/executor.rs`

Responsibility:

- run the shared training loop

What it does:

- resolves the selected `train model` execution run
- builds or initializes parameter tensors
- runs forward computation
- computes parameter gradients
- applies simple update logic
- supports local, Metal, and PyTorch-backed forward execution with shared Rust-side training logic

Conceptually:

- forward may vary by backend
- parameter ownership, gradient accumulation, and update rules stay in Rust here

## Ops Metadata Layer

The ops metadata layer lives under `src/ops/`.

This is where builtin operation knowledge is centralized so it can be shared by the compiler,
runtime, and training code.

### `src/ops/library.rs`

Responsibility:

- define builtin operations and their metadata

What it contains:

- builtin signatures
- runtime primitive classification
- metadata such as:
  - whether an op is a primitive tensor op
  - whether it is a library op
  - whether it is a callable constructor
  - whether it preserves the first tensor argument
  - whether the runtime directly supports it

This file is the source of truth for many builtin decisions.

### `src/ops/model.rs`

Responsibility:

- expose higher-level model-oriented queries over builtin/library metadata

Examples:

- whether a name is treated as a layer constructor
- whether a name is considered a tensor builtin
- whether a constructor/runtime path preserves input tensor shape semantics

This file keeps the compiler/frontends from reaching too deeply into raw op tables.

### `src/ops/linear_support.rs`

Responsibility:

- placeholder support for collecting linear-layer metadata

Current state:

- this file is largely scaffolding today
- the helper functions currently return empty/default results

This is a good example of an area reserved for future structure, not a mature subsystem yet.

## Native Layer

### `build.rs`

Responsibility:

- compile and link the native Metal bridge on macOS

What it does:

- recompiles when `native/metal_bridge.mm` changes
- skips native Metal build steps on non-macOS targets
- invokes `xcrun clang++` to compile the Objective-C++ source
- archives it into a static library
- links Foundation, Metal, and the C++ runtime

### `native/metal_bridge.mm`

Responsibility:

- provide the native Objective-C++ Metal implementation used by Rust FFI

What it contains:

- Metal device/context setup
- shader-library compilation
- compute pipeline caching
- buffer allocation and readback
- dispatch helpers for unary, binary, and matmul-style kernels
- native error capture

This file is the lowest-level execution component in the repo.

## How Data Moves Across The System

A typical forward run on the local backend looks like this:

1. `src/main.rs` parses flags and loads the source file
2. `lexer.rs` tokenizes source text
3. `parser.rs` builds `Program`
4. `semantic_analyzer.rs` validates the program
5. `frontend_ir.rs` lowers it into `LoweredModule`
6. `ir/builder.rs` turns the entry function into `GraphFunction`
7. `execution_plan.rs` builds a local `ExecutionPlan`
8. `graph_executor.rs` executes the plan
9. `tensor.rs` prints the final output tensor

A Metal run differs after lowering:

1. graph becomes a Metal execution plan
2. `backend/metal/runtime.rs` tries native execution first
3. `backend/metal/codegen.rs` and `backend/metal/native.rs` support the native path
4. if allowed, unsupported native environments fall back to the simulated Metal runtime

A PyTorch run differs after graph building:

1. `backend/pytorch/runtime.rs` generates a Python program
2. that script runs under `python3`
3. results are parsed back into Rust

A train run adds:

1. train execution metadata from lowered IR
2. forward execution through the chosen backend
3. backward traversal in `training/backward.rs`
4. parameter updates in `training/executor.rs`

## Tests And Validation

The project’s main test surface is fixture-driven.

Important facts:

- `tests/` contains Tysor programs, not Rust unit tests
- `src/bin/run_tests.rs` is the main validation harness
- backend coverage is currently represented by allowlists inside the harness
- some fixtures are expected failures by design
- runtime and training tests often require explicit `--shape` arguments

Practical implication:

- when someone says “support this on Metal”, it usually means:
  - the compiler accepts the fixture
  - the backend executes it
  - the harness includes it in the supported allowlist
  - comparison against local passes

## Architectural Boundaries

The cleanest mental boundaries in the code today are:

- `compiler/`
  source language understanding
- `ir/`
  graph construction
- `backend/core/`
  backend-neutral planning
- `backend/*`
  backend-specific execution/export behavior
- `runtime/`
  tensor math and local execution support
- `training/`
  backward and update logic
- `ops/`
  shared builtin metadata

These are strong boundaries conceptually, even though the implementation is still evolving and some
files have direct knowledge of neighboring layers.

## Important Current Limitations

The codebase is functional, but it is still intentionally pragmatic rather than finished.

Current limitations worth knowing before making changes:

- runtime tensor storage is still float32-backed even when dtype annotations are more specific
- graph building currently targets a straight-line subset
- backend support is broader than a stub, but still fixture-limited for Metal and PyTorch
- PyTorch runtime is script-generation based, not a native embedded runtime
- some support modules, such as `ops/linear_support.rs`, are scaffolding rather than fully realized

Knowing these limits prevents misreading a type-system feature as a fully realized backend/runtime
feature.

## Suggested Reading Order

If you are new to the codebase, read in this order:

1. `src/main.rs`
2. `src/lib.rs`
3. `src/compiler/parser.rs`
4. `src/compiler/semantic_analyzer.rs`
5. `src/compiler/frontend_ir.rs`
6. `src/ir/builder.rs`
7. `src/backend/core/execution_plan.rs`
8. `src/runtime/graph_executor.rs`
9. `src/backend/metal/runtime.rs`
10. `src/training/executor.rs`
11. `src/bin/run_tests.rs`

That order matches the real flow of the project and usually gives the fastest understanding.
