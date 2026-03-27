# Tysor Language Overview

`tysor` is a DSL for defining and experimenting with deep learning models with minimal boilerplate.

## Core Constructs

- `layer`: graph-building, parameter-owning computation
- `fn`: helper computation without tracked parameter-owning state
- `config`: compile-time structured configuration
- `layer model`: forward/model entrypoint
- `train model`: training entrypoint

## `layer`

`layer` is the main graph-building construct.

Rules:

- a `layer` may create tracked or trainable state
- a `layer` may call `layer`
- a `layer` may call `fn`

Typical examples include `linear(...)` and `Embedding(...)`.

## `fn`

`fn` is helper computation.

Rules:

- `fn` may call `fn`
- `fn` may not call `layer`
- `fn` does not introduce tracked parameter-owning state
- tensor computation inside `fn` may still participate in a larger model computation

## `config`

`config` is a compile-time structured record.

Typical uses:

- architecture sizes
- dtype policy
- compile-time knobs

## `layer model`

`layer model` is the forward/model execution entrypoint.

It defines the function the current runtime backends execute.

## `train model`

`train model` is the training entrypoint.

Current design:

- `train model` holds the training specification directly
- tuple-valued fields create multiple training variants
- scalar-valued fields broadcast across all variants
- `backend` / `target` / `device` may specify intended execution metadata
- old `subtrain` orchestration is no longer part of the design

Example:

```tysor
train model:
  backend = metal
  optimizer = adam
  lr = (1e-4, 5e-4, 1e-3)
  objective = loss
  iteration = 5
```

This creates three runs with the same optimizer, objective, iteration count, and backend, while
varying only `lr`.

Current note:

- backend intent is carried through the main pipeline and execution plan
- today the fully working execution paths are the local, Metal, and PyTorch backends supported by
  the runtime and training layers

## Variant Rules

Tuple-valued train fields are temporary collection syntax because the DSL does not yet have
general list syntax.

Rules:

- scalar field values broadcast to every variant
- tuple field values define one value per variant
- all tuple-valued train fields must have the same length
- mismatched tuple lengths are a semantic error

## Training Root

Training roots from a named tensor value, typically a loss tensor.

Rules:

- `objective` must reference a named tensor root
- `objective` may be scalar or tuple-valued under the same variant rules
- objective names must resolve against the surface of `layer model`

## Types

The language is progressively typed.

Surface types may be omitted, but the compiler should resolve enough semantic structure before
lowering.

Scalar integer types:

- `int16`
- `int32`
- `int64`

Tensor types support richer metadata:

- `tensor`
- `tensor[float16]`
- `tensor[float32]`
- `tensor[float64]`
- `tensor[int16]`
- `tensor[int32]`
- `tensor[int64]`
- `tensor[float16, TransformerDims[:2]]`

Scalar float types:

- `float16`
- `float32`
- `float64`

## Callable Layer Constructors

Some builtins produce callable layer-like values.

Example:

```tysor
proj = linear(128, 128)
return proj(x)
```

Semantically:

- `linear(...)` is a parameter-owning layer constructor
- the result is a callable value
- invoking that callable inside a `layer` is part of model execution semantics

Current implementation direction:

- frontend IR represents `linear(...)` as an explicit layer-constructor node
- invoking a local callable lowers as `Apply`

## Arrow Composition

`->` is function-composition sugar inside `layer` code.

Core rule:

```tysor
x -> f() -> g() -> h()
```

means:

```tysor
h(g(f(x)))
```
