//! Rust implementation of the tysor compiler/runtime stack.
//!
//! The main phase split is:
//! - `compiler`: source -> tokens -> AST -> semantic analysis -> frontend IR
//! - `ir`: frontend IR -> graph form
//! - `backend`: graph -> backend-specific execution/export plans
//! - `runtime`: tensor helpers and execution support
//! - `training`: backward pass and SGD training loop
//! - `ops`: builtin metadata shared by lowering/runtime code

pub mod backend;
pub mod compiler;
pub mod ir;
pub mod ops;
pub mod runtime;
pub mod training;
