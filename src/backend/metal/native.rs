use crate::backend::core::execution_plan::{ExecutionPlan, PlanOp, PlanOpKind};
use crate::backend::metal::codegen::{generate_metal_code, MetalCodegenResult};
use crate::compiler::frontend_ir::LoweredModule;
use crate::compiler::frontend_ir::{FeTypeKind, FeValue};
use crate::runtime::graph_executor::{GraphExecutionResult, GraphRuntimeValue};
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::runtime::layers::{make_embedding_weight, make_linear_bias, make_linear_weight, EmbeddingClosure, LinearClosure};
use crate::runtime::tensor::{make_synthetic_tensor, num_elements, SimpleTensor};

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn tysor_metal_last_error() -> *const c_char;
    fn tysor_metal_context_new(source: *const c_char) -> *mut c_void;
    fn tysor_metal_context_free(context: *mut c_void);
    fn tysor_metal_buffer_new_with_data(context: *mut c_void, data: *const f32, count: usize) -> *mut c_void;
    fn tysor_metal_buffer_new_zeroed(context: *mut c_void, count: usize) -> *mut c_void;
    fn tysor_metal_buffer_free(buffer: *mut c_void);
    fn tysor_metal_buffer_read(buffer: *mut c_void, out_data: *mut f32, count: usize) -> bool;
    fn tysor_metal_dispatch_matmul(
        context: *mut c_void,
        kernel_name: *const c_char,
        lhs: *mut c_void,
        rhs: *mut c_void,
        out: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool;
    fn tysor_metal_dispatch_unary(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        count: u32,
    ) -> bool;
    fn tysor_metal_dispatch_binary_tt(
        context: *mut c_void,
        kernel_name: *const c_char,
        lhs: *mut c_void,
        rhs: *mut c_void,
        out: *mut c_void,
        count: u32,
    ) -> bool;
    fn tysor_metal_dispatch_binary_ts(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        count: u32,
    ) -> bool;
    fn tysor_metal_dispatch_binary_st(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        count: u32,
    ) -> bool;
    fn tysor_metal_dispatch_rms_norm(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        rows: u32,
        width: u32,
    ) -> bool;
    fn tysor_metal_dispatch_softmax(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        rows: u32,
        width: u32,
    ) -> bool;
    fn tysor_metal_dispatch_cross_entropy(
        context: *mut c_void,
        kernel_name: *const c_char,
        logits: *mut c_void,
        target: *mut c_void,
        out: *mut c_void,
        rows: u32,
        width: u32,
    ) -> bool;
    fn tysor_metal_dispatch_linear(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        out: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        with_bias: bool,
    ) -> bool;
    fn tysor_metal_dispatch_embedding(
        context: *mut c_void,
        kernel_name: *const c_char,
        indices: *mut c_void,
        weight: *mut c_void,
        out: *mut c_void,
        index_count: u32,
        embedding_dim: u32,
    ) -> bool;
    fn tysor_metal_dispatch_repeat_kv(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        outer: u32,
        out_heads: u32,
        inner: u32,
        repeats: u32,
    ) -> bool;
    fn tysor_metal_dispatch_causal_mask(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        outer: u32,
        q: u32,
        k: u32,
    ) -> bool;
    fn tysor_metal_dispatch_rope(
        context: *mut c_void,
        kernel_name: *const c_char,
        input: *mut c_void,
        out: *mut c_void,
        outer: u32,
        seq_len: u32,
        half_dim: u32,
    ) -> bool;
}

#[cfg(not(target_os = "macos"))]
pub fn try_execute_metal_module_native(
    _lowered: &LoweredModule,
    _options: &RuntimeRunOptions,
    _parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<GraphExecutionResult, String> {
    Err("Native Metal runtime is only available on macOS".to_string())
}

#[cfg(target_os = "macos")]
fn last_error() -> String {
    unsafe {
        let ptr = tysor_metal_last_error();
        if ptr.is_null() {
            "unknown Metal bridge error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(target_os = "macos")]
fn make_tensor_argument(
    value: &crate::backend::core::execution_plan::PlanValue,
    options: &RuntimeRunOptions,
) -> Result<SimpleTensor, String> {
    let shape = options
        .tensor_shapes
        .get(&value.name)
        .ok_or_else(|| format!("Missing --shape for tensor parameter '{}'", value.name))?;
    Ok(make_synthetic_tensor(
        shape,
        value.ty.tensor_dtype.clone().unwrap_or_else(|| "float32".to_string()),
    ))
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum HostValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Linear(LinearClosure),
    Embedding(EmbeddingClosure),
    Activation { op: String, probability: f64 },
}

#[cfg(target_os = "macos")]
fn require_int(value: &HostValue) -> Result<i64, String> {
    match value {
        HostValue::Int(value) => Ok(*value),
        _ => Err("Native Metal runtime expected integer value".to_string()),
    }
}

#[cfg(target_os = "macos")]
fn require_bool(value: &HostValue) -> Result<bool, String> {
    match value {
        HostValue::Bool(value) => Ok(*value),
        _ => Err("Native Metal runtime expected bool value".to_string()),
    }
}

#[cfg(target_os = "macos")]
fn build_linear_closure(
    op: &PlanOp,
    plan: &ExecutionPlan,
    host_values: &BTreeMap<usize, HostValue>,
) -> Result<LinearClosure, String> {
    let output = &plan.values[op.output];
    let mut closure = LinearClosure {
        dtype: output
            .ty
            .callable_return
            .as_deref()
            .and_then(|ty| ty.tensor_dtype.clone())
            .unwrap_or_else(|| "float32".to_string()),
        ..LinearClosure::default()
    };
    if op.inputs.len() == 1 {
        closure.out_features = require_int(host_values.get(&op.inputs[0]).ok_or_else(|| "Missing out_features".to_string())?)?;
        return Ok(closure);
    }
    if op.inputs.len() == 2 {
        if let Ok(with_bias) = require_bool(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing with_bias".to_string())?) {
            closure.out_features = require_int(host_values.get(&op.inputs[0]).ok_or_else(|| "Missing out_features".to_string())?)?;
            closure.with_bias = with_bias;
            return Ok(closure);
        }
    }
    closure.in_features =
        Some(require_int(host_values.get(&op.inputs[0]).ok_or_else(|| "Missing in_features".to_string())?)?);
    closure.out_features =
        require_int(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing out_features".to_string())?)?;
    if op.inputs.len() == 3 {
        closure.with_bias = require_bool(host_values.get(&op.inputs[2]).ok_or_else(|| "Missing with_bias".to_string())?)?;
    }
    Ok(closure)
}

#[cfg(target_os = "macos")]
fn infer_output_shape(
    op: &PlanOp,
    shapes: &BTreeMap<usize, Vec<i64>>,
    host_values: &BTreeMap<usize, HostValue>,
) -> Result<Vec<i64>, String> {
    match op.kind {
        PlanOpKind::PrimitiveCall => match op.op.as_str() {
            "matmul" => {
                let lhs = shapes.get(&op.inputs[0]).ok_or_else(|| "Missing matmul lhs shape".to_string())?;
                let rhs = shapes.get(&op.inputs[1]).ok_or_else(|| "Missing matmul rhs shape".to_string())?;
                Ok(vec![lhs[0], rhs[1]])
            }
            "relu" | "scale" => shapes
                .get(&op.inputs[0])
                .cloned()
                .ok_or_else(|| "Missing unary input shape".to_string()),
            _ => Err("Unsupported primitive op in native Metal path".to_string()),
        },
        PlanOpKind::Binary => shapes
            .get(&op.inputs[0])
            .or_else(|| shapes.get(&op.inputs[1]))
            .cloned()
            .ok_or_else(|| "Missing binary input shape".to_string()),
        PlanOpKind::LibraryCall => match op.op.as_str() {
            "rms_norm" => shapes
                .get(&op.inputs[0])
                .cloned()
                .ok_or_else(|| "Missing rms_norm shape".to_string()),
            "cross_entropy" => {
                let logits = shapes.get(&op.inputs[0]).ok_or_else(|| "Missing logits shape".to_string())?;
                Ok(vec![logits[0], 1])
            }
            "reshape" => op.inputs[1..]
                .iter()
                .map(|id| match host_values.get(id).ok_or_else(|| "Missing reshape dim".to_string())? {
                    HostValue::Int(value) => Ok(*value),
                    _ => Err("reshape expects integer dims".to_string()),
                })
                .collect(),
            "repeat_kv" => {
                let mut shape = shapes.get(&op.inputs[0]).ok_or_else(|| "Missing repeat_kv input shape".to_string())?.clone();
                let repeats = match host_values.get(&op.inputs[1]).ok_or_else(|| "Missing repeats".to_string())? {
                    HostValue::Int(value) => *value,
                    _ => return Err("repeat_kv expects integer repeats".to_string()),
                };
                shape[1] *= repeats;
                Ok(shape)
            }
            "flatten_heads" => {
                let input = shapes.get(&op.inputs[0]).ok_or_else(|| "Missing flatten_heads input shape".to_string())?;
                if input.len() < 3 {
                    Ok(input.clone())
                } else {
                    let mut shape = input[..input.len() - 2].to_vec();
                    shape.push(input[input.len() - 2] * input[input.len() - 1]);
                    Ok(shape)
                }
            }
            "causal_mask" | "rope" => shapes
                .get(&op.inputs[0])
                .cloned()
                .ok_or_else(|| format!("Missing {} input shape", op.op)),
            _ => Err("Unsupported library call in native Metal path".to_string()),
        },
        PlanOpKind::Apply => {
            let input = shapes.get(&op.inputs[1]).ok_or_else(|| "Missing apply input shape".to_string())?;
            let callee = host_values.get(&op.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
            match callee {
                HostValue::Linear(closure) => Ok(vec![input[0], closure.out_features]),
                HostValue::Embedding(closure) => {
                    let mut shape = input.clone();
                    shape.push(closure.embedding_dim);
                    Ok(shape)
                }
                HostValue::Activation { .. } => Ok(input.clone()),
                _ => Err("Unsupported apply shape inference in native Metal path".to_string()),
            }
        }
        _ => Err("Unsupported native Metal shape inference".to_string()),
    }
}

#[cfg(target_os = "macos")]
struct NativeBuffer(*mut c_void);

#[cfg(target_os = "macos")]
impl Drop for NativeBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                tysor_metal_buffer_free(self.0);
            }
        }
    }
}

#[cfg(target_os = "macos")]
struct NativeContext(*mut c_void);

#[cfg(target_os = "macos")]
impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                tysor_metal_context_free(self.0);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn supported_native_op(op: &PlanOp, host_values: &BTreeMap<usize, HostValue>) -> bool {
    match op.kind {
        PlanOpKind::Constant | PlanOpKind::LibraryCtor => true,
        PlanOpKind::PrimitiveCall => matches!(op.op.as_str(), "matmul" | "relu" | "scale"),
        PlanOpKind::Binary => true,
        PlanOpKind::LibraryCall => matches!(
            op.op.as_str(),
            "rms_norm" | "cross_entropy" | "reshape" | "repeat_kv" | "flatten_heads" | "causal_mask" | "rope"
        ),
        PlanOpKind::Apply => matches!(
            host_values.get(&op.inputs[0]),
            Some(HostValue::Linear(_)) | Some(HostValue::Embedding(_)) | Some(HostValue::Activation { .. })
        ),
    }
}

#[cfg(target_os = "macos")]
fn make_context(codegen: &MetalCodegenResult) -> Result<NativeContext, String> {
    let source = CString::new(codegen.source.as_str()).map_err(|_| "Metal source contains interior NUL".to_string())?;
    let handle = unsafe { tysor_metal_context_new(source.as_ptr()) };
    if handle.is_null() {
        return Err(last_error());
    }
    Ok(NativeContext(handle))
}

#[cfg(target_os = "macos")]
fn buffer_from_tensor(context: &NativeContext, tensor: &SimpleTensor) -> Result<NativeBuffer, String> {
    let handle = unsafe { tysor_metal_buffer_new_with_data(context.0, tensor.data.as_ptr(), tensor.data.len()) };
    if handle.is_null() {
        return Err(last_error());
    }
    Ok(NativeBuffer(handle))
}

#[cfg(target_os = "macos")]
fn zero_buffer(context: &NativeContext, count: usize) -> Result<NativeBuffer, String> {
    let handle = unsafe { tysor_metal_buffer_new_zeroed(context.0, count) };
    if handle.is_null() {
        return Err(last_error());
    }
    Ok(NativeBuffer(handle))
}

#[cfg(target_os = "macos")]
fn require_kernel_name(codegen: &MetalCodegenResult, op_index: usize) -> Result<CString, String> {
    let kernel = codegen
        .kernels
        .iter()
        .find(|kernel| kernel.op_index == op_index)
        .ok_or_else(|| format!("Missing Metal kernel for op {op_index}"))?;
    CString::new(kernel.function_name.as_str()).map_err(|_| "kernel name contains interior NUL".to_string())
}

#[cfg(target_os = "macos")]
fn linear_weight_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("linear_{callee_id}_weight"), format!("{}.weight", value.name)]
}

#[cfg(target_os = "macos")]
fn linear_bias_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("linear_{callee_id}_bias"), format!("{}.bias", value.name)]
}

#[cfg(target_os = "macos")]
fn embedding_weight_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("embedding_{callee_id}_weight"), format!("{}.weight", value.name)]
}

#[cfg(target_os = "macos")]
pub fn try_execute_metal_module_native(
    lowered: &LoweredModule,
    options: &RuntimeRunOptions,
    parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<GraphExecutionResult, String> {
    let entry = lowered
        .functions
        .iter()
        .find(|function| function.name == options.entry)
        .ok_or_else(|| format!("Entry function '{}' not found in lowered module", options.entry))?;
    let plan = crate::backend::core::execution_plan::compile_function_execution_plan(entry, crate::backend::core::kind::BackendKind::Metal)?;
    let codegen = generate_metal_code(&plan)?;
    let context = make_context(&codegen)?;

    let mut host_values = BTreeMap::new();
    let mut shapes = BTreeMap::new();
    let mut buffers: BTreeMap<usize, NativeBuffer> = BTreeMap::new();

    for step in &plan.steps {
        match step.kind {
            crate::backend::core::execution_plan::PlanStepKind::AllocateHostValue => {
                let value = &plan.values[step.value_id];
                if value.is_parameter {
                    if value.ty.kind != FeTypeKind::Tensor {
                        return Err("Native Metal runtime currently supports tensor parameters only".to_string());
                    }
                    let tensor = make_tensor_argument(value, options)?;
                    shapes.insert(value.id, tensor.shape.clone());
                    buffers.insert(value.id, buffer_from_tensor(&context, &tensor)?);
                }
            }
            crate::backend::core::execution_plan::PlanStepKind::AllocateDeviceValue => {}
            crate::backend::core::execution_plan::PlanStepKind::UploadToDevice => {}
            crate::backend::core::execution_plan::PlanStepKind::DispatchDeviceOp => {
                let op_index = step.op_index.ok_or_else(|| "Native Metal dispatch step missing op index".to_string())?;
                let op = &plan.ops[op_index];
                match op.kind {
                    PlanOpKind::Constant => {
                        let value = match &op.constant {
                            FeValue::Int(value) => HostValue::Int(*value),
                            FeValue::Float(value) => HostValue::Float(*value),
                            FeValue::Bool(value) => HostValue::Bool(*value),
                            _ => return Err("Unsupported native Metal constant".to_string()),
                        };
                        host_values.insert(op.output, value);
                    }
                    PlanOpKind::LibraryCtor => {
                        if op.op == "linear" {
                            host_values.insert(op.output, HostValue::Linear(build_linear_closure(op, &plan, &host_values)?));
                        } else if op.op == "Embedding" {
                            host_values.insert(
                                op.output,
                                HostValue::Embedding(EmbeddingClosure {
                                    num_embeddings: require_int(
                                        host_values
                                            .get(&op.inputs[0])
                                            .ok_or_else(|| "Missing num_embeddings".to_string())?,
                                    )?,
                                    embedding_dim: require_int(
                                        host_values
                                            .get(&op.inputs[1])
                                            .ok_or_else(|| "Missing embedding_dim".to_string())?,
                                    )?,
                                    dtype: plan.values[op.output]
                                        .ty
                                        .callable_return
                                        .as_deref()
                                        .and_then(|ty| ty.tensor_dtype.clone())
                                        .unwrap_or_else(|| "float32".to_string()),
                                }),
                            );
                        } else if op.op == "SiLU" || op.op == "Softmax" {
                            host_values.insert(
                                op.output,
                                HostValue::Activation {
                                    op: op.op.clone(),
                                    probability: 0.0,
                                },
                            );
                        } else if op.op == "Dropout" {
                            let probability = match host_values.get(&op.inputs[0]).ok_or_else(|| "Missing dropout probability".to_string())? {
                                &HostValue::Int(value) => value as f64,
                                &HostValue::Float(value) => value,
                                &HostValue::Bool(value) => {
                                    if value {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }
                                _ => return Err("Native Metal runtime expected numeric dropout probability".to_string()),
                            };
                            host_values.insert(
                                op.output,
                                HostValue::Activation {
                                    op: "Dropout".to_string(),
                                    probability,
                                },
                            );
                        } else {
                            return Err(format!("Unsupported library constructor '{}' in native Metal path", op.op));
                        }
                    }
                    _ => {
                        if !supported_native_op(op, &host_values) {
                            return Err(format!("Unsupported op '{}' in native Metal path", op.op));
                        }
                        let shape = infer_output_shape(op, &shapes, &host_values)?;
                        let count = num_elements(&shape);
                        let kernel = require_kernel_name(&codegen, op_index)?;
                        let out_buffer = zero_buffer(&context, count)?;
                        match op.kind {
                            PlanOpKind::PrimitiveCall if op.op == "matmul" => {
                                let lhs = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing lhs buffer".to_string())?;
                                let rhs = buffers.get(&op.inputs[1]).ok_or_else(|| "Missing rhs buffer".to_string())?;
                                let m = shape[0] as u32;
                                let n = shape[1] as u32;
                                let k = shapes.get(&op.inputs[0]).ok_or_else(|| "Missing lhs shape".to_string())?[1] as u32;
                                if !unsafe {
                                    tysor_metal_dispatch_matmul(context.0, kernel.as_ptr(), lhs.0, rhs.0, out_buffer.0, m, n, k)
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::PrimitiveCall | PlanOpKind::Binary => {
                                match op.kind {
                                    PlanOpKind::PrimitiveCall => {
                                        let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                        if !unsafe {
                                            tysor_metal_dispatch_unary(context.0, kernel.as_ptr(), input.0, out_buffer.0, count as u32)
                                        } {
                                            return Err(last_error());
                                        }
                                    }
                                    PlanOpKind::Binary => {
                                        let lhs = buffers.get(&op.inputs[0]);
                                        let rhs = buffers.get(&op.inputs[1]);
                                        let ok = match (lhs, rhs) {
                                            (Some(lhs), Some(rhs)) => unsafe {
                                                tysor_metal_dispatch_binary_tt(
                                                    context.0,
                                                    kernel.as_ptr(),
                                                    lhs.0,
                                                    rhs.0,
                                                    out_buffer.0,
                                                    count as u32,
                                                )
                                            },
                                            (Some(lhs), None) => unsafe {
                                                tysor_metal_dispatch_binary_ts(
                                                    context.0,
                                                    kernel.as_ptr(),
                                                    lhs.0,
                                                    out_buffer.0,
                                                    count as u32,
                                                )
                                            },
                                            (None, Some(rhs)) => unsafe {
                                                tysor_metal_dispatch_binary_st(
                                                    context.0,
                                                    kernel.as_ptr(),
                                                    rhs.0,
                                                    out_buffer.0,
                                                    count as u32,
                                                )
                                            },
                                            _ => return Err("Native Metal path does not support scalar-scalar binary ops".to_string()),
                                        };
                                        if !ok {
                                            return Err(last_error());
                                        }
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "rms_norm" => {
                                let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                if !unsafe {
                                    tysor_metal_dispatch_rms_norm(
                                        context.0,
                                        kernel.as_ptr(),
                                        input.0,
                                        out_buffer.0,
                                        shape[0] as u32,
                                        shape[1] as u32,
                                    )
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "cross_entropy" => {
                                let logits = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing logits buffer".to_string())?;
                                let target = buffers.get(&op.inputs[1]).ok_or_else(|| "Missing target buffer".to_string())?;
                                let logits_shape =
                                    shapes.get(&op.inputs[0]).ok_or_else(|| "Missing logits shape".to_string())?;
                                if !unsafe {
                                    tysor_metal_dispatch_cross_entropy(
                                        context.0,
                                        kernel.as_ptr(),
                                        logits.0,
                                        target.0,
                                        out_buffer.0,
                                        logits_shape[0] as u32,
                                        logits_shape[1] as u32,
                                    )
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "reshape" || op.op == "flatten_heads" => {
                                let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                if !unsafe {
                                    tysor_metal_dispatch_unary(context.0, kernel.as_ptr(), input.0, out_buffer.0, count as u32)
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "repeat_kv" => {
                                let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                let input_shape =
                                    shapes.get(&op.inputs[0]).ok_or_else(|| "Missing repeat_kv input shape".to_string())?;
                                let repeats = match host_values
                                    .get(&op.inputs[1])
                                    .ok_or_else(|| "Missing repeats".to_string())?
                                {
                                    HostValue::Int(value) => *value as u32,
                                    _ => return Err("repeat_kv expects integer repeats".to_string()),
                                };
                                let outer = input_shape[0] as u32;
                                let out_heads = shape[1] as u32;
                                let inner = input_shape[2..].iter().product::<i64>() as u32;
                                if !unsafe {
                                    tysor_metal_dispatch_repeat_kv(
                                        context.0,
                                        kernel.as_ptr(),
                                        input.0,
                                        out_buffer.0,
                                        outer,
                                        out_heads,
                                        inner,
                                        repeats,
                                    )
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "causal_mask" => {
                                let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                let q = shape[shape.len() - 2] as u32;
                                let k = shape[shape.len() - 1] as u32;
                                let outer = shape[..shape.len() - 2].iter().product::<i64>() as u32;
                                if !unsafe {
                                    tysor_metal_dispatch_causal_mask(
                                        context.0,
                                        kernel.as_ptr(),
                                        input.0,
                                        out_buffer.0,
                                        outer,
                                        q,
                                        k,
                                    )
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::LibraryCall if op.op == "rope" => {
                                let input = buffers.get(&op.inputs[0]).ok_or_else(|| "Missing input buffer".to_string())?;
                                let head_dim = shape[shape.len() - 1] as u32;
                                let seq_len = shape[shape.len() - 2] as u32;
                                let outer = shape[..shape.len() - 2].iter().product::<i64>() as u32;
                                if !unsafe {
                                    tysor_metal_dispatch_rope(
                                        context.0,
                                        kernel.as_ptr(),
                                        input.0,
                                        out_buffer.0,
                                        outer,
                                        seq_len,
                                        head_dim / 2,
                                    )
                                } {
                                    return Err(last_error());
                                }
                            }
                            PlanOpKind::Apply => {
                                let callee = host_values.get(&op.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
                                let input = buffers.get(&op.inputs[1]).ok_or_else(|| "Missing apply input buffer".to_string())?;
                                match callee {
                                    HostValue::Linear(closure) => {
                                        let input_shape =
                                            shapes.get(&op.inputs[1]).ok_or_else(|| "Missing apply input shape".to_string())?;
                                        let input_tensor = SimpleTensor {
                                            shape: input_shape.clone(),
                                            data: vec![0.0; num_elements(input_shape)],
                                            dtype: closure.dtype.clone(),
                                        };
                                        let weight_tensor = linear_weight_keys(&plan, op.inputs[0])
                                            .into_iter()
                                            .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                                            .unwrap_or_else(|| {
                                                make_linear_weight(
                                                    closure.in_features.unwrap_or(input_shape[1]),
                                                    closure.out_features,
                                                    &closure.dtype,
                                                )
                                            });
                                        let weight = buffer_from_tensor(&context, &weight_tensor)?;
                                        let bias_tensor = if closure.with_bias {
                                            Some(
                                                linear_bias_keys(&plan, op.inputs[0])
                                                    .into_iter()
                                                    .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                                                    .unwrap_or_else(|| make_linear_bias(closure.out_features, &closure.dtype)),
                                            )
                                        } else {
                                            None
                                        };
                                        let bias = if let Some(tensor) = &bias_tensor {
                                            Some(buffer_from_tensor(&context, tensor)?)
                                        } else {
                                            None
                                        };
                                        if !unsafe {
                                            tysor_metal_dispatch_linear(
                                                context.0,
                                                kernel.as_ptr(),
                                                input.0,
                                                weight.0,
                                                bias.as_ref().map(|buffer| buffer.0).unwrap_or(std::ptr::null_mut()),
                                                out_buffer.0,
                                                shape[0] as u32,
                                                shape[1] as u32,
                                                input_tensor.shape[1] as u32,
                                                closure.with_bias,
                                            )
                                        } {
                                            return Err(last_error());
                                        }
                                    }
                                    HostValue::Embedding(closure) => {
                                        let weight_tensor = embedding_weight_keys(&plan, op.inputs[0])
                                            .into_iter()
                                            .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                                            .unwrap_or_else(|| {
                                                make_embedding_weight(
                                                    closure.num_embeddings,
                                                    closure.embedding_dim,
                                                    &closure.dtype,
                                                )
                                            });
                                        let weight = buffer_from_tensor(&context, &weight_tensor)?;
                                        let index_shape =
                                            shapes.get(&op.inputs[1]).ok_or_else(|| "Missing embedding input shape".to_string())?;
                                        let index_count = index_shape.iter().product::<i64>() as u32;
                                        if !unsafe {
                                            tysor_metal_dispatch_embedding(
                                                context.0,
                                                kernel.as_ptr(),
                                                input.0,
                                                weight.0,
                                                out_buffer.0,
                                                index_count,
                                                closure.embedding_dim as u32,
                                            )
                                        } {
                                            return Err(last_error());
                                        }
                                    }
                                    HostValue::Activation { op, probability } => {
                                        let _ = probability;
                                        let ok = if op == "Softmax" {
                                            unsafe {
                                                tysor_metal_dispatch_softmax(
                                                    context.0,
                                                    kernel.as_ptr(),
                                                    input.0,
                                                    out_buffer.0,
                                                    shape[0] as u32,
                                                    shape[1] as u32,
                                                )
                                            }
                                        } else {
                                            unsafe {
                                                tysor_metal_dispatch_unary(
                                                    context.0,
                                                    kernel.as_ptr(),
                                                    input.0,
                                                    out_buffer.0,
                                                    count as u32,
                                                )
                                            }
                                        };
                                        if !ok {
                                            return Err(last_error());
                                        }
                                    }
                                    _ => return Err("Unsupported apply in native Metal path".to_string()),
                                }
                            }
                            _ => return Err("Unsupported native Metal dispatch".to_string()),
                        }
                        shapes.insert(op.output, shape);
                        buffers.insert(op.output, out_buffer);
                    }
                }
            }
            crate::backend::core::execution_plan::PlanStepKind::DownloadToHost
            | crate::backend::core::execution_plan::PlanStepKind::MaterializeOutput => {}
            crate::backend::core::execution_plan::PlanStepKind::ExecuteOp => {
                return Err("Native Metal runtime cannot execute local steps".to_string());
            }
        }
    }

    let mut result = GraphExecutionResult::default();
    for value in &plan.values {
        if value.ty.kind != FeTypeKind::Tensor {
            if let Some(host_value) = host_values.get(&value.id) {
                match host_value {
                    HostValue::Int(int_value) => {
                        result.values.insert(value.id, GraphRuntimeValue::Int(int_value.clone()));
                    }
                    HostValue::Float(float_value) => {
                        result.values.insert(value.id, GraphRuntimeValue::Float(float_value.clone()));
                    }
                    HostValue::Bool(bool_value) => {
                        result.values.insert(value.id, GraphRuntimeValue::Bool(bool_value.clone()));
                    }
                    HostValue::Linear(_) | HostValue::Embedding(_) | HostValue::Activation { .. } => {}
                }
            }
            continue;
        }
        let shape = shapes.get(&value.id).ok_or_else(|| format!("Missing shape for value {}", value.id))?;
        let mut data = vec![0.0f32; num_elements(shape)];
        let buffer = buffers.get(&value.id).ok_or_else(|| format!("Missing buffer for value {}", value.id))?;
        if !unsafe { tysor_metal_buffer_read(buffer.0, data.as_mut_ptr(), data.len()) } {
            return Err(last_error());
        }
        let tensor = SimpleTensor {
            shape: shape.clone(),
            data,
            dtype: value.ty.tensor_dtype.clone().unwrap_or_else(|| "float32".to_string()),
        };
        result.values.insert(value.id, GraphRuntimeValue::Tensor(tensor.clone()));
        if plan.outputs.contains(&value.id) {
            result.outputs.insert(value.id, GraphRuntimeValue::Tensor(tensor));
        }
    }
    Ok(result)
}
