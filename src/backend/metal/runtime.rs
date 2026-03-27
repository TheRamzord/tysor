//! Metal runtime entrypoints.
//!
//! This module prefers the native Metal path from `native.rs`, but it also keeps a
//! Rust fallback interpreter for environments without a usable Metal device.

use crate::backend::core::execution_plan::{compile_function_execution_plan, ExecutionPlan, PlanOpKind, PlanStepKind};
use crate::backend::core::kind::BackendKind;
use crate::backend::metal::native::try_execute_metal_module_native;
use crate::compiler::frontend_ir::{FeTypeKind, FeValue, LoweredModule};
use crate::compiler::lexer::TokenType;
use crate::runtime::graph_executor::{GraphExecutionResult, GraphRuntimeValue};
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::runtime::layers::{
    apply_causal_mask, apply_cross_entropy, apply_dropout, apply_embedding_with_parameters, apply_flatten_heads,
    apply_linear_with_parameters, apply_repeat_kv, apply_reshape, apply_rms_norm, apply_rope, make_embedding_weight,
    make_linear_bias, make_linear_weight, EmbeddingClosure, LinearClosure,
};
use crate::runtime::tensor::{
    apply_relu, apply_silu, apply_softmax, elementwise_binary, make_synthetic_tensor, print_tensor, scalar_tensor_binary,
    tensor_scalar_binary, SimpleTensor,
};

use std::collections::BTreeMap;
use std::env;

#[derive(Debug, Clone, PartialEq)]
enum HostValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Linear(LinearClosure),
    Embedding(EmbeddingClosure),
    Activation { op: String, probability: f64 },
    Tensor(SimpleTensor),
}

fn find_entry_function<'a>(
    lowered: &'a LoweredModule,
    entry: &str,
) -> Result<&'a crate::compiler::frontend_ir::FeFunction, String> {
    lowered
        .functions
        .iter()
        .find(|function| function.name == entry)
        .ok_or_else(|| format!("Entry function '{}' not found in lowered module", entry))
}

fn require_int(value: &HostValue) -> Result<i64, String> {
    match value {
        HostValue::Int(value) => Ok(*value),
        _ => Err("Metal runtime expected integer value".to_string()),
    }
}

fn require_bool(value: &HostValue) -> Result<bool, String> {
    match value {
        HostValue::Bool(value) => Ok(*value),
        _ => Err("Metal runtime expected bool value".to_string()),
    }
}

fn require_number(value: &HostValue) -> Result<f64, String> {
    match value {
        HostValue::Int(value) => Ok(*value as f64),
        HostValue::Float(value) => Ok(*value),
        HostValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err("Metal runtime expected scalar value".to_string()),
    }
}

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

fn build_linear_closure(
    op: &crate::backend::core::execution_plan::PlanOp,
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

fn linear_weight_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("linear_{callee_id}_weight"), format!("{}.weight", value.name)]
}

fn linear_bias_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("linear_{callee_id}_bias"), format!("{}.bias", value.name)]
}

fn embedding_weight_keys(plan: &ExecutionPlan, callee_id: usize) -> Vec<String> {
    let value = &plan.values[callee_id];
    vec![format!("embedding_{callee_id}_weight"), format!("{}.weight", value.name)]
}

fn execute_device_op(
    plan: &ExecutionPlan,
    op: &crate::backend::core::execution_plan::PlanOp,
    host_values: &mut BTreeMap<usize, HostValue>,
    device_tensors: &mut BTreeMap<usize, SimpleTensor>,
    parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<(), String> {
    // Simulated Metal execution path. It mirrors the execution-plan categories used
    // by the native runtime so behavior stays aligned across both implementations.
    match op.kind {
        PlanOpKind::Constant => {
            let value = match &op.constant {
                FeValue::Int(value) => HostValue::Int(*value),
                FeValue::Float(value) => HostValue::Float(*value),
                FeValue::Bool(value) => HostValue::Bool(*value),
                _ => return Err("Unsupported Metal constant".to_string()),
            };
            host_values.insert(op.output, value);
        }
        PlanOpKind::LibraryCtor => {
            let value = if op.op == "linear" {
                HostValue::Linear(build_linear_closure(op, plan, host_values)?)
            } else if op.op == "Embedding" {
                HostValue::Embedding(EmbeddingClosure {
                    num_embeddings: require_int(host_values.get(&op.inputs[0]).ok_or_else(|| "Missing num_embeddings".to_string())?)?,
                    embedding_dim: require_int(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing embedding_dim".to_string())?)?,
                    dtype: plan.values[op.output]
                        .ty
                        .callable_return
                        .as_deref()
                        .and_then(|ty| ty.tensor_dtype.clone())
                        .unwrap_or_else(|| "float32".to_string()),
                })
            } else if op.op == "SiLU" || op.op == "Softmax" {
                HostValue::Activation {
                    op: op.op.clone(),
                    probability: 0.0,
                }
            } else if op.op == "Dropout" {
                HostValue::Activation {
                    op: "Dropout".to_string(),
                    probability: require_number(
                        host_values.get(&op.inputs[0]).ok_or_else(|| "Missing probability".to_string())?,
                    )?,
                }
            } else {
                return Err(format!("Unsupported Metal library constructor '{}'", op.op));
            };
            host_values.insert(op.output, value);
        }
        PlanOpKind::PrimitiveCall => {
            let tensor = match op.op.as_str() {
                "matmul" => crate::runtime::tensor::matmul(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing lhs tensor".to_string())?,
                    device_tensors.get(&op.inputs[1]).ok_or_else(|| "Missing rhs tensor".to_string())?,
                )?,
                "relu" => apply_relu(device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing relu input".to_string())?),
                "scale" => tensor_scalar_binary(
                    TokenType::Star,
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing scale input".to_string())?,
                    require_number(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing scale factor".to_string())?)?,
                )?,
                _ => return Err(format!("Unsupported Metal primitive op '{}'", op.op)),
            };
            device_tensors.insert(op.output, tensor);
        }
        PlanOpKind::Binary => {
            let lhs_tensor = device_tensors.get(&op.inputs[0]);
            let rhs_tensor = device_tensors.get(&op.inputs[1]);
            let tensor = if let (Some(lhs), Some(rhs)) = (lhs_tensor, rhs_tensor) {
                elementwise_binary(op.binary_op, lhs, rhs)?
            } else if let Some(lhs) = lhs_tensor {
                tensor_scalar_binary(
                    op.binary_op,
                    lhs,
                    require_number(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing rhs".to_string())?)?,
                )?
            } else if let Some(rhs) = rhs_tensor {
                scalar_tensor_binary(
                    op.binary_op,
                    require_number(host_values.get(&op.inputs[0]).ok_or_else(|| "Missing lhs".to_string())?)?,
                    rhs,
                )?
            } else {
                return Err("Metal runtime does not support scalar-scalar binary ops".to_string());
            };
            device_tensors.insert(op.output, tensor);
        }
        PlanOpKind::LibraryCall => {
            let tensor = if op.op == "rms_norm" {
                apply_rms_norm(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing rms_norm input".to_string())?,
                    require_int(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing hidden size".to_string())?)?,
                )?
            } else if op.op == "cross_entropy" {
                apply_cross_entropy(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing logits".to_string())?,
                    device_tensors.get(&op.inputs[1]).ok_or_else(|| "Missing target".to_string())?,
                )?
            } else if op.op == "reshape" {
                let shape = op.inputs[1..]
                    .iter()
                    .map(|id| require_int(host_values.get(id).ok_or_else(|| "Missing reshape dim".to_string())?))
                    .collect::<Result<Vec<_>, _>>()?;
                apply_reshape(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing reshape input".to_string())?,
                    &shape,
                )?
            } else if op.op == "repeat_kv" {
                apply_repeat_kv(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing repeat_kv input".to_string())?,
                    require_int(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing repeats".to_string())?)?,
                )?
            } else if op.op == "flatten_heads" {
                apply_flatten_heads(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing flatten_heads input".to_string())?,
                )?
            } else if op.op == "causal_mask" {
                apply_causal_mask(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing causal_mask input".to_string())?,
                )?
            } else if op.op == "rope" {
                apply_rope(
                    device_tensors.get(&op.inputs[0]).ok_or_else(|| "Missing rope input".to_string())?,
                    require_int(host_values.get(&op.inputs[1]).ok_or_else(|| "Missing rope head_dim".to_string())?)?,
                    require_number(host_values.get(&op.inputs[2]).ok_or_else(|| "Missing rope theta".to_string())?)?,
                )?
            } else {
                return Err(format!("Unsupported Metal library op '{}'", op.op));
            };
            device_tensors.insert(op.output, tensor);
        }
        PlanOpKind::Apply => {
            let callee = host_values.get(&op.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
            let input = device_tensors.get(&op.inputs[1]).ok_or_else(|| "Missing apply input".to_string())?;
            let tensor = match callee {
                HostValue::Linear(closure) => {
                    let weight = linear_weight_keys(plan, op.inputs[0])
                        .into_iter()
                        .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                        .unwrap_or_else(|| {
                            make_linear_weight(
                                closure.in_features.unwrap_or_else(|| *input.shape.get(1).unwrap_or(&0)),
                                closure.out_features,
                                &closure.dtype,
                            )
                        });
                    let bias = if closure.with_bias {
                        linear_bias_keys(plan, op.inputs[0])
                            .into_iter()
                            .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                            .or_else(|| Some(make_linear_bias(closure.out_features, &closure.dtype)))
                    } else {
                        None
                    };
                    apply_linear_with_parameters(input, &weight, bias.as_ref())?
                }
                HostValue::Embedding(closure) => {
                    let weight = embedding_weight_keys(plan, op.inputs[0])
                        .into_iter()
                        .find_map(|key| parameters.and_then(|params| params.get(&key).cloned()))
                        .unwrap_or_else(|| {
                            make_embedding_weight(
                                closure.num_embeddings,
                                closure.embedding_dim,
                                &closure.dtype,
                            )
                        });
                    apply_embedding_with_parameters(
                        input,
                        &weight,
                        closure.num_embeddings,
                        closure.embedding_dim,
                    )?
                }
                HostValue::Activation { op, probability } => match op.as_str() {
                    "SiLU" => apply_silu(input),
                    "Softmax" => apply_softmax(input)?,
                    "Dropout" => apply_dropout(input, *probability)?,
                    _ => return Err("Unsupported Metal activation apply".to_string()),
                },
                _ => return Err("Unsupported Metal apply op".to_string()),
            };
            device_tensors.insert(op.output, tensor);
        }
    }
    Ok(())
}

pub fn execute_metal_module(
    lowered: &LoweredModule,
    options: &RuntimeRunOptions,
    parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<GraphExecutionResult, String> {
    let allow_fallback = env::var("TYSOR_METAL_ALLOW_FALLBACK")
        .map(|value| value != "0" && !value.is_empty())
        .unwrap_or(false);
    let require_native = env::var("TYSOR_METAL_REQUIRE_NATIVE")
        .map(|value| value != "0" && !value.is_empty())
        .unwrap_or(false);
    // Native Metal is preferred whenever it works. The fallback keeps the backend
    // usable in CI or on developer machines that cannot expose a default Metal device.
    match try_execute_metal_module_native(lowered, options, parameters) {
        Ok(execution) => return Ok(execution),
        Err(error) if require_native => return Err(format!("Native Metal execution required: {error}")),
        Err(error) if cfg!(target_os = "macos") && !allow_fallback && !error.contains("could not create a default device") => {
            return Err(format!("Native Metal execution failed: {error}. Set TYSOR_METAL_ALLOW_FALLBACK=1 to force the simulated fallback path."));
        }
        Err(_) => {}
    }
    let entry = find_entry_function(lowered, &options.entry)?;
    let plan = compile_function_execution_plan(entry, BackendKind::Metal)?;

    let mut host_values = BTreeMap::new();
    let mut device_tensors = BTreeMap::new();
    let mut result = GraphExecutionResult::default();

    for step in &plan.steps {
        match step.kind {
            PlanStepKind::AllocateHostValue => {
                let value = &plan.values[step.value_id];
                if value.is_parameter {
                    if value.ty.kind != FeTypeKind::Tensor {
                        return Err("Metal runtime currently supports tensor parameters only".to_string());
                    }
                    let tensor = make_tensor_argument(value, options)?;
                    host_values.insert(value.id, HostValue::Tensor(tensor.clone()));
                    device_tensors.insert(value.id, tensor);
                }
            }
            PlanStepKind::AllocateDeviceValue => {}
            PlanStepKind::UploadToDevice => {
                let value = &plan.values[step.value_id];
                if let Some(HostValue::Tensor(tensor)) = host_values.get(&value.id) {
                    device_tensors.insert(value.id, tensor.clone());
                }
            }
            PlanStepKind::DispatchDeviceOp => {
                let op_index = step.op_index.ok_or_else(|| "Metal dispatch step missing op index".to_string())?;
                let op = &plan.ops[op_index];
                execute_device_op(&plan, op, &mut host_values, &mut device_tensors, parameters)?;
            }
            PlanStepKind::DownloadToHost => {
                if let Some(tensor) = device_tensors.get(&step.value_id).cloned() {
                    host_values.insert(step.value_id, HostValue::Tensor(tensor));
                }
            }
            PlanStepKind::MaterializeOutput => {}
            PlanStepKind::ExecuteOp => return Err("Metal runtime cannot execute local plan steps".to_string()),
        }
    }

    for (id, value) in host_values {
        match value {
            HostValue::Int(value) => {
                result.values.insert(id, GraphRuntimeValue::Int(value));
            }
            HostValue::Float(value) => {
                result.values.insert(id, GraphRuntimeValue::Float(value));
            }
            HostValue::Bool(value) => {
                result.values.insert(id, GraphRuntimeValue::Bool(value));
            }
            HostValue::Tensor(tensor) => {
                result.values.insert(id, GraphRuntimeValue::Tensor(tensor));
            }
            HostValue::Linear(_) | HostValue::Embedding(_) | HostValue::Activation { .. } => {}
        }
    }
    for (id, tensor) in device_tensors {
        result
            .values
            .entry(id)
            .or_insert_with(|| GraphRuntimeValue::Tensor(tensor));
    }
    for output_id in &plan.outputs {
        if let Some(value) = result.values.get(output_id).cloned() {
            result.outputs.insert(*output_id, value);
        }
    }
    Ok(result)
}

pub fn run_metal_module(lowered: &LoweredModule, options: &RuntimeRunOptions) -> Result<(), String> {
    let execution = execute_metal_module(lowered, options, None)?;
    if execution.outputs.len() != 1 {
        return Err("Metal runtime currently supports a single output".to_string());
    }
    let value = execution
        .outputs
        .values()
        .next()
        .ok_or_else(|| "Metal runtime could not resolve output".to_string())?;
    match value {
        GraphRuntimeValue::Tensor(tensor) => {
            print_tensor(tensor);
            Ok(())
        }
        _ => Err("Metal runtime currently supports only tensor outputs".to_string()),
    }
}
