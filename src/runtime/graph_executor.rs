use crate::backend::core::execution_plan::{compile_execution_plan, ExecutionPlan, PlanOpKind, PlanStepKind, Placement};
use crate::backend::core::kind::BackendKind;
use crate::compiler::frontend_ir::{FeTypeKind, FeValue};
use crate::compiler::lexer::TokenType;
use crate::ir::graph::GraphFunction;
use crate::ops::library::{runtime_primitive, RuntimePrimitiveKind};
use crate::runtime::layers::{
    apply_causal_mask, apply_cross_entropy, apply_dropout, apply_embedding_with_parameters, apply_flatten_heads,
    apply_linear, apply_repeat_kv, apply_reshape, apply_rms_norm, apply_rope, make_embedding_weight,
    EmbeddingClosure, LinearClosure,
};
use crate::runtime::tensor::{
    apply_relu, apply_silu, apply_softmax, elementwise_binary, make_synthetic_tensor, matmul,
    scalar_tensor_binary, tensor_scalar_binary, SimpleTensor,
};

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphExecutorOptions {
    pub tensor_shapes: BTreeMap<String, Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphRuntimeValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Tensor(SimpleTensor),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphExecutionResult {
    pub values: BTreeMap<usize, GraphRuntimeValue>,
    pub outputs: BTreeMap<usize, GraphRuntimeValue>,
}

#[derive(Debug, Clone, PartialEq)]
enum RuntimeValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Tensor(SimpleTensor),
    Linear(LinearClosure),
    Embedding(EmbeddingClosure),
    Activation { op: String, probability: f64 },
}

fn require_number(value: &RuntimeValue) -> Result<f64, String> {
    match value {
        RuntimeValue::Int(value) => Ok(*value as f64),
        RuntimeValue::Float(value) => Ok(*value),
        RuntimeValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err("Graph executor expected scalar value".to_string()),
    }
}

fn require_int(value: &RuntimeValue) -> Result<i64, String> {
    match value {
        RuntimeValue::Int(value) => Ok(*value),
        _ => Err("Graph executor expected integer value".to_string()),
    }
}

fn require_bool(value: &RuntimeValue) -> Result<bool, String> {
    match value {
        RuntimeValue::Bool(value) => Ok(*value),
        _ => Err("Graph executor expected bool value".to_string()),
    }
}

fn require_tensor(value: &RuntimeValue) -> Result<&SimpleTensor, String> {
    match value {
        RuntimeValue::Tensor(tensor) => Ok(tensor),
        _ => Err("Graph executor expected tensor value".to_string()),
    }
}

fn make_tensor_argument(
    value: &crate::backend::core::execution_plan::PlanValue,
    options: &GraphExecutorOptions,
) -> Result<SimpleTensor, String> {
    if value.placement != Placement::Host {
        return Err("Local executor requires host-resident tensor parameters".to_string());
    }
    let shape = options
        .tensor_shapes
        .get(&value.name)
        .ok_or_else(|| format!("Missing --shape for tensor parameter '{}'", value.name))?;
    Ok(make_synthetic_tensor(
        shape,
        value.ty.tensor_dtype.clone().unwrap_or_else(|| "float32".to_string()),
    ))
}

fn execute_primitive(node: &crate::backend::core::execution_plan::PlanOp, values: &BTreeMap<usize, RuntimeValue>) -> Result<RuntimeValue, String> {
    match runtime_primitive(&node.op) {
        RuntimePrimitiveKind::Matmul => Ok(RuntimeValue::Tensor(matmul(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing lhs".to_string())?)?,
            require_tensor(values.get(&node.inputs[1]).ok_or_else(|| "Missing rhs".to_string())?)?,
        )?)),
        RuntimePrimitiveKind::Relu => Ok(RuntimeValue::Tensor(apply_relu(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing input".to_string())?)?,
        ))),
        RuntimePrimitiveKind::Scale => Ok(RuntimeValue::Tensor(tensor_scalar_binary(
            TokenType::Star,
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing tensor".to_string())?)?,
            require_number(values.get(&node.inputs[1]).ok_or_else(|| "Missing scale".to_string())?)?,
        )?)),
        RuntimePrimitiveKind::Unsupported => Err(format!("Unsupported primitive graph op '{}'", node.op)),
    }
}

fn execute_library_call(node: &crate::backend::core::execution_plan::PlanOp, values: &BTreeMap<usize, RuntimeValue>) -> Result<RuntimeValue, String> {
    if node.op == "rms_norm" {
        return Ok(RuntimeValue::Tensor(apply_rms_norm(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing tensor".to_string())?)?,
            require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing hidden size".to_string())?)?,
        )?));
    }
    if node.op == "cross_entropy" {
        return Ok(RuntimeValue::Tensor(apply_cross_entropy(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing logits".to_string())?)?,
            require_tensor(values.get(&node.inputs[1]).ok_or_else(|| "Missing target".to_string())?)?,
        )?));
    }
    if node.op == "reshape" {
        let shape = node.inputs[1..]
            .iter()
            .map(|id| require_int(values.get(id).ok_or_else(|| "Missing reshape dim".to_string())?))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(RuntimeValue::Tensor(apply_reshape(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing reshape input".to_string())?)?,
            &shape,
        )?));
    }
    if node.op == "repeat_kv" {
        return Ok(RuntimeValue::Tensor(apply_repeat_kv(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing repeat_kv input".to_string())?)?,
            require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing repeats".to_string())?)?,
        )?));
    }
    if node.op == "flatten_heads" {
        return Ok(RuntimeValue::Tensor(apply_flatten_heads(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing flatten_heads input".to_string())?)?,
        )?));
    }
    if node.op == "causal_mask" {
        return Ok(RuntimeValue::Tensor(apply_causal_mask(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing causal_mask input".to_string())?)?,
        )?));
    }
    if node.op == "rope" {
        return Ok(RuntimeValue::Tensor(apply_rope(
            require_tensor(values.get(&node.inputs[0]).ok_or_else(|| "Missing rope input".to_string())?)?,
            require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing rope head_dim".to_string())?)?,
            require_number(values.get(&node.inputs[2]).ok_or_else(|| "Missing rope theta".to_string())?)?,
        )?));
    }
    Err(format!("Unsupported library graph call '{}'", node.op))
}

fn execute_library_ctor(
    node: &crate::backend::core::execution_plan::PlanOp,
    output: &crate::backend::core::execution_plan::PlanValue,
    values: &BTreeMap<usize, RuntimeValue>,
) -> Result<RuntimeValue, String> {
    if node.op == "linear" {
        let mut closure = LinearClosure {
            dtype: output
                .ty
                .callable_return
                .as_deref()
                .and_then(|ty| ty.tensor_dtype.clone())
                .unwrap_or_else(|| "float32".to_string()),
            ..LinearClosure::default()
        };
        if node.inputs.len() == 1 {
            closure.out_features = require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing out_features".to_string())?)?;
            return Ok(RuntimeValue::Linear(closure));
        }
        if node.inputs.len() == 2 {
            if let Ok(with_bias) = require_bool(values.get(&node.inputs[1]).ok_or_else(|| "Missing with_bias".to_string())?) {
                closure.out_features = require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing out_features".to_string())?)?;
                closure.with_bias = with_bias;
                return Ok(RuntimeValue::Linear(closure));
            }
        }
        closure.in_features =
            Some(require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing in_features".to_string())?)?);
        closure.out_features =
            require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing out_features".to_string())?)?;
        if node.inputs.len() == 3 {
            closure.with_bias = require_bool(values.get(&node.inputs[2]).ok_or_else(|| "Missing with_bias".to_string())?)?;
        }
        return Ok(RuntimeValue::Linear(closure));
    }
    if node.op == "SiLU" || node.op == "Softmax" {
        return Ok(RuntimeValue::Activation {
            op: node.op.clone(),
            probability: 0.0,
        });
    }
    if node.op == "Embedding" {
        return Ok(RuntimeValue::Embedding(EmbeddingClosure {
            num_embeddings: require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing num_embeddings".to_string())?)?,
            embedding_dim: require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing embedding_dim".to_string())?)?,
            dtype: output
                .ty
                .callable_return
                .as_deref()
                .and_then(|ty| ty.tensor_dtype.clone())
                .unwrap_or_else(|| "float32".to_string()),
        }));
    }
    if node.op == "Dropout" {
        return Ok(RuntimeValue::Activation {
            op: "Dropout".to_string(),
            probability: require_number(values.get(&node.inputs[0]).ok_or_else(|| "Missing probability".to_string())?)?,
        });
    }
    Err(format!("Unsupported library constructor '{}'", node.op))
}

fn execute_apply(node: &crate::backend::core::execution_plan::PlanOp, values: &BTreeMap<usize, RuntimeValue>) -> Result<RuntimeValue, String> {
    let callee = values.get(&node.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
    let input = require_tensor(values.get(&node.inputs[1]).ok_or_else(|| "Missing apply input".to_string())?)?;
    match callee {
        RuntimeValue::Linear(closure) => Ok(RuntimeValue::Tensor(apply_linear(closure, input)?)),
        RuntimeValue::Embedding(closure) => {
            let weight = make_embedding_weight(closure.num_embeddings, closure.embedding_dim, &closure.dtype);
            Ok(RuntimeValue::Tensor(apply_embedding_with_parameters(
                input,
                &weight,
                closure.num_embeddings,
                closure.embedding_dim,
            )?))
        }
        RuntimeValue::Activation { op, probability } => match op.as_str() {
            "SiLU" => Ok(RuntimeValue::Tensor(apply_silu(input))),
            "Softmax" => Ok(RuntimeValue::Tensor(apply_softmax(input)?)),
            "Dropout" => Ok(RuntimeValue::Tensor(apply_dropout(input, *probability)?)),
            _ => Err("Unsupported graph apply operation".to_string()),
        },
        _ => Err("Unsupported graph apply operation".to_string()),
    }
}

pub fn execute_execution_plan(plan: &ExecutionPlan, options: &GraphExecutorOptions) -> Result<GraphExecutionResult, String> {
    if plan.backend != BackendKind::Local {
        return Err("Execution plan backend is not implemented yet".to_string());
    }

    let mut runtime_values = BTreeMap::new();
    let mut materialized_outputs = BTreeMap::new();

    for step in &plan.steps {
        match step.kind {
            PlanStepKind::AllocateHostValue => {
                let value = &plan.values[step.value_id];
                if value.placement != Placement::Host {
                    return Err("AllocateHostValue step requires a host-resident value".to_string());
                }
                if value.is_parameter {
                    if value.ty.kind != FeTypeKind::Tensor {
                        return Err("Graph executor currently supports tensor parameters only".to_string());
                    }
                    runtime_values.insert(value.id, RuntimeValue::Tensor(make_tensor_argument(value, options)?));
                }
            }
            PlanStepKind::AllocateDeviceValue => {
                return Err("Local executor does not support device value allocation".to_string());
            }
            PlanStepKind::ExecuteOp => {
                let op_index = step.op_index.ok_or_else(|| "ExecuteOp step is missing an op index".to_string())?;
                let node = &plan.ops[op_index];
                if node.backend != BackendKind::Local {
                    return Err("Local executor cannot run non-local plan operations".to_string());
                }
                let value = match node.kind {
                    PlanOpKind::Constant => match &node.constant {
                        FeValue::Int(value) => RuntimeValue::Int(*value),
                        FeValue::Float(value) => RuntimeValue::Float(*value),
                        FeValue::Bool(value) => RuntimeValue::Bool(*value),
                        _ => return Err("Unsupported graph constant".to_string()),
                    },
                    PlanOpKind::Binary => {
                        let lhs = runtime_values.get(&node.inputs[0]).ok_or_else(|| "Missing lhs".to_string())?;
                        let rhs = runtime_values.get(&node.inputs[1]).ok_or_else(|| "Missing rhs".to_string())?;
                        if let (RuntimeValue::Tensor(lhs), RuntimeValue::Tensor(rhs)) = (lhs, rhs) {
                            RuntimeValue::Tensor(elementwise_binary(node.binary_op, lhs, rhs)?)
                        } else if let RuntimeValue::Tensor(lhs) = lhs {
                            RuntimeValue::Tensor(tensor_scalar_binary(node.binary_op, lhs, require_number(rhs)?)?)
                        } else if let RuntimeValue::Tensor(rhs) = rhs {
                            RuntimeValue::Tensor(scalar_tensor_binary(node.binary_op, require_number(lhs)?, rhs)?)
                        } else {
                            let left = require_number(lhs)?;
                            let right = require_number(rhs)?;
                            match node.binary_op {
                                TokenType::Plus => RuntimeValue::Float(left + right),
                                TokenType::Minus => RuntimeValue::Float(left - right),
                                TokenType::Star => RuntimeValue::Float(left * right),
                                TokenType::Slash => RuntimeValue::Float(left / right),
                                _ => return Err("Unsupported scalar graph binary op".to_string()),
                            }
                        }
                    }
                    PlanOpKind::PrimitiveCall => execute_primitive(node, &runtime_values)?,
                    PlanOpKind::LibraryCall => execute_library_call(node, &runtime_values)?,
                    PlanOpKind::LibraryCtor => execute_library_ctor(node, &plan.values[node.output], &runtime_values)?,
                    PlanOpKind::Apply => execute_apply(node, &runtime_values)?,
                };
                runtime_values.insert(node.output, value);
            }
            PlanStepKind::MaterializeOutput => {
                let value = runtime_values
                    .get(&step.value_id)
                    .ok_or_else(|| "Runtime interpreter could not resolve the graph output value".to_string())?;
                let output = match value {
                    RuntimeValue::Int(value) => GraphRuntimeValue::Int(*value),
                    RuntimeValue::Float(value) => GraphRuntimeValue::Float(*value),
                    RuntimeValue::Bool(value) => GraphRuntimeValue::Bool(*value),
                    RuntimeValue::Tensor(tensor) => GraphRuntimeValue::Tensor(tensor.clone()),
                    RuntimeValue::Linear(_) | RuntimeValue::Embedding(_) | RuntimeValue::Activation { .. } => continue,
                };
                materialized_outputs.insert(step.value_id, output);
            }
            PlanStepKind::UploadToDevice | PlanStepKind::DispatchDeviceOp | PlanStepKind::DownloadToHost => {
                return Err("Device execution plan steps are not implemented yet".to_string());
            }
        }
    }

    let mut result = GraphExecutionResult {
        outputs: materialized_outputs,
        ..GraphExecutionResult::default()
    };
    for (id, value) in runtime_values {
        match value {
            RuntimeValue::Int(value) => {
                result.values.insert(id, GraphRuntimeValue::Int(value));
            }
            RuntimeValue::Float(value) => {
                result.values.insert(id, GraphRuntimeValue::Float(value));
            }
            RuntimeValue::Bool(value) => {
                result.values.insert(id, GraphRuntimeValue::Bool(value));
            }
            RuntimeValue::Tensor(tensor) => {
                result.values.insert(id, GraphRuntimeValue::Tensor(tensor));
            }
            RuntimeValue::Linear(_) | RuntimeValue::Embedding(_) | RuntimeValue::Activation { .. } => {}
        }
    }
    Ok(result)
}

pub fn execute_graph_function(function: &GraphFunction, options: &GraphExecutorOptions) -> Result<GraphExecutionResult, String> {
    execute_execution_plan(&compile_execution_plan(function, BackendKind::Local), options)
}
