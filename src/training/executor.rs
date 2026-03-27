//! Shared Rust training loop.
//!
//! Forward execution can come from the local, PyTorch, or Metal backend, but the
//! parameter-gradient accumulation and SGD update step stay in one Rust implementation.

use crate::backend::core::kind::BackendKind;
use crate::backend::cuda::runtime::execute_cuda_module;
use crate::backend::metal::runtime::execute_metal_module;
use crate::backend::pytorch::runtime::execute_pytorch_module;
use crate::compiler::frontend_ir::{FeValue, LoweredModule};
use crate::compiler::lexer::TokenType;
use crate::ir::builder::build_graph_function;
use crate::ir::graph::{GraphFunction, GraphNodeKind};
use crate::runtime::graph_executor::GraphRuntimeValue;
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::ops::library::{runtime_primitive, RuntimePrimitiveKind};
use crate::runtime::layers::{
    apply_causal_mask, apply_cross_entropy, apply_dropout, apply_embedding_with_parameters, apply_flatten_heads,
    apply_linear_with_parameters, apply_repeat_kv, apply_reshape, apply_rms_norm, apply_rope, backward_causal_mask_input,
    backward_cross_entropy_logits, backward_dropout, backward_embedding_weight, backward_flatten_heads_input,
    backward_linear_bias, backward_linear_input, backward_linear_weight, backward_repeat_kv_input,
    backward_rms_norm_input, backward_rope_input, backward_reshape_input, backward_silu, backward_softmax,
    make_embedding_weight, make_linear_bias, make_linear_weight, EmbeddingClosure, LinearClosure,
};
use crate::runtime::tensor::{
    add_in_place, apply_relu, apply_silu, apply_softmax, elementwise_binary, make_synthetic_tensor, negate, ones_like,
    scalar_tensor_binary, tensor_scalar_binary, SimpleTensor,
};

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRunOptions {
    pub backend: BackendKind,
    pub backend_overridden: bool,
    pub entry: String,
    pub tensor_shapes: BTreeMap<String, Vec<i64>>,
}

impl Default for TrainRunOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Local,
            backend_overridden: false,
            entry: "model".to_string(),
            tensor_shapes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RuntimeValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Tensor(SimpleTensor),
    Linear(LinearRuntimeSpec),
    Embedding(EmbeddingRuntimeSpec),
    Activation { op: String, probability: f64 },
}

#[derive(Debug, Clone, PartialEq)]
struct LinearRuntimeSpec {
    closure: LinearClosure,
    weight_name: String,
    bias_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct EmbeddingRuntimeSpec {
    closure: EmbeddingClosure,
    weight_name: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct TrainExecutionResult {
    values: BTreeMap<usize, RuntimeValue>,
}

fn require_tensor(value: &RuntimeValue) -> Result<&SimpleTensor, String> {
    match value {
        RuntimeValue::Tensor(tensor) => Ok(tensor),
        _ => Err("Train executor expected tensor value".to_string()),
    }
}

fn require_number(value: &RuntimeValue) -> Result<f64, String> {
    match value {
        RuntimeValue::Int(value) => Ok(*value as f64),
        RuntimeValue::Float(value) => Ok(*value),
        RuntimeValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err("Train executor expected scalar value".to_string()),
    }
}

fn require_int(value: &RuntimeValue) -> Result<i64, String> {
    match value {
        RuntimeValue::Int(value) => Ok(*value),
        _ => Err("Train executor expected integer value".to_string()),
    }
}

fn require_bool(value: &RuntimeValue) -> Result<bool, String> {
    match value {
        RuntimeValue::Bool(value) => Ok(*value),
        _ => Err("Train executor expected bool value".to_string()),
    }
}

fn tensor_sum(tensor: &SimpleTensor) -> f32 {
    tensor.data.iter().copied().sum()
}

fn make_tensor_argument(name: &str, dtype: &str, options: &TrainRunOptions) -> Result<SimpleTensor, String> {
    let shape = options
        .tensor_shapes
        .get(name)
        .ok_or_else(|| format!("Missing --shape for tensor parameter '{}'", name))?;
    Ok(make_synthetic_tensor(shape, dtype.to_string()))
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

fn find_train_run<'a>(lowered: &'a LoweredModule, entry: &str) -> Result<&'a crate::compiler::frontend_ir::FeExecutionRun, String> {
    lowered
        .execution_plan
        .as_ref()
        .and_then(|plan| plan.runs.iter().find(|run| run.model_name == entry))
        .ok_or_else(|| "Train executor requires a resolved 'train model' execution run".to_string())
}

fn linear_weight_name(output_id: usize) -> String {
    format!("linear_{output_id}_weight")
}

fn linear_bias_name(output_id: usize) -> String {
    format!("linear_{output_id}_bias")
}

fn embedding_weight_name(output_id: usize) -> String {
    format!("embedding_{output_id}_weight")
}

fn build_linear_spec(
    graph: &GraphFunction,
    node: &crate::ir::graph::GraphNode,
    values: &BTreeMap<usize, RuntimeValue>,
) -> Result<LinearRuntimeSpec, String> {
    let output = &graph.values[node.output];
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
        closure.out_features = require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing linear out_features".to_string())?)?;
    } else if node.inputs.len() == 2 {
        if matches!(values.get(&node.inputs[1]), Some(RuntimeValue::Bool(_))) {
            closure.out_features =
                require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing linear out_features".to_string())?)?;
            closure.with_bias =
                require_bool(values.get(&node.inputs[1]).ok_or_else(|| "Missing linear with_bias".to_string())?)?;
        } else {
            closure.in_features =
                Some(require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing linear in_features".to_string())?)?);
            closure.out_features =
                require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing linear out_features".to_string())?)?;
        }
    } else {
        closure.in_features =
            Some(require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing linear in_features".to_string())?)?);
        closure.out_features =
            require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing linear out_features".to_string())?)?;
        closure.with_bias =
            require_bool(values.get(&node.inputs[2]).ok_or_else(|| "Missing linear with_bias".to_string())?)?;
    }

    let with_bias = closure.with_bias;
    Ok(LinearRuntimeSpec {
        closure,
        weight_name: linear_weight_name(node.output),
        bias_name: if with_bias {
            Some(linear_bias_name(node.output))
        } else {
            None
        },
    })
}

fn build_embedding_spec(
    graph: &GraphFunction,
    node: &crate::ir::graph::GraphNode,
    values: &BTreeMap<usize, RuntimeValue>,
) -> Result<EmbeddingRuntimeSpec, String> {
    let output = &graph.values[node.output];
    Ok(EmbeddingRuntimeSpec {
        closure: EmbeddingClosure {
            num_embeddings: require_int(values.get(&node.inputs[0]).ok_or_else(|| "Missing num_embeddings".to_string())?)?,
            embedding_dim: require_int(values.get(&node.inputs[1]).ok_or_else(|| "Missing embedding_dim".to_string())?)?,
            dtype: output
                .ty
                .callable_return
                .as_deref()
                .and_then(|ty| ty.tensor_dtype.clone())
                .unwrap_or_else(|| "float32".to_string()),
        },
        weight_name: embedding_weight_name(node.output),
    })
}

fn ensure_linear_parameters(
    spec: &LinearRuntimeSpec,
    input: &SimpleTensor,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) {
    let in_features = spec.closure.in_features.unwrap_or(input.shape[1]);
    parameters
        .entry(spec.weight_name.clone())
        .or_insert_with(|| make_linear_weight(in_features, spec.closure.out_features, &spec.closure.dtype));
    if let Some(bias_name) = &spec.bias_name {
        parameters
            .entry(bias_name.clone())
            .or_insert_with(|| make_linear_bias(spec.closure.out_features, &spec.closure.dtype));
    }
}

fn ensure_embedding_parameters(
    spec: &EmbeddingRuntimeSpec,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) {
    parameters.entry(spec.weight_name.clone()).or_insert_with(|| {
        make_embedding_weight(
            spec.closure.num_embeddings,
            spec.closure.embedding_dim,
            &spec.closure.dtype,
        )
    });
}

fn execute_train_graph(
    graph: &GraphFunction,
    options: &TrainRunOptions,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) -> Result<TrainExecutionResult, String> {
    let mut result = TrainExecutionResult::default();
    for value in &graph.values {
        if value.is_parameter {
            result.values.insert(
                value.id,
                RuntimeValue::Tensor(make_tensor_argument(
                    &value.name,
                    value.ty.tensor_dtype.as_deref().unwrap_or("float32"),
                    options,
                )?),
            );
        }
    }

    for node in &graph.nodes {
        let value = match node.kind {
            GraphNodeKind::Constant => match &node.constant {
                FeValue::Int(value) => RuntimeValue::Int(*value),
                FeValue::Float(value) => RuntimeValue::Float(*value),
                FeValue::Bool(value) => RuntimeValue::Bool(*value),
                _ => return Err("Unsupported train graph constant".to_string()),
            },
            GraphNodeKind::Binary => {
                let lhs = result.values.get(&node.inputs[0]).ok_or_else(|| "Missing binary lhs".to_string())?;
                let rhs = result.values.get(&node.inputs[1]).ok_or_else(|| "Missing binary rhs".to_string())?;
                match (lhs, rhs) {
                    (RuntimeValue::Tensor(lhs), RuntimeValue::Tensor(rhs)) => {
                        RuntimeValue::Tensor(elementwise_binary(node.binary_op, lhs, rhs)?)
                    }
                    (RuntimeValue::Tensor(lhs), _) => {
                        RuntimeValue::Tensor(tensor_scalar_binary(node.binary_op, lhs, require_number(rhs)?)?)
                    }
                    (_, RuntimeValue::Tensor(rhs)) => {
                        RuntimeValue::Tensor(scalar_tensor_binary(node.binary_op, require_number(lhs)?, rhs)?)
                    }
                    _ => {
                        let left = require_number(lhs)?;
                        let right = require_number(rhs)?;
                        match node.binary_op {
                            TokenType::Plus => RuntimeValue::Float(left + right),
                            TokenType::Minus => RuntimeValue::Float(left - right),
                            TokenType::Star => RuntimeValue::Float(left * right),
                            TokenType::Slash => RuntimeValue::Float(left / right),
                            _ => return Err("Unsupported scalar train binary op".to_string()),
                        }
                    }
                }
            }
            GraphNodeKind::PrimitiveCall => match runtime_primitive(&node.op) {
                RuntimePrimitiveKind::Matmul => RuntimeValue::Tensor(crate::runtime::tensor::matmul(
                    require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing matmul lhs".to_string())?)?,
                    require_tensor(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing matmul rhs".to_string())?)?,
                )?),
                RuntimePrimitiveKind::Relu => RuntimeValue::Tensor(apply_relu(
                    require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing relu input".to_string())?)?,
                )),
                RuntimePrimitiveKind::Scale => RuntimeValue::Tensor(tensor_scalar_binary(
                    TokenType::Star,
                    require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing scale input".to_string())?)?,
                    require_number(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing scale factor".to_string())?)?,
                )?),
                RuntimePrimitiveKind::Unsupported => {
                    return Err(format!("Unsupported primitive train op '{}'", node.op));
                }
            },
            GraphNodeKind::LibraryCall => {
                if node.op == "rms_norm" {
                    RuntimeValue::Tensor(apply_rms_norm(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing rms_norm input".to_string())?)?,
                        require_int(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing hidden size".to_string())?)?,
                    )?)
                } else if node.op == "cross_entropy" {
                    RuntimeValue::Tensor(apply_cross_entropy(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing logits".to_string())?)?,
                        require_tensor(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing target".to_string())?)?,
                    )?)
                } else if node.op == "reshape" {
                    let shape = node.inputs[1..]
                        .iter()
                        .map(|id| require_int(result.values.get(id).ok_or_else(|| "Missing reshape dim".to_string())?))
                        .collect::<Result<Vec<_>, _>>()?;
                    RuntimeValue::Tensor(apply_reshape(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing reshape input".to_string())?)?,
                        &shape,
                    )?)
                } else if node.op == "repeat_kv" {
                    RuntimeValue::Tensor(apply_repeat_kv(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing repeat_kv input".to_string())?)?,
                        require_int(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing repeats".to_string())?)?,
                    )?)
                } else if node.op == "flatten_heads" {
                    RuntimeValue::Tensor(apply_flatten_heads(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing flatten_heads input".to_string())?)?,
                    )?)
                } else if node.op == "causal_mask" {
                    RuntimeValue::Tensor(apply_causal_mask(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing causal_mask input".to_string())?)?,
                    )?)
                } else if node.op == "rope" {
                    RuntimeValue::Tensor(apply_rope(
                        require_tensor(result.values.get(&node.inputs[0]).ok_or_else(|| "Missing rope input".to_string())?)?,
                        require_int(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing rope head_dim".to_string())?)?,
                        require_number(result.values.get(&node.inputs[2]).ok_or_else(|| "Missing rope theta".to_string())?)?,
                    )?)
                } else {
                    return Err(format!("Unsupported library train op '{}'", node.op));
                }
            }
            GraphNodeKind::LibraryCtor => {
                if node.op == "linear" {
                    RuntimeValue::Linear(build_linear_spec(graph, node, &result.values)?)
                } else if node.op == "Embedding" {
                    RuntimeValue::Embedding(build_embedding_spec(graph, node, &result.values)?)
                } else if node.op == "SiLU" || node.op == "Softmax" {
                    RuntimeValue::Activation {
                        op: node.op.clone(),
                        probability: 0.0,
                    }
                } else if node.op == "Dropout" {
                    RuntimeValue::Activation {
                        op: "Dropout".to_string(),
                        probability: require_number(
                            result.values.get(&node.inputs[0]).ok_or_else(|| "Missing dropout probability".to_string())?,
                        )?,
                    }
                } else {
                    return Err(format!("Unsupported library constructor '{}' in train graph", node.op));
                }
            }
            GraphNodeKind::Apply => {
                let callee = result.values.get(&node.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
                let input = require_tensor(result.values.get(&node.inputs[1]).ok_or_else(|| "Missing apply input".to_string())?)?;
                match callee {
                    RuntimeValue::Linear(spec) => {
                        ensure_linear_parameters(spec, input, parameters);
                        let weight = parameters
                            .get(&spec.weight_name)
                            .ok_or_else(|| format!("Missing linear weight parameter '{}'", spec.weight_name))?;
                        let bias = spec
                            .bias_name
                            .as_ref()
                            .and_then(|name| parameters.get(name));
                        RuntimeValue::Tensor(apply_linear_with_parameters(input, weight, bias)?)
                    }
                    RuntimeValue::Embedding(spec) => {
                        ensure_embedding_parameters(spec, parameters);
                        let weight = parameters
                            .get(&spec.weight_name)
                            .ok_or_else(|| format!("Missing embedding weight parameter '{}'", spec.weight_name))?;
                        RuntimeValue::Tensor(apply_embedding_with_parameters(
                            input,
                            weight,
                            spec.closure.num_embeddings,
                            spec.closure.embedding_dim,
                        )?)
                    }
                    RuntimeValue::Activation { op, probability } => match op.as_str() {
                        "SiLU" => RuntimeValue::Tensor(apply_silu(input)),
                        "Softmax" => RuntimeValue::Tensor(apply_softmax(input)?),
                        "Dropout" => RuntimeValue::Tensor(apply_dropout(input, *probability)?),
                        _ => return Err("Unsupported activation in train graph".to_string()),
                    },
                    _ => return Err("Unsupported apply in train graph".to_string()),
                }
            }
        };
        result.values.insert(node.output, value);
    }

    Ok(result)
}

fn convert_graph_runtime_value(value: &GraphRuntimeValue) -> RuntimeValue {
    match value {
        GraphRuntimeValue::Int(value) => RuntimeValue::Int(*value),
        GraphRuntimeValue::Float(value) => RuntimeValue::Float(*value),
        GraphRuntimeValue::Bool(value) => RuntimeValue::Bool(*value),
        GraphRuntimeValue::Tensor(tensor) => RuntimeValue::Tensor(tensor.clone()),
    }
}

fn materialize_ctor_values(
    graph: &GraphFunction,
    values: &mut BTreeMap<usize, RuntimeValue>,
) -> Result<(), String> {
    for node in &graph.nodes {
        if node.kind != GraphNodeKind::LibraryCtor {
            continue;
        }
        let value = if node.op == "linear" {
            RuntimeValue::Linear(build_linear_spec(graph, node, values)?)
        } else if node.op == "Embedding" {
            RuntimeValue::Embedding(build_embedding_spec(graph, node, values)?)
        } else if node.op == "SiLU" || node.op == "Softmax" {
            RuntimeValue::Activation {
                op: node.op.clone(),
                probability: 0.0,
            }
        } else if node.op == "Dropout" {
            RuntimeValue::Activation {
                op: "Dropout".to_string(),
                probability: require_number(
                    values
                        .get(&node.inputs[0])
                        .ok_or_else(|| "Missing dropout probability".to_string())?,
                )?,
            }
        } else {
            return Err(format!("Unsupported library constructor '{}' in train graph", node.op));
        };
        values.insert(node.output, value);
    }
    Ok(())
}

fn initialize_train_parameters(
    graph: &GraphFunction,
    options: &TrainRunOptions,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) -> Result<(), String> {
    if !parameters.is_empty() {
        return Ok(());
    }
    // A dry forward pass is enough to discover callable specs and lazily materialize
    // the parameter tensors expected by the current graph.
    let _ = execute_train_graph(graph, options, parameters)?;
    Ok(())
}

fn execute_train_graph_pytorch(
    lowered: &LoweredModule,
    graph: &GraphFunction,
    options: &TrainRunOptions,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) -> Result<TrainExecutionResult, String> {
    initialize_train_parameters(graph, options, parameters)?;
    let runtime_options = RuntimeRunOptions {
        entry: options.entry.clone(),
        tensor_shapes: options.tensor_shapes.clone(),
    };
    let execution = execute_pytorch_module(lowered, &runtime_options, Some(parameters))?;
    let mut values = execution
        .values
        .iter()
        .map(|(id, value)| (*id, convert_graph_runtime_value(value)))
        .collect::<BTreeMap<_, _>>();
    materialize_ctor_values(graph, &mut values)?;
    Ok(TrainExecutionResult { values })
}

fn execute_train_graph_metal(
    lowered: &LoweredModule,
    graph: &GraphFunction,
    options: &TrainRunOptions,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) -> Result<TrainExecutionResult, String> {
    initialize_train_parameters(graph, options, parameters)?;
    let runtime_options = RuntimeRunOptions {
        entry: options.entry.clone(),
        tensor_shapes: options.tensor_shapes.clone(),
    };
    let execution = execute_metal_module(lowered, &runtime_options, Some(parameters))?;
    let mut values = execution
        .values
        .iter()
        .map(|(id, value)| (*id, convert_graph_runtime_value(value)))
        .collect::<BTreeMap<_, _>>();
    materialize_ctor_values(graph, &mut values)?;
    Ok(TrainExecutionResult { values })
}

fn execute_train_graph_cuda(
    lowered: &LoweredModule,
    graph: &GraphFunction,
    options: &TrainRunOptions,
    parameters: &mut BTreeMap<String, SimpleTensor>,
) -> Result<TrainExecutionResult, String> {
    initialize_train_parameters(graph, options, parameters)?;
    let runtime_options = RuntimeRunOptions {
        entry: options.entry.clone(),
        tensor_shapes: options.tensor_shapes.clone(),
    };
    let execution = execute_cuda_module(lowered, &runtime_options, Some(parameters))?;
    let mut values = execution
        .values
        .iter()
        .map(|(id, value)| (*id, convert_graph_runtime_value(value)))
        .collect::<BTreeMap<_, _>>();
    materialize_ctor_values(graph, &mut values)?;
    Ok(TrainExecutionResult { values })
}

fn resolve_objective_value_id(
    run: &crate::compiler::frontend_ir::FeExecutionRun,
    graph: &GraphFunction,
) -> Result<usize, String> {
    if let Some(symbol) = &run.objective_symbol {
        return graph
            .named_values
            .get(symbol)
            .copied()
            .ok_or_else(|| format!("Could not resolve objective '{}' in graph", symbol));
    }
    graph.outputs
        .first()
        .copied()
        .ok_or_else(|| "Graph does not have a return value".to_string())
}

fn accumulate_value(gradients: &mut BTreeMap<usize, SimpleTensor>, id: usize, grad: &SimpleTensor) -> Result<(), String> {
    if let Some(existing) = gradients.get_mut(&id) {
        add_in_place(existing, grad)?;
    } else {
        gradients.insert(id, grad.clone());
    }
    Ok(())
}

fn accumulate_parameter(gradients: &mut BTreeMap<String, SimpleTensor>, name: &str, grad: &SimpleTensor) -> Result<(), String> {
    if let Some(existing) = gradients.get_mut(name) {
        add_in_place(existing, grad)?;
    } else {
        gradients.insert(name.to_string(), grad.clone());
    }
    Ok(())
}

fn compute_parameter_gradients(
    graph: &GraphFunction,
    execution: &TrainExecutionResult,
    objective_id: usize,
    parameters: &BTreeMap<String, SimpleTensor>,
) -> Result<BTreeMap<String, SimpleTensor>, String> {
    // Gradients are keyed by parameter name rather than value id because constructors
    // like `linear(...)` materialize trainable tensors lazily.
    let objective = require_tensor(
        execution
            .values
            .get(&objective_id)
            .ok_or_else(|| "Train executor could not resolve objective runtime value".to_string())?,
    )?;
    let mut value_grads = BTreeMap::new();
    let mut param_grads = BTreeMap::new();
    value_grads.insert(objective_id, ones_like(objective));

    for node in graph.nodes.iter().rev() {
        let Some(grad) = value_grads.get(&node.output).cloned() else {
            continue;
        };
        match node.kind {
            GraphNodeKind::Constant | GraphNodeKind::LibraryCtor => {}
            GraphNodeKind::Binary => {
                let lhs = execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing lhs".to_string())?;
                let rhs = execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing rhs".to_string())?;
                match (lhs, rhs) {
                    (RuntimeValue::Tensor(lhs), RuntimeValue::Tensor(rhs)) => match node.binary_op {
                        TokenType::Plus => {
                            accumulate_value(&mut value_grads, node.inputs[0], &grad)?;
                            accumulate_value(&mut value_grads, node.inputs[1], &grad)?;
                        }
                        TokenType::Minus => {
                            accumulate_value(&mut value_grads, node.inputs[0], &grad)?;
                            accumulate_value(&mut value_grads, node.inputs[1], &negate(&grad))?;
                        }
                        TokenType::Star => {
                            accumulate_value(&mut value_grads, node.inputs[0], &elementwise_binary(TokenType::Star, &grad, rhs)?)?;
                            accumulate_value(&mut value_grads, node.inputs[1], &elementwise_binary(TokenType::Star, &grad, lhs)?)?;
                        }
                        TokenType::Slash => {
                            accumulate_value(&mut value_grads, node.inputs[0], &elementwise_binary(TokenType::Slash, &grad, rhs)?)?;
                        }
                        _ => return Err("Unsupported binary op in train backward".to_string()),
                    },
                    (RuntimeValue::Tensor(_), _) => match node.binary_op {
                        TokenType::Plus | TokenType::Minus => {
                            accumulate_value(&mut value_grads, node.inputs[0], &grad)?;
                        }
                        TokenType::Star | TokenType::Slash => {
                            accumulate_value(
                                &mut value_grads,
                                node.inputs[0],
                                &tensor_scalar_binary(node.binary_op, &grad, require_number(rhs)?)?,
                            )?;
                        }
                        _ => return Err("Unsupported tensor-scalar op in train backward".to_string()),
                    },
                    (_, RuntimeValue::Tensor(_)) => match node.binary_op {
                        TokenType::Plus => accumulate_value(&mut value_grads, node.inputs[1], &grad)?,
                        TokenType::Minus => accumulate_value(&mut value_grads, node.inputs[1], &negate(&grad))?,
                        TokenType::Star => {
                            accumulate_value(
                                &mut value_grads,
                                node.inputs[1],
                                &tensor_scalar_binary(TokenType::Star, &grad, require_number(lhs)?)?,
                            )?;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            GraphNodeKind::PrimitiveCall => match runtime_primitive(&node.op) {
                RuntimePrimitiveKind::Matmul => {
                    let lhs = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing lhs".to_string())?)?;
                    let rhs = require_tensor(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing rhs".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_linear_input(&grad, rhs)?)?;
                    accumulate_value(&mut value_grads, node.inputs[1], &backward_linear_weight(&grad, lhs)?)?;
                }
                RuntimePrimitiveKind::Relu => {
                    let output = require_tensor(execution.values.get(&node.output).ok_or_else(|| "Missing relu output".to_string())?)?;
                    let mask = SimpleTensor {
                        shape: output.shape.clone(),
                        data: output.data.iter().map(|value| if *value > 0.0 { 1.0 } else { 0.0 }).collect(),
                        dtype: output.dtype.clone(),
                    };
                    accumulate_value(&mut value_grads, node.inputs[0], &elementwise_binary(TokenType::Star, &grad, &mask)?)?;
                }
                RuntimePrimitiveKind::Scale => {
                    let scale = require_number(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing scale".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &tensor_scalar_binary(TokenType::Star, &grad, scale)?)?;
                }
                RuntimePrimitiveKind::Unsupported => {
                    return Err("Unsupported primitive op in train backward".to_string());
                }
            },
            GraphNodeKind::LibraryCall => {
                if node.op == "rms_norm" {
                    let input = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing rms_norm input".to_string())?)?;
                    let hidden_size = require_int(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing hidden size".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_rms_norm_input(input, hidden_size, &grad)?)?;
                } else if node.op == "cross_entropy" {
                    let logits = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing logits".to_string())?)?;
                    let target = require_tensor(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing target".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_cross_entropy_logits(logits, target)?)?;
                } else if node.op == "reshape" {
                    let input = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing reshape input".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_reshape_input(&grad, input)?)?;
                } else if node.op == "flatten_heads" {
                    let input = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing flatten_heads input".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_flatten_heads_input(&grad, input)?)?;
                } else if node.op == "repeat_kv" {
                    let input = require_tensor(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing repeat_kv input".to_string())?)?;
                    let repeats = require_int(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing repeats".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_repeat_kv_input(&grad, input, repeats)?)?;
                } else if node.op == "causal_mask" {
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_causal_mask_input(&grad)?)?;
                } else if node.op == "rope" {
                    let head_dim = require_int(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing rope head_dim".to_string())?)?;
                    let theta = require_number(execution.values.get(&node.inputs[2]).ok_or_else(|| "Missing rope theta".to_string())?)?;
                    accumulate_value(&mut value_grads, node.inputs[0], &backward_rope_input(&grad, head_dim, theta)?)?;
                }
            }
            GraphNodeKind::Apply => {
                let callee = execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing apply callee".to_string())?;
                match callee {
                    RuntimeValue::Linear(spec) => {
                        let input = require_tensor(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing linear input".to_string())?)?;
                        let weight = parameters
                            .get(&spec.weight_name)
                            .cloned()
                            .unwrap_or_else(|| {
                                make_linear_weight(
                                    spec.closure.in_features.unwrap_or(input.shape[1]),
                                    spec.closure.out_features,
                                    &spec.closure.dtype,
                                )
                            });
                        accumulate_value(&mut value_grads, node.inputs[1], &backward_linear_input(&grad, &weight)?)?;
                        accumulate_parameter(&mut param_grads, &spec.weight_name, &backward_linear_weight(&grad, input)?)?;
                        if let Some(bias_name) = &spec.bias_name {
                            accumulate_parameter(&mut param_grads, bias_name, &backward_linear_bias(&grad)?)?;
                        }
                    }
                    RuntimeValue::Embedding(spec) => {
                        let input = require_tensor(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing embedding input".to_string())?)?;
                        accumulate_parameter(
                            &mut param_grads,
                            &spec.weight_name,
                            &backward_embedding_weight(
                                &grad,
                                input,
                                spec.closure.num_embeddings,
                                spec.closure.embedding_dim,
                            )?,
                        )?;
                    }
                    RuntimeValue::Activation { op, probability } => {
                        let input = require_tensor(execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing activation input".to_string())?)?;
                        match op.as_str() {
                            "SiLU" => accumulate_value(&mut value_grads, node.inputs[1], &backward_silu(input, &grad)?)?,
                            "Softmax" => {
                                let output = require_tensor(execution.values.get(&node.output).ok_or_else(|| "Missing softmax output".to_string())?)?;
                                accumulate_value(&mut value_grads, node.inputs[1], &backward_softmax(output, &grad)?)?;
                            }
                            "Dropout" => {
                                accumulate_value(&mut value_grads, node.inputs[1], &backward_dropout(&grad, *probability)?)?
                            }
                            _ => return Err("Unsupported activation in train backward".to_string()),
                        }
                    }
                    _ => return Err("Unsupported apply in train backward".to_string()),
                }
            }
        }
    }

    Ok(param_grads)
}

fn apply_sgd(
    parameters: &mut BTreeMap<String, SimpleTensor>,
    gradients: &BTreeMap<String, SimpleTensor>,
    learning_rate: f64,
) -> Result<(), String> {
    for (name, grad) in gradients {
        let Some(parameter) = parameters.get_mut(name) else {
            continue;
        };
        if parameter.shape != grad.shape {
            return Err(format!("Parameter gradient shape mismatch for '{}'", name));
        }
        for (parameter_value, grad_value) in parameter.data.iter_mut().zip(&grad.data) {
            *parameter_value -= learning_rate as f32 * grad_value;
        }
    }
    Ok(())
}

pub fn run_train_module(lowered: &LoweredModule, options: &TrainRunOptions) -> Result<(), String> {
    if options.backend != BackendKind::Local
        && options.backend != BackendKind::Cuda
        && options.backend != BackendKind::PyTorch
        && options.backend != BackendKind::Metal
    {
        return Err("Rust train executor currently supports only the local, CUDA, Metal, and PyTorch backends".to_string());
    }

    let entry = find_entry_function(lowered, &options.entry)?;
    let run = find_train_run(lowered, &options.entry)?;
    let graph = build_graph_function(entry)?;
    let optimizer = match &run.optimizer {
        Some(FeValue::String(value)) => value.as_str(),
        _ => "sgd",
    };
    if optimizer != "sgd" {
        return Err("Train executor currently supports only optimizer='sgd'".to_string());
    }
    let learning_rate = match &run.learning_rate {
        Some(FeValue::Float(value)) => *value,
        Some(FeValue::Int(value)) => *value as f64,
        _ => 0.01,
    };
    let iterations = match &run.iteration {
        Some(FeValue::Int(value)) => *value,
        _ => 1,
    };
    let objective_id = resolve_objective_value_id(run, &graph)?;
    let mut parameters = BTreeMap::new();

    println!("\n--- Training Output ---");
    println!("backend={}", options.backend.as_str());
    for step in 0..iterations {
        // Only the forward pass varies by backend. Gradient accumulation and SGD stay shared.
        let execution = match options.backend {
            BackendKind::Local => execute_train_graph(&graph, options, &mut parameters)?,
            BackendKind::Cuda => execute_train_graph_cuda(lowered, &graph, options, &mut parameters)?,
            BackendKind::PyTorch => execute_train_graph_pytorch(lowered, &graph, options, &mut parameters)?,
            BackendKind::Metal => execute_train_graph_metal(lowered, &graph, options, &mut parameters)?,
        };
        let objective = require_tensor(
            execution
                .values
                .get(&objective_id)
                .ok_or_else(|| "Train executor could not resolve objective runtime value".to_string())?,
        )?;
        let gradients = compute_parameter_gradients(&graph, &execution, objective_id, &parameters)?;
        let loss = tensor_sum(objective);
        apply_sgd(&mut parameters, &gradients, learning_rate)?;
        println!("step={} loss={:.6} params={}", step + 1, loss, parameters.len());
    }
    println!("-----------------------");
    Ok(())
}
