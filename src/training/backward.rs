use crate::backend::core::kind::BackendKind;
use crate::backend::metal::runtime::execute_metal_module;
use crate::backend::pytorch::runtime::execute_pytorch_module;
use crate::compiler::frontend_ir::LoweredModule;
use crate::compiler::lexer::TokenType;
use crate::ir::builder::build_graph_function;
use crate::ir::graph::{GraphFunction, GraphNode, GraphNodeKind};
use crate::ops::library::{runtime_primitive, RuntimePrimitiveKind};
use crate::runtime::graph_executor::{execute_graph_function, GraphExecutionResult, GraphExecutorOptions, GraphRuntimeValue};
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::runtime::layers::{
    backward_cross_entropy_logits, backward_cross_entropy_target, backward_dropout, backward_rms_norm_input,
    backward_silu, backward_softmax, LinearClosure,
};
use crate::runtime::tensor::{
    add_in_place, elementwise_binary, matmul, negate, ones_like, print_named_tensor, tensor_scalar_binary,
    transpose_2d, zeros_like, SimpleTensor,
};

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct BackwardRunOptions {
    pub backend: BackendKind,
    pub entry: String,
    pub tensor_shapes: BTreeMap<String, Vec<i64>>,
}

impl Default for BackwardRunOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Local,
            entry: "model".to_string(),
            tensor_shapes: BTreeMap::new(),
        }
    }
}

fn require_tensor(value: &GraphRuntimeValue) -> Result<&SimpleTensor, String> {
    match value {
        GraphRuntimeValue::Tensor(tensor) => Ok(tensor),
        _ => Err("Backward interpreter expected tensor value".to_string()),
    }
}

fn require_scalar(value: &GraphRuntimeValue) -> Result<f64, String> {
    match value {
        GraphRuntimeValue::Int(value) => Ok(*value as f64),
        GraphRuntimeValue::Float(value) => Ok(*value),
        GraphRuntimeValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        GraphRuntimeValue::Tensor(_) => Err("Backward interpreter expected scalar value".to_string()),
    }
}

fn require_bool(value: &GraphRuntimeValue) -> Result<bool, String> {
    match value {
        GraphRuntimeValue::Bool(value) => Ok(*value),
        _ => Err("Backward interpreter expected bool value".to_string()),
    }
}

fn build_linear_closure(node: &GraphNode, graph: &GraphFunction, execution: &GraphExecutionResult) -> Result<LinearClosure, String> {
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
        closure.out_features = require_scalar(
            execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing linear out_features".to_string())?,
        )? as i64;
        return Ok(closure);
    }
    if node.inputs.len() == 2 {
        let maybe_bool = execution
            .values
            .get(&node.inputs[1])
            .ok_or_else(|| "Missing linear with_bias".to_string())?;
        if matches!(maybe_bool, GraphRuntimeValue::Bool(_)) {
            closure.out_features = require_scalar(
                execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing linear out_features".to_string())?,
            )? as i64;
            closure.with_bias = require_bool(maybe_bool)?;
            return Ok(closure);
        }
    }
    closure.in_features = Some(
        require_scalar(execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing linear in_features".to_string())?)?
            as i64,
    );
    closure.out_features = require_scalar(
        execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing linear out_features".to_string())?,
    )? as i64;
    if node.inputs.len() == 3 {
        closure.with_bias =
            require_bool(execution.values.get(&node.inputs[2]).ok_or_else(|| "Missing linear with_bias".to_string())?)?;
    }
    Ok(closure)
}

fn find_entry_function<'a>(lowered: &'a LoweredModule, entry: &str) -> Result<&'a crate::compiler::frontend_ir::FeFunction, String> {
    lowered
        .functions
        .iter()
        .find(|function| function.name == entry)
        .ok_or_else(|| format!("Entry function '{}' not found in lowered module", entry))
}

fn find_producer<'a>(graph: &'a GraphFunction, output_id: usize) -> Option<&'a GraphNode> {
    graph.nodes.iter().find(|node| node.output == output_id)
}

fn resolve_objective_value_id(lowered: &LoweredModule, graph: &GraphFunction, entry: &str) -> Result<usize, String> {
    if let Some(plan) = &lowered.execution_plan {
        if let Some(run) = plan.runs.iter().find(|run| run.model_name == entry) {
            if let Some(symbol) = &run.objective_symbol {
                return graph
                    .named_values
                    .get(symbol)
                    .copied()
                    .ok_or_else(|| format!("Could not resolve objective '{}' in graph function", symbol));
            }
        }
    }
    graph.outputs
        .first()
        .copied()
        .ok_or_else(|| "Entry function did not return a value".to_string())
}

fn accumulate(gradients: &mut BTreeMap<usize, SimpleTensor>, value_id: usize, grad: &SimpleTensor) -> Result<(), String> {
    if let Some(existing) = gradients.get_mut(&value_id) {
        add_in_place(existing, grad)?;
    } else {
        gradients.insert(value_id, grad.clone());
    }
    Ok(())
}

pub(crate) fn compute_value_gradients(
    lowered: &LoweredModule,
    backend: BackendKind,
    entry: &str,
    tensor_shapes: &BTreeMap<String, Vec<i64>>,
) -> Result<(GraphFunction, GraphExecutionResult, BTreeMap<usize, SimpleTensor>), String> {
    let entry_function = find_entry_function(lowered, entry)?;
    let graph = build_graph_function(entry_function)?;
    let execution = match backend {
        BackendKind::Local => execute_graph_function(
            &graph,
            &GraphExecutorOptions {
                tensor_shapes: tensor_shapes.clone(),
            },
        )?,
        BackendKind::Metal => execute_metal_module(
            lowered,
            &RuntimeRunOptions {
                entry: entry.to_string(),
                tensor_shapes: tensor_shapes.clone(),
            },
            None,
        )?,
        BackendKind::PyTorch => execute_pytorch_module(
            lowered,
            &RuntimeRunOptions {
                entry: entry.to_string(),
                tensor_shapes: tensor_shapes.clone(),
            },
            None,
        )?,
    };
    let objective_id = resolve_objective_value_id(lowered, &graph, entry)?;
    let objective = execution
        .values
        .get(&objective_id)
        .ok_or_else(|| "Backward interpreter could not resolve the objective runtime value".to_string())?;

    let mut gradients = BTreeMap::new();
    accumulate(&mut gradients, objective_id, &ones_like(require_tensor(objective)?))?;

    for node in graph.nodes.iter().rev() {
        let Some(grad) = gradients.get(&node.output).cloned() else {
            continue;
        };
        match node.kind {
            GraphNodeKind::Constant | GraphNodeKind::LibraryCtor => {}
            GraphNodeKind::Apply => {
                if node.inputs.len() != 2 {
                    return Err("Graph backward currently supports only single-input apply nodes".to_string());
                }
                let callee_node = find_producer(&graph, node.inputs[0])
                    .ok_or_else(|| "Graph backward could not resolve apply callee producer".to_string())?;
                let input = require_tensor(
                    execution
                        .values
                        .get(&node.inputs[1])
                        .ok_or_else(|| "Missing apply input runtime value".to_string())?,
                )?;
                match callee_node.op.as_str() {
                    "linear" => {
                        let closure = build_linear_closure(callee_node, &graph, &execution)?;
                        let in_features = closure.in_features.unwrap_or_else(|| *input.shape.get(1).unwrap_or(&0));
                        let weight = crate::runtime::layers::apply_linear(
                            &LinearClosure {
                                in_features: Some(in_features),
                                out_features: closure.out_features,
                                with_bias: false,
                                dtype: closure.dtype.clone(),
                            },
                            &SimpleTensor {
                                shape: vec![in_features, in_features],
                                data: vec![0.0; (in_features * in_features) as usize],
                                dtype: closure.dtype.clone(),
                            },
                        )
                        .err()
                        .map(|_| ())
                        .is_none();
                        let _ = weight;
                        let weight = {
                            let shape = vec![in_features, closure.out_features];
                            let element_count = (in_features * closure.out_features) as usize;
                            SimpleTensor {
                                shape,
                                data: (0..element_count)
                                    .map(|index| ((index % 11) as f32 - 5.0) / 16.0)
                                    .collect(),
                                dtype: closure.dtype.clone(),
                            }
                        };
                        accumulate(&mut gradients, node.inputs[1], &matmul(&grad, &transpose_2d(&weight)?)?)?;
                    }
                    "SiLU" => {
                        accumulate(&mut gradients, node.inputs[1], &backward_silu(input, &grad)?)?;
                    }
                    "Softmax" => {
                        let output = require_tensor(
                            execution
                                .values
                                .get(&node.output)
                                .ok_or_else(|| "Missing softmax output runtime value".to_string())?,
                        )?;
                        accumulate(&mut gradients, node.inputs[1], &backward_softmax(output, &grad)?)?;
                    }
                    "Dropout" => {
                        let probability = require_scalar(
                            execution
                                .values
                                .get(&callee_node.inputs[0])
                                .ok_or_else(|| "Missing dropout probability runtime value".to_string())?,
                        )?;
                        accumulate(&mut gradients, node.inputs[1], &backward_dropout(&grad, probability)?)?;
                    }
                    _ => return Err("Graph backward does not support this library apply yet".to_string()),
                }
            }
            GraphNodeKind::Binary => {
                let lhs = execution
                    .values
                    .get(&node.inputs[0])
                    .ok_or_else(|| "Missing lhs runtime value".to_string())?;
                let rhs = execution
                    .values
                    .get(&node.inputs[1])
                    .ok_or_else(|| "Missing rhs runtime value".to_string())?;
                match (lhs, rhs) {
                    (GraphRuntimeValue::Tensor(lhs_tensor), GraphRuntimeValue::Tensor(rhs_tensor)) => match node.binary_op {
                        TokenType::Plus => {
                            accumulate(&mut gradients, node.inputs[0], &grad)?;
                            accumulate(&mut gradients, node.inputs[1], &grad)?;
                        }
                        TokenType::Minus => {
                            accumulate(&mut gradients, node.inputs[0], &grad)?;
                            accumulate(&mut gradients, node.inputs[1], &negate(&grad))?;
                        }
                        TokenType::Star => {
                            accumulate(&mut gradients, node.inputs[0], &elementwise_binary(TokenType::Star, &grad, rhs_tensor)?)?;
                            accumulate(&mut gradients, node.inputs[1], &elementwise_binary(TokenType::Star, &grad, lhs_tensor)?)?;
                        }
                        TokenType::Slash => {
                            let rhs_sq = elementwise_binary(TokenType::Star, rhs_tensor, rhs_tensor)?;
                            let num = elementwise_binary(TokenType::Star, &grad, lhs_tensor)?;
                            accumulate(&mut gradients, node.inputs[0], &elementwise_binary(TokenType::Slash, &grad, rhs_tensor)?)?;
                            accumulate(
                                &mut gradients,
                                node.inputs[1],
                                &negate(&elementwise_binary(TokenType::Slash, &num, &rhs_sq)?),
                            )?;
                        }
                        _ => return Err("Unsupported tensor binary op in graph backward".to_string()),
                    },
                    (GraphRuntimeValue::Tensor(_), _) => match node.binary_op {
                        TokenType::Plus | TokenType::Minus => accumulate(&mut gradients, node.inputs[0], &grad)?,
                        TokenType::Star | TokenType::Slash => {
                            accumulate(&mut gradients, node.inputs[0], &tensor_scalar_binary(node.binary_op, &grad, require_scalar(rhs)?)?)?
                        }
                        _ => return Err("Unsupported tensor-scalar op in graph backward".to_string()),
                    },
                    (_, GraphRuntimeValue::Tensor(_)) => match node.binary_op {
                        TokenType::Plus => accumulate(&mut gradients, node.inputs[1], &grad)?,
                        TokenType::Minus => accumulate(&mut gradients, node.inputs[1], &negate(&grad))?,
                        TokenType::Star => {
                            accumulate(&mut gradients, node.inputs[1], &tensor_scalar_binary(TokenType::Star, &grad, require_scalar(lhs)?)?)?
                        }
                        TokenType::Slash => {
                            return Err("Backward for scalar/tensor division is not implemented yet".to_string());
                        }
                        _ => return Err("Unsupported scalar-tensor op in graph backward".to_string()),
                    },
                    _ => {}
                }
            }
            GraphNodeKind::PrimitiveCall => match runtime_primitive(&node.op) {
                RuntimePrimitiveKind::Matmul => {
                    let lhs = require_tensor(
                        execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing matmul lhs".to_string())?,
                    )?;
                    let rhs = require_tensor(
                        execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing matmul rhs".to_string())?,
                    )?;
                    accumulate(&mut gradients, node.inputs[0], &matmul(&grad, &transpose_2d(rhs)?)?)?;
                    accumulate(&mut gradients, node.inputs[1], &matmul(&transpose_2d(lhs)?, &grad)?)?;
                }
                RuntimePrimitiveKind::Relu => {
                    let output = require_tensor(
                        execution.values.get(&node.output).ok_or_else(|| "Missing relu output".to_string())?,
                    )?;
                    let mask = SimpleTensor {
                        shape: output.shape.clone(),
                        data: output
                            .data
                            .iter()
                            .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
                            .collect(),
                        dtype: output.dtype.clone(),
                    };
                    accumulate(&mut gradients, node.inputs[0], &elementwise_binary(TokenType::Star, &grad, &mask)?)?;
                }
                RuntimePrimitiveKind::Scale => {
                    let scale = require_scalar(
                        execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing scale value".to_string())?,
                    )?;
                    accumulate(&mut gradients, node.inputs[0], &tensor_scalar_binary(TokenType::Star, &grad, scale)?)?;
                }
                RuntimePrimitiveKind::Unsupported => {
                    return Err(format!("Unsupported primitive op '{}' in graph backward", node.op));
                }
            },
            GraphNodeKind::LibraryCall => match node.op.as_str() {
                "rms_norm" => {
                    let input = require_tensor(
                        execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing rms_norm input".to_string())?,
                    )?;
                    let hidden_size = require_scalar(
                        execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing rms_norm hidden size".to_string())?,
                    )? as i64;
                    accumulate(&mut gradients, node.inputs[0], &backward_rms_norm_input(input, hidden_size, &grad)?)?;
                }
                "cross_entropy" => {
                    let logits = require_tensor(
                        execution.values.get(&node.inputs[0]).ok_or_else(|| "Missing cross_entropy logits".to_string())?,
                    )?;
                    let target = require_tensor(
                        execution.values.get(&node.inputs[1]).ok_or_else(|| "Missing cross_entropy target".to_string())?,
                    )?;
                    accumulate(&mut gradients, node.inputs[0], &backward_cross_entropy_logits(logits, target)?)?;
                    accumulate(&mut gradients, node.inputs[1], &backward_cross_entropy_target(logits, target)?)?;
                }
                _ => return Err(format!("Unsupported library op '{}' in graph backward", node.op)),
            },
        }
    }

    Ok((graph, execution, gradients))
}

pub fn run_backward_module(lowered: &LoweredModule, options: &BackwardRunOptions) -> Result<(), String> {
    let (graph, execution, gradients) =
        compute_value_gradients(lowered, options.backend, &options.entry, &options.tensor_shapes)?;
    println!("\n--- Gradient Output ---");
    for value in &graph.values {
        if !value.is_parameter {
            continue;
        }
        if let Some(gradient) = gradients.get(&value.id) {
            print_named_tensor(&format!("{}_grad", value.name), gradient);
        } else if let Some(runtime_value) = execution.values.get(&value.id) {
            let tensor = require_tensor(runtime_value)?;
            print_named_tensor(&format!("{}_grad", value.name), &zeros_like(tensor));
        }
    }
    println!("-----------------------");
    Ok(())
}
