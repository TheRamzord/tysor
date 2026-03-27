//! PyTorch runtime/export bridge.
//!
//! The current Rust path does not embed libtorch directly. Instead it generates a
//! small Python program from the lowered graph, executes that with `python3`, and
//! parses the resulting tensors back into Rust.

use crate::compiler::frontend_ir::LoweredModule;
use crate::ir::builder::build_graph_function;
use crate::ir::graph::GraphNodeKind;
use crate::runtime::graph_executor::{GraphExecutionResult, GraphRuntimeValue};
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::runtime::tensor::{print_tensor, SimpleTensor};
use crate::training::executor::{run_train_module, TrainRunOptions};

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn python_tuple(values: &[i64]) -> String {
    let mut out = String::from("(");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    if values.len() == 1 {
        out.push(',');
    }
    out.push(')');
    out
}

fn python_quoted(text: &str) -> String {
    let mut out = String::from("'");
    for ch in text.chars() {
        if ch == '\\' || ch == '\'' {
            out.push('\\');
        }
        out.push(ch);
    }
    out.push('\'');
    out
}

fn python_tensor_literal(tensor: &SimpleTensor) -> String {
    let values = tensor
        .data
        .iter()
        .map(|value| format!("{:.9}", *value as f64))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "torch.tensor([{}], dtype=torch.float32, device=_tysor_device).reshape{}",
        values,
        python_tuple(&tensor.shape)
    )
}

fn value_var(value_id: usize) -> String {
    format!("v{}", value_id)
}

fn linear_weight_name(output_id: usize) -> String {
    format!("linear_{output_id}_weight")
}

fn linear_bias_name(output_id: usize) -> String {
    format!("linear_{output_id}_bias")
}

fn build_input_bindings(
    entry: &crate::compiler::frontend_ir::FeFunction,
    shapes: &BTreeMap<String, Vec<i64>>,
) -> Result<String, String> {
    let mut out = String::new();
    for (name, _ty) in &entry.params {
        let shape = shapes
            .get(name)
            .ok_or_else(|| format!("Missing --shape for tensor parameter '{}'", name))?;
        out.push_str(&format!(
            "{} = _tysor_make_input({}, torch.float32)\n",
            name,
            python_tuple(shape)
        ));
    }
    Ok(out)
}

fn build_graph_execution_script(
    lowered: &LoweredModule,
    entry: &crate::compiler::frontend_ir::FeFunction,
    options: &RuntimeRunOptions,
    parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<String, String> {
    let graph = build_graph_function(entry)?;
    let mut out = String::new();
    let _ = lowered;
    // Keep the generated script self-contained so the Rust side only needs to pass
    // shapes and optional parameter tensors.
    out.push_str("import torch\n\n");
    out.push_str(
        "def tysor_pattern_tensor(shape, mod, scale, dtype=torch.float32, device=None):\n    count = 1\n    for dim in shape:\n        count *= int(dim)\n    values = (torch.arange(count, dtype=torch.float32, device=device) % mod + 1.0) / scale\n    return values.reshape(tuple(shape)).to(dtype)\n\n",
    );
    out.push_str(
        "def tysor_init_linear_weight(in_features, out_features, dtype=torch.float32, device=None):\n    return tysor_pattern_tensor((in_features, out_features), 13.0, 16.0, dtype=dtype, device=device)\n\n",
    );
    out.push_str(
        "def tysor_init_linear_bias(out_features, dtype=torch.float32, device=None):\n    return tysor_pattern_tensor((out_features,), 7.0, 32.0, dtype=dtype, device=device)\n\n",
    );
    out.push_str(
        "def tysor_init_embedding_weight(num_embeddings, embedding_dim, dtype=torch.float32, device=None):\n    return tysor_pattern_tensor((num_embeddings, embedding_dim), 17.0, 8.0, dtype=dtype, device=device)\n\n",
    );
    out.push_str(
        "def tysor_linear(x, weight, bias=None):\n    y = torch.matmul(x, weight)\n    if bias is not None:\n        y = y + bias\n    return y\n\n",
    );
    out.push_str("def tysor_embedding(idx, weight):\n    return weight[idx.to(torch.long)]\n\n");
    out.push_str("def tysor_silu(x):\n    return x / (1.0 + torch.exp(-x))\n\n");
    out.push_str(
        "def tysor_softmax(x):\n    max_value = torch.amax(x, dim=-1, keepdim=True)\n    exp_value = torch.exp(x - max_value)\n    return exp_value / torch.sum(exp_value, dim=-1, keepdim=True)\n\n",
    );
    out.push_str("def tysor_dropout(x, probability):\n    return x * (1.0 - probability)\n\n");
    out.push_str(
        "def tysor_rms_norm(x, hidden_size, eps=1.0e-5):\n    if x.shape[-1] != hidden_size:\n        raise ValueError('rms_norm hidden size mismatch')\n    mean_sq = torch.mean(x * x, dim=-1, keepdim=True)\n    return x * torch.rsqrt(mean_sq + eps)\n\n",
    );
    out.push_str(
        "def tysor_cross_entropy(logits, target, eps=1.0e-6):\n    probs = tysor_softmax(logits)\n    return -(target * torch.log(torch.clamp(probs, min=eps))).sum(dim=-1, keepdim=True)\n\n",
    );
    out.push_str(
        "def tysor_rope(x, head_dim, theta=10000.0):\n    if x.shape[-1] != head_dim:\n        raise ValueError('rope head_dim mismatch')\n    if head_dim % 2 != 0:\n        raise ValueError('rope requires an even head_dim')\n    orig_dtype = x.dtype\n    half = head_dim // 2\n    seq_len = x.shape[-2] if x.dim() >= 2 else x.shape[0]\n    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)\n    inv_freq = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)\n    angles = positions[:, None] * inv_freq[None, :]\n    cos = torch.cos(angles)\n    sin = torch.sin(angles)\n    while cos.dim() < x.dim() - 1:\n        cos = cos.unsqueeze(0)\n        sin = sin.unsqueeze(0)\n    x1 = x[..., :half].to(torch.float32)\n    x2 = x[..., half:].to(torch.float32)\n    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)\n    return rotated.to(orig_dtype)\n\n",
    );
    out.push_str(
        "def tysor_reshape(x, *shape):\n    if not shape:\n        raise ValueError('reshape expects at least one dimension')\n    return x.reshape(tuple(int(dim) for dim in shape))\n\n",
    );
    out.push_str(
        "def tysor_causal_mask(x):\n    if x.dim() < 2:\n        raise ValueError('causal_mask expects rank >= 2')\n    q = x.shape[-2]\n    k = x.shape[-1]\n    mask = torch.triu(torch.ones((q, k), device=x.device, dtype=torch.bool), diagonal=1)\n    fill = -1.0e4 if x.is_floating_point() else -10000\n    return x.masked_fill(mask, fill)\n\n",
    );
    out.push_str(
        "def tysor_flatten_heads(x):\n    if x.dim() < 3:\n        return x\n    shape = list(x.shape)\n    merged = int(shape[-2]) * int(shape[-1])\n    return x.reshape(tuple(shape[:-2] + [merged]))\n\n",
    );
    out.push_str(
        "def tysor_repeat_kv(x, repeats):\n    if repeats <= 0:\n        raise ValueError('repeat_kv repeats must be positive')\n    if x.dim() < 2:\n        raise ValueError('repeat_kv expects rank >= 2')\n    return torch.repeat_interleave(x, repeats, dim=1)\n\n",
    );
    out.push_str("def build_parameters(dtype=torch.float32, device=None):\n    params = {}\n");
    let mut has_params = false;
    for node in &graph.nodes {
        if node.kind != GraphNodeKind::LibraryCtor {
            continue;
        }
        if node.op == "linear" {
            let mut in_features = 0i64;
            let mut out_features = 0i64;
            let mut with_bias = true;
            if node.inputs.len() == 1 {
                let producer = graph
                    .nodes
                    .iter()
                    .find(|candidate| candidate.output == node.inputs[0])
                    .ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                match producer.constant {
                    crate::compiler::frontend_ir::FeValue::Int(value) => {
                        in_features = value;
                        out_features = value;
                    }
                    _ => return Err("PyTorch runtime expected integer linear ctor input".to_string()),
                }
            } else if node.inputs.len() == 2 {
                let first = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[0]).ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                let second = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[1]).ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                match (&first.constant, &second.constant) {
                    (crate::compiler::frontend_ir::FeValue::Int(out), crate::compiler::frontend_ir::FeValue::Bool(bias)) => {
                        in_features = *out;
                        out_features = *out;
                        with_bias = *bias;
                    }
                    (crate::compiler::frontend_ir::FeValue::Int(input), crate::compiler::frontend_ir::FeValue::Int(output)) => {
                        in_features = *input;
                        out_features = *output;
                    }
                    _ => return Err("PyTorch runtime expected constant linear ctor inputs".to_string()),
                }
            } else if node.inputs.len() == 3 {
                let first = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[0]).ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                let second = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[1]).ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                let third = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[2]).ok_or_else(|| "PyTorch runtime could not resolve linear ctor input".to_string())?;
                match (&first.constant, &second.constant, &third.constant) {
                    (
                        crate::compiler::frontend_ir::FeValue::Int(input),
                        crate::compiler::frontend_ir::FeValue::Int(output),
                        crate::compiler::frontend_ir::FeValue::Bool(bias),
                    ) => {
                        in_features = *input;
                        out_features = *output;
                        with_bias = *bias;
                    }
                    _ => return Err("PyTorch runtime expected constant linear ctor inputs".to_string()),
                }
            }
            has_params = true;
            out.push_str(&format!(
                "    params[{}] = tysor_init_linear_weight({}, {}, dtype=dtype, device=device)\n",
                python_quoted(&linear_weight_name(node.output)),
                in_features,
                out_features
            ));
            if with_bias {
                out.push_str(&format!(
                    "    params[{}] = tysor_init_linear_bias({}, dtype=dtype, device=device)\n",
                    python_quoted(&linear_bias_name(node.output)),
                    out_features
                ));
            }
        } else if node.op == "Embedding" {
            let first = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[0]).ok_or_else(|| "PyTorch runtime could not resolve embedding ctor input".to_string())?;
            let second = graph.nodes.iter().find(|candidate| candidate.output == node.inputs[1]).ok_or_else(|| "PyTorch runtime could not resolve embedding ctor input".to_string())?;
            let (num_embeddings, embedding_dim) = match (&first.constant, &second.constant) {
                (crate::compiler::frontend_ir::FeValue::Int(a), crate::compiler::frontend_ir::FeValue::Int(b)) => (*a, *b),
                _ => return Err("PyTorch runtime expected constant embedding ctor inputs".to_string()),
            };
            has_params = true;
            out.push_str(&format!(
                "    params[{}] = tysor_init_embedding_weight({}, {}, dtype=dtype, device=device)\n",
                python_quoted(&format!("embedding_{}_weight", node.output)),
                num_embeddings,
                embedding_dim
            ));
        }
    }
    if !has_params {
        out.push_str("    pass\n");
    }
    out.push_str("    return params\n\n");
    out.push_str("\n_tysor_device = torch.device('cpu')\n\n");
    out.push_str(
        "def _tysor_make_input(shape, dtype):\n    return tysor_pattern_tensor(tuple(shape), 19.0, 8.0, dtype=dtype, device=_tysor_device)\n\n",
    );
    out.push_str(
        "def _tysor_dump_value(value_id, value):\n    if torch.is_tensor(value):\n        value = value.detach().to('cpu')\n        shape = ','.join(str(int(dim)) for dim in value.shape)\n        dtype = str(value.dtype).replace('torch.', '')\n        flat = value.reshape(-1).to(torch.float32)\n        values = ','.join(f'{float(item):.9g}' for item in flat.tolist())\n        print(f'__TYSOR_VALUE__\\t{value_id}\\ttensor\\t{dtype}\\t{shape}\\t{values}')\n    elif isinstance(value, bool):\n        print(f'__TYSOR_VALUE__\\t{value_id}\\tbool\\t{int(value)}')\n    elif isinstance(value, int):\n        print(f'__TYSOR_VALUE__\\t{value_id}\\tint\\t{value}')\n    elif isinstance(value, float):\n        print(f'__TYSOR_VALUE__\\t{value_id}\\tfloat\\t{value:.17g}')\n    else:\n        raise TypeError(f'unsupported tysor runtime value: {type(value)}')\n\n",
    );
    out.push_str(&build_input_bindings(entry, &options.tensor_shapes)?);
    if let Some(parameters) = parameters {
        out.push_str("params = {}\n");
        for (name, tensor) in parameters {
            out.push_str(&format!(
                "params[{}] = {}\n",
                python_quoted(name),
                python_tensor_literal(tensor)
            ));
        }
    } else {
        out.push_str("params = build_parameters(dtype=torch.float32, device=_tysor_device)\n");
    }

    for value in &graph.values {
        if value.is_parameter {
            out.push_str(&format!("{} = {}\n", value_var(value.id), value.name));
        }
    }
    for node in &graph.nodes {
        let target = value_var(node.output);
        match node.kind {
            GraphNodeKind::Constant => match &node.constant {
                crate::compiler::frontend_ir::FeValue::Int(value) => {
                    out.push_str(&format!("{target} = {value}\n"));
                }
                crate::compiler::frontend_ir::FeValue::Float(value) => {
                    out.push_str(&format!("{target} = {:.17}\n", value));
                }
                crate::compiler::frontend_ir::FeValue::Bool(value) => {
                    out.push_str(&format!("{target} = {}\n", if *value { "True" } else { "False" }));
                }
                _ => return Err("Unsupported PyTorch runtime constant".to_string()),
            },
            GraphNodeKind::Binary => {
                let op = match node.binary_op {
                    crate::compiler::lexer::TokenType::Plus => "+",
                    crate::compiler::lexer::TokenType::Minus => "-",
                    crate::compiler::lexer::TokenType::Star => "*",
                    crate::compiler::lexer::TokenType::Slash => "/",
                    _ => return Err("Unsupported PyTorch runtime binary op".to_string()),
                };
                out.push_str(&format!(
                    "{target} = {} {} {}\n",
                    value_var(node.inputs[0]),
                    op,
                    value_var(node.inputs[1])
                ));
            }
            GraphNodeKind::PrimitiveCall => match node.op.as_str() {
                "matmul" => out.push_str(&format!(
                    "{target} = torch.matmul({}, {})\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1])
                )),
                "relu" => out.push_str(&format!("{target} = torch.clamp_min({}, 0.0)\n", value_var(node.inputs[0]))),
                "scale" => out.push_str(&format!(
                    "{target} = {} * {}\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1])
                )),
                _ => return Err(format!("Unsupported primitive PyTorch runtime op '{}'", node.op)),
            },
            GraphNodeKind::LibraryCall => match node.op.as_str() {
                "rms_norm" => out.push_str(&format!(
                    "{target} = tysor_rms_norm({}, {})\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1])
                )),
                "cross_entropy" => out.push_str(&format!(
                    "{target} = tysor_cross_entropy({}, {})\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1])
                )),
                "rope" => out.push_str(&format!(
                    "{target} = tysor_rope({}, {}, {})\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1]),
                    value_var(node.inputs[2])
                )),
                "reshape" => {
                    out.push_str(&format!("{target} = tysor_reshape({}", value_var(node.inputs[0])));
                    for id in &node.inputs[1..] {
                        out.push_str(&format!(", {}", value_var(*id)));
                    }
                    out.push_str(")\n");
                }
                "causal_mask" => out.push_str(&format!("{target} = tysor_causal_mask({})\n", value_var(node.inputs[0]))),
                "flatten_heads" => out.push_str(&format!("{target} = tysor_flatten_heads({})\n", value_var(node.inputs[0]))),
                "repeat_kv" => out.push_str(&format!(
                    "{target} = tysor_repeat_kv({}, {})\n",
                    value_var(node.inputs[0]),
                    value_var(node.inputs[1])
                )),
                _ => return Err(format!("Unsupported library PyTorch runtime op '{}'", node.op)),
            },
            GraphNodeKind::LibraryCtor => {
                out.push_str(&format!("{target} = None\n"));
            }
            GraphNodeKind::Apply => {
                let callee = graph
                    .nodes
                    .iter()
                    .find(|producer| producer.output == node.inputs[0])
                    .ok_or_else(|| format!("Unsupported PyTorch runtime apply op for value {}", node.inputs[0]))?;
                match callee.op.as_str() {
                    "linear" => {
                        let bias_name = linear_bias_name(callee.output);
                        let weight_name = linear_weight_name(callee.output);
                        let with_bias = if callee.inputs.len() == 3 {
                            let producer = graph
                                .nodes
                                .iter()
                                .find(|candidate| candidate.output == callee.inputs[2])
                                .ok_or_else(|| "PyTorch runtime could not resolve linear bias flag".to_string())?;
                            matches!(producer.constant, crate::compiler::frontend_ir::FeValue::Bool(true))
                        } else if callee.inputs.len() == 2 {
                            let producer = graph
                                .nodes
                                .iter()
                                .find(|candidate| candidate.output == callee.inputs[1])
                                .ok_or_else(|| "PyTorch runtime could not resolve linear bias flag".to_string())?;
                            matches!(producer.constant, crate::compiler::frontend_ir::FeValue::Bool(true))
                                || !matches!(producer.constant, crate::compiler::frontend_ir::FeValue::Bool(false))
                        } else {
                            true
                        };
                        out.push_str(&format!(
                            "{target} = tysor_linear({}, params[{}], {})\n",
                            value_var(node.inputs[1]),
                            python_quoted(&weight_name),
                            if with_bias {
                                format!("params[{}]", python_quoted(&bias_name))
                            } else {
                                "None".to_string()
                            }
                        ));
                    }
                    "Embedding" => {
                        out.push_str(&format!(
                            "{target} = tysor_embedding({}, params[{}])\n",
                            value_var(node.inputs[1]),
                            python_quoted(&format!("embedding_{}_weight", callee.output))
                        ));
                    }
                    "SiLU" => out.push_str(&format!("{target} = tysor_silu({})\n", value_var(node.inputs[1]))),
                    "Softmax" => out.push_str(&format!("{target} = tysor_softmax({})\n", value_var(node.inputs[1]))),
                    "Dropout" => out.push_str(&format!(
                        "{target} = tysor_dropout({}, {})\n",
                        value_var(node.inputs[1]),
                        value_var(callee.inputs[0])
                    )),
                    _ => return Err(format!("Unsupported PyTorch runtime apply op '{}'", callee.op)),
                }
            }
        }
    }

    let skip_dump_ids = graph
        .nodes
        .iter()
        .filter(|node| node.kind == GraphNodeKind::LibraryCtor)
        .map(|node| node.output)
        .collect::<std::collections::BTreeSet<_>>();
    out.push_str("print('__TYSOR_BEGIN_VALUES__')\n");
    for value in &graph.values {
        if skip_dump_ids.contains(&value.id) {
            continue;
        }
        out.push_str(&format!("_tysor_dump_value({}, {})\n", value.id, value_var(value.id)));
    }
    out.push_str("print('__TYSOR_END_VALUES__')\n");
    Ok(out)
}

fn write_temp_script(source: &str) -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tysor_pytorch_{stamp}.py"));
    fs::write(&path, source).map_err(|err| format!("Failed to write temporary PyTorch script: {err}"))?;
    Ok(path)
}

fn execute_script_capture(path: &PathBuf) -> Result<String, String> {
    let output = Command::new("python3")
        .arg(path)
        .output()
        .map_err(|err| format!("Failed to launch python3 for PyTorch backend: {err}"))?;
    let _ = fs::remove_file(path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("PyTorch backend execution failed:\n{}{}", stdout, stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_execution_output(
    graph: &crate::ir::graph::GraphFunction,
    output: &str,
) -> Result<GraphExecutionResult, String> {
    let mut result = GraphExecutionResult::default();
    for line in output.lines() {
        if !line.starts_with("__TYSOR_VALUE__\t") {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 4 {
            return Err(format!("Malformed PyTorch runtime value line: {line}"));
        }
        let value_id = fields[1]
            .parse::<usize>()
            .map_err(|_| format!("Malformed PyTorch runtime value id: {line}"))?;
        let value = match fields[2] {
            "int" => GraphRuntimeValue::Int(
                fields[3]
                    .parse::<i64>()
                    .map_err(|_| format!("Malformed int runtime value: {line}"))?,
            ),
            "float" => GraphRuntimeValue::Float(
                fields[3]
                    .parse::<f64>()
                    .map_err(|_| format!("Malformed float runtime value: {line}"))?,
            ),
            "bool" => GraphRuntimeValue::Bool(fields[3] != "0"),
            "tensor" => {
                if fields.len() < 6 {
                    return Err(format!("Malformed PyTorch tensor value line: {line}"));
                }
                let shape = if fields[4].is_empty() {
                    Vec::new()
                } else {
                    fields[4]
                        .split(',')
                        .filter(|item| !item.is_empty())
                        .map(|item| item.parse::<i64>().map_err(|_| format!("Malformed tensor shape: {line}")))
                        .collect::<Result<Vec<_>, _>>()?
                };
                let data = if fields[5].is_empty() {
                    Vec::new()
                } else {
                    fields[5]
                        .split(',')
                        .filter(|item| !item.is_empty())
                        .map(|item| item.parse::<f32>().map_err(|_| format!("Malformed tensor data: {line}")))
                        .collect::<Result<Vec<_>, _>>()?
                };
                GraphRuntimeValue::Tensor(SimpleTensor {
                    shape,
                    data,
                    dtype: fields[3].to_string(),
                })
            }
            kind => return Err(format!("Unsupported PyTorch runtime value kind '{kind}'")),
        };
        result.values.insert(value_id, value);
    }
    for output_id in &graph.outputs {
        if let Some(value) = result.values.get(output_id).cloned() {
            result.outputs.insert(*output_id, value);
        }
    }
    Ok(result)
}

pub fn execute_pytorch_module(
    lowered: &LoweredModule,
    options: &RuntimeRunOptions,
    parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<GraphExecutionResult, String> {
    let entry = find_entry_function(lowered, &options.entry)?;
    let graph = build_graph_function(entry)?;
    let script = build_graph_execution_script(lowered, entry, options, parameters)?;
    let output = execute_script_capture(&write_temp_script(&script)?)?;
    parse_execution_output(&graph, &output)
}

pub fn run_pytorch_forward_module(
    lowered: &LoweredModule,
    options: &RuntimeRunOptions,
) -> Result<(), String> {
    let execution = execute_pytorch_module(lowered, options, None)?;
    let entry = find_entry_function(lowered, &options.entry)?;
    let graph = build_graph_function(entry)?;
    let output_id = *graph
        .outputs
        .first()
        .ok_or_else(|| "Entry function did not return a value".to_string())?;
    let output = execution
        .outputs
        .get(&output_id)
        .ok_or_else(|| "PyTorch runtime could not resolve graph output".to_string())?;
    match output {
        GraphRuntimeValue::Tensor(tensor) => {
            print_tensor(tensor);
            Ok(())
        }
        _ => Err("PyTorch runtime only supports tensor returns".to_string()),
    }
}

pub fn run_pytorch_train_module(_lowered: &LoweredModule, _options: &TrainRunOptions) -> Result<(), String> {
    run_train_module(_lowered, _options)
}
