use crate::compiler::frontend_ir::{FeExecutionRun, FeValue, LoweredModule};
use crate::compiler::lexer::TokenType;
use crate::ir::builder::build_graph_function;
use crate::ir::graph::{GraphFunction, GraphNode, GraphNodeKind, GraphValue};

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PyTorchCodegenResult {
    pub source: String,
}

fn find_entry_function<'a>(lowered: &'a LoweredModule, entry: &str) -> Result<&'a crate::compiler::frontend_ir::FeFunction, String> {
    lowered
        .functions
        .iter()
        .find(|function| function.name == entry)
        .ok_or_else(|| format!("Entry function '{}' not found in lowered module", entry))
}

fn find_execution_run<'a>(lowered: &'a LoweredModule, entry: &str) -> Option<&'a FeExecutionRun> {
    lowered
        .execution_plan
        .as_ref()
        .and_then(|plan| plan.runs.iter().find(|run| run.model_name == entry))
}

fn sanitize_ident(value: &str, fallback: &str) -> String {
    let mut out = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' })
        .collect::<String>();
    if out.is_empty() {
        out = fallback.to_string();
    }
    if !out.chars().next().unwrap_or('_').is_ascii_alphabetic() && !out.starts_with('_') {
        out.insert(0, '_');
    }
    out
}

fn to_class_name(value: &str, fallback: &str) -> String {
    let ident = sanitize_ident(value, fallback);
    let mut result = String::new();
    let mut capitalize = true;
    for ch in ident.chars() {
        if ch == '_' {
            capitalize = true;
            continue;
        }
        if capitalize {
            result.push(ch.to_ascii_uppercase());
            capitalize = false;
        } else {
            result.push(ch);
        }
    }
    if result.is_empty() {
        result = fallback.to_string();
    }
    if !result.ends_with("Model") {
        result.push_str("Model");
    }
    result
}

fn python_literal(value: &FeValue) -> Result<String, String> {
    Ok(match value {
        FeValue::Int(value) => value.to_string(),
        FeValue::Float(value) => format!("{value:.6}"),
        FeValue::Bool(value) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        _ => return Err("PyTorch codegen encountered an unsupported constant".to_string()),
    })
}

fn binary_op_text(op: TokenType) -> Result<&'static str, String> {
    Ok(match op {
        TokenType::Plus => "+",
        TokenType::Minus => "-",
        TokenType::Star => "*",
        TokenType::Slash => "/",
        _ => return Err("PyTorch codegen encountered an unsupported binary op".to_string()),
    })
}

fn value_name(value: &GraphValue) -> String {
    if value.name.is_empty() {
        format!("v{}", value.id)
    } else {
        sanitize_ident(&value.name, &format!("v{}", value.id))
    }
}

fn find_producer<'a>(graph: &'a GraphFunction, value_id: usize) -> Option<&'a GraphNode> {
    graph.nodes.iter().find(|node| node.output == value_id)
}

fn require_producer<'a>(graph: &'a GraphFunction, value_id: usize) -> Result<&'a GraphNode, String> {
    find_producer(graph, value_id).ok_or_else(|| format!("PyTorch codegen could not resolve producer for value {value_id}"))
}

fn build_ctor_constant_values(graph: &GraphFunction) -> BTreeSet<usize> {
    let mut ctor_constant_values = BTreeSet::new();
    for node in &graph.nodes {
        if node.kind != GraphNodeKind::LibraryCtor {
            continue;
        }
        for input_id in &node.inputs {
            if let Some(producer) = find_producer(graph, *input_id) {
                if producer.kind == GraphNodeKind::Constant {
                    ctor_constant_values.insert(*input_id);
                }
            }
        }
    }
    ctor_constant_values
}

fn build_consumers(graph: &GraphFunction) -> BTreeMap<usize, Vec<GraphNodeKind>> {
    let mut consumers = BTreeMap::new();
    for node in &graph.nodes {
        for input_id in &node.inputs {
            consumers.entry(*input_id).or_insert_with(Vec::new).push(node.kind);
        }
    }
    consumers
}

fn emit_helpers(out: &mut String) {
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
}

fn linear_spec(node: &GraphNode, graph: &GraphFunction) -> Result<(i64, i64, bool), String> {
    fn const_i64(graph: &GraphFunction, value_id: usize) -> Result<i64, String> {
        let producer = require_producer(graph, value_id)?;
        match producer.constant {
            FeValue::Int(value) if producer.kind == GraphNodeKind::Constant => Ok(value),
            _ => Err("PyTorch codegen expected an integer constant".to_string()),
        }
    }
    fn const_bool(graph: &GraphFunction, value_id: usize) -> Result<bool, String> {
        let producer = require_producer(graph, value_id)?;
        match producer.constant {
            FeValue::Bool(value) if producer.kind == GraphNodeKind::Constant => Ok(value),
            _ => Err("PyTorch codegen expected a bool constant".to_string()),
        }
    }
    if node.inputs.len() == 1 {
        let out_features = const_i64(graph, node.inputs[0])?;
        return Ok((out_features, out_features, true));
    }
    if node.inputs.len() == 2 {
        if let Ok(with_bias) = const_bool(graph, node.inputs[1]) {
            let out_features = const_i64(graph, node.inputs[0])?;
            return Ok((out_features, out_features, with_bias));
        }
        return Ok((const_i64(graph, node.inputs[0])?, const_i64(graph, node.inputs[1])?, true));
    }
    Ok((
        const_i64(graph, node.inputs[0])?,
        const_i64(graph, node.inputs[1])?,
        const_bool(graph, node.inputs[2])?,
    ))
}

fn embedding_spec(node: &GraphNode, graph: &GraphFunction) -> Result<(i64, i64), String> {
    fn const_i64(graph: &GraphFunction, value_id: usize) -> Result<i64, String> {
        let producer = require_producer(graph, value_id)?;
        match producer.constant {
            FeValue::Int(value) if producer.kind == GraphNodeKind::Constant => Ok(value),
            _ => Err("PyTorch codegen expected an integer constant".to_string()),
        }
    }
    Ok((const_i64(graph, node.inputs[0])?, const_i64(graph, node.inputs[1])?))
}

fn emit_train_config(out: &mut String, lowered: &LoweredModule, entry: &str) {
    let Some(run) = find_execution_run(lowered, entry) else {
        return;
    };
    out.push_str("\nTRAIN_CONFIG = {\n");
    if let Some(FeValue::String(value)) = &run.backend {
        out.push_str(&format!("    \"backend\": \"{}\",\n", value));
    }
    if let Some(FeValue::String(value)) = &run.optimizer {
        out.push_str(&format!("    \"optimizer\": \"{}\",\n", value));
    }
    if let Some(value) = &run.learning_rate {
        match value {
            FeValue::Float(v) => out.push_str(&format!("    \"lr\": {},\n", v)),
            FeValue::Int(v) => out.push_str(&format!("    \"lr\": {},\n", v)),
            _ => {}
        }
    }
    if let Some(FeValue::Int(value)) = &run.iteration {
        out.push_str(&format!("    \"iteration\": {},\n", value));
    }
    if let Some(symbol) = &run.objective_symbol {
        out.push_str(&format!("    \"objective\": \"{}\",\n", symbol));
    }
    out.push_str("}\n");
}

pub fn generate_primitive_pytorch_module(lowered: &LoweredModule, entry: &str) -> Result<PyTorchCodegenResult, String> {
    let entry_fn = find_entry_function(lowered, entry)?;
    let graph = build_graph_function(entry_fn)?;
    let consumers = build_consumers(&graph);
    let ctor_constant_values = build_ctor_constant_values(&graph);
    let mut value_exprs = BTreeMap::new();
    for value in &graph.values {
        if value.is_parameter {
            value_exprs.insert(value.id, value_name(value));
        }
    }

    let mut out = String::from("import torch\n\n");
    emit_helpers(&mut out);
    out.push_str("def build_parameters(dtype=torch.float32, device=None):\n    params = {}\n");
    let mut has_params = false;
    for node in &graph.nodes {
        if node.kind == GraphNodeKind::LibraryCtor && node.op == "linear" {
            let (in_features, out_features, with_bias) = linear_spec(node, &graph)?;
            has_params = true;
            out.push_str(&format!(
                "    params[\"linear_{}.weight\"] = tysor_init_linear_weight({}, {}, dtype=dtype, device=device)\n",
                node.output, in_features, out_features
            ));
            if with_bias {
                out.push_str(&format!(
                    "    params[\"linear_{}.bias\"] = tysor_init_linear_bias({}, dtype=dtype, device=device)\n",
                    node.output, out_features
                ));
            }
        } else if node.kind == GraphNodeKind::LibraryCtor && node.op == "Embedding" {
            let (num_embeddings, embedding_dim) = embedding_spec(node, &graph)?;
            has_params = true;
            out.push_str(&format!(
                "    params[\"embedding_{}_weight\"] = tysor_init_embedding_weight({}, {}, dtype=dtype, device=device)\n",
                node.output, num_embeddings, embedding_dim
            ));
        }
    }
    if !has_params {
        out.push_str("    pass\n");
    }
    out.push_str("    return params\n\n");

    let model_name = sanitize_ident(entry, "model");
    out.push_str(&format!("def {}(", model_name));
    for (index, (name, _)) in entry_fn.params.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        out.push_str(&sanitize_ident(name, name));
    }
    if !entry_fn.params.is_empty() {
        out.push_str(", ");
    }
    out.push_str("params):\n");

    for node in &graph.nodes {
        let output = &graph.values[node.output];
        let target = value_name(output);
        match node.kind {
            GraphNodeKind::Constant => {
                value_exprs.insert(node.output, python_literal(&node.constant)?);
                if ctor_constant_values.contains(&node.output) {
                    continue;
                }
                let ctor_only = consumers
                    .get(&node.output)
                    .map(|kinds| kinds.iter().all(|kind| *kind == GraphNodeKind::LibraryCtor))
                    .unwrap_or(false);
                if ctor_only {
                    continue;
                }
                out.push_str(&format!("    {} = {}\n", target, value_exprs[&node.output]));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::Binary => {
                out.push_str(&format!(
                    "    {} = {} {} {}\n",
                    target,
                    value_exprs[&node.inputs[0]],
                    binary_op_text(node.binary_op)?,
                    value_exprs[&node.inputs[1]]
                ));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::PrimitiveCall => {
                let expr = match node.op.as_str() {
                    "matmul" => format!("torch.matmul({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    "relu" => format!("torch.clamp_min({}, 0.0)", value_exprs[&node.inputs[0]]),
                    "scale" => format!("{} * {}", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    _ => return Err(format!("PyTorch export does not support primitive op '{}'", node.op)),
                };
                out.push_str(&format!("    {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::LibraryCall => {
                let expr = match node.op.as_str() {
                    "rms_norm" => format!("tysor_rms_norm({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    "cross_entropy" => {
                        format!("tysor_cross_entropy({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]])
                    }
                    "rope" => format!(
                        "tysor_rope({}, {}, {})",
                        value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]], value_exprs[&node.inputs[2]]
                    ),
                    "reshape" => {
                        let mut expr = format!("tysor_reshape({}", value_exprs[&node.inputs[0]]);
                        for id in &node.inputs[1..] {
                            expr.push_str(&format!(", {}", value_exprs[id]));
                        }
                        expr.push(')');
                        expr
                    }
                    "causal_mask" => format!("tysor_causal_mask({})", value_exprs[&node.inputs[0]]),
                    "flatten_heads" => format!("tysor_flatten_heads({})", value_exprs[&node.inputs[0]]),
                    "repeat_kv" => format!("tysor_repeat_kv({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    _ => return Err(format!("PyTorch export does not support library op '{}'", node.op)),
                };
                out.push_str(&format!("    {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::LibraryCtor => {}
            GraphNodeKind::Apply => {
                let callee = require_producer(&graph, node.inputs[0])?;
                let expr = match callee.op.as_str() {
                    "linear" => {
                        let (_, _, with_bias) = linear_spec(callee, &graph)?;
                        let bias = if with_bias {
                            format!("params[\"linear_{}.bias\"]", callee.output)
                        } else {
                            "None".to_string()
                        };
                        format!(
                            "tysor_linear({}, params[\"linear_{}.weight\"], {})",
                            value_exprs[&node.inputs[1]], callee.output, bias
                        )
                    }
                    "Embedding" => {
                        format!(
                            "tysor_embedding({}, params[\"embedding_{}_weight\"])",
                            value_exprs[&node.inputs[1]], callee.output
                        )
                    }
                    "SiLU" => format!("tysor_silu({})", value_exprs[&node.inputs[1]]),
                    "Softmax" => format!("tysor_softmax({})", value_exprs[&node.inputs[1]]),
                    "Dropout" => {
                        let probability = match &require_producer(&graph, callee.inputs[0])?.constant {
                            FeValue::Float(value) => *value,
                            FeValue::Int(value) => *value as f64,
                            FeValue::Bool(value) => {
                                if *value {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            _ => return Err("Unsupported dropout probability".to_string()),
                        };
                        format!("tysor_dropout({}, {:.6})", value_exprs[&node.inputs[1]], probability)
                    }
                    _ => return Err(format!("PyTorch export does not support apply op '{}'", callee.op)),
                };
                out.push_str(&format!("    {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
        }
    }

    if graph.outputs.is_empty() {
        out.push_str("    return None\n");
    } else if graph.outputs.len() == 1 {
        out.push_str(&format!("    return {}\n", value_exprs[&graph.outputs[0]]));
    } else {
        out.push_str("    return (");
        for (index, output_id) in graph.outputs.iter().enumerate() {
            if index != 0 {
                out.push_str(", ");
            }
            out.push_str(&value_exprs[output_id]);
        }
        out.push_str(")\n");
    }

    emit_train_config(&mut out, lowered, entry);
    Ok(PyTorchCodegenResult { source: out })
}

pub fn generate_standalone_pytorch_module(lowered: &LoweredModule, entry: &str) -> Result<PyTorchCodegenResult, String> {
    let entry_fn = find_entry_function(lowered, entry)?;
    let graph = build_graph_function(entry_fn)?;
    let consumers = build_consumers(&graph);
    let ctor_constant_values = build_ctor_constant_values(&graph);
    let mut value_exprs = BTreeMap::new();
    for value in &graph.values {
        if value.is_parameter {
            value_exprs.insert(value.id, value_name(value));
        }
    }
    let class_name = to_class_name(entry, "TysorModel");

    let mut out = String::from("import torch\nimport torch.nn as nn\n\n");
    emit_helpers(&mut out);
    out.push_str(&format!("class {}(nn.Module):\n    def __init__(self):\n        super().__init__()\n", class_name));
    let mut has_members = false;
    for node in &graph.nodes {
        if node.kind == GraphNodeKind::LibraryCtor && node.op == "linear" {
            let (in_features, out_features, with_bias) = linear_spec(node, &graph)?;
            has_members = true;
            out.push_str(&format!(
                "        self.linear_{} = nn.Linear({}, {}, bias={})\n        with torch.no_grad():\n            self.linear_{}.weight.copy_(tysor_init_linear_weight({}, {}))\n",
                node.output,
                in_features,
                out_features,
                if with_bias { "True" } else { "False" },
                node.output,
                in_features,
                out_features
            ));
            if with_bias {
                out.push_str(&format!(
                    "            self.linear_{}.bias.copy_(tysor_init_linear_bias({}))\n",
                    node.output, out_features
                ));
            }
        } else if node.kind == GraphNodeKind::LibraryCtor && node.op == "Embedding" {
            let (num_embeddings, embedding_dim) = embedding_spec(node, &graph)?;
            has_members = true;
            out.push_str(&format!(
                "        self.embedding_{} = nn.Embedding({}, {})\n        with torch.no_grad():\n            self.embedding_{}.weight.copy_(tysor_init_embedding_weight({}, {}))\n",
                node.output, num_embeddings, embedding_dim, node.output, num_embeddings, embedding_dim
            ));
        } else if node.kind == GraphNodeKind::LibraryCtor && node.op == "SiLU" {
            has_members = true;
            out.push_str(&format!("        self.silu_{} = nn.SiLU()\n", node.output));
        } else if node.kind == GraphNodeKind::LibraryCtor && node.op == "Softmax" {
            has_members = true;
            out.push_str(&format!("        self.softmax_{} = nn.Softmax(dim=-1)\n", node.output));
        } else if node.kind == GraphNodeKind::LibraryCtor && node.op == "Dropout" {
            let probability = match &require_producer(&graph, node.inputs[0])?.constant {
                FeValue::Float(value) => *value,
                FeValue::Int(value) => *value as f64,
                _ => 0.0,
            };
            has_members = true;
            out.push_str(&format!("        self.dropout_{} = nn.Dropout(p={:.6})\n", node.output, probability));
        }
    }
    if !has_members {
        out.push_str("        pass\n");
    }

    out.push_str("\n    def forward(self");
    for (name, _) in &entry_fn.params {
        out.push_str(&format!(", {}", sanitize_ident(name, name)));
    }
    out.push_str("):\n");

    for node in &graph.nodes {
        let output = &graph.values[node.output];
        let target = value_name(output);
        match node.kind {
            GraphNodeKind::Constant => {
                value_exprs.insert(node.output, python_literal(&node.constant)?);
                if ctor_constant_values.contains(&node.output) {
                    continue;
                }
                let ctor_only = consumers
                    .get(&node.output)
                    .map(|kinds| kinds.iter().all(|kind| *kind == GraphNodeKind::LibraryCtor))
                    .unwrap_or(false);
                if ctor_only {
                    continue;
                }
                out.push_str(&format!("        {} = {}\n", target, value_exprs[&node.output]));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::Binary => {
                out.push_str(&format!(
                    "        {} = {} {} {}\n",
                    target,
                    value_exprs[&node.inputs[0]],
                    binary_op_text(node.binary_op)?,
                    value_exprs[&node.inputs[1]]
                ));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::PrimitiveCall => {
                let expr = match node.op.as_str() {
                    "matmul" => format!("torch.matmul({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    "relu" => format!("torch.relu({})", value_exprs[&node.inputs[0]]),
                    "scale" => format!("{} * {}", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    _ => return Err(format!("Standalone PyTorch export does not support primitive op '{}'", node.op)),
                };
                out.push_str(&format!("        {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::LibraryCall => {
                let expr = match node.op.as_str() {
                    "rms_norm" => format!("tysor_rms_norm({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    "cross_entropy" => {
                        format!("tysor_cross_entropy({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]])
                    }
                    "rope" => format!(
                        "tysor_rope({}, {}, {})",
                        value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]], value_exprs[&node.inputs[2]]
                    ),
                    "reshape" => {
                        let mut expr = format!("tysor_reshape({}", value_exprs[&node.inputs[0]]);
                        for id in &node.inputs[1..] {
                            expr.push_str(&format!(", {}", value_exprs[id]));
                        }
                        expr.push(')');
                        expr
                    }
                    "causal_mask" => format!("tysor_causal_mask({})", value_exprs[&node.inputs[0]]),
                    "flatten_heads" => format!("tysor_flatten_heads({})", value_exprs[&node.inputs[0]]),
                    "repeat_kv" => format!("tysor_repeat_kv({}, {})", value_exprs[&node.inputs[0]], value_exprs[&node.inputs[1]]),
                    _ => return Err(format!("Standalone PyTorch export does not support library op '{}'", node.op)),
                };
                out.push_str(&format!("        {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
            GraphNodeKind::LibraryCtor => {}
            GraphNodeKind::Apply => {
                let callee = require_producer(&graph, node.inputs[0])?;
                let expr = match callee.op.as_str() {
                    "linear" => format!("self.linear_{}({})", callee.output, value_exprs[&node.inputs[1]]),
                    "Embedding" => format!("self.embedding_{}({})", callee.output, value_exprs[&node.inputs[1]]),
                    "SiLU" => format!("self.silu_{}({})", callee.output, value_exprs[&node.inputs[1]]),
                    "Softmax" => format!("self.softmax_{}({})", callee.output, value_exprs[&node.inputs[1]]),
                    "Dropout" => format!("self.dropout_{}({})", callee.output, value_exprs[&node.inputs[1]]),
                    _ => {
                        return Err(format!("Standalone PyTorch export does not support apply op '{}'", callee.op))
                    }
                };
                out.push_str(&format!("        {} = {}\n", target, expr));
                value_exprs.insert(node.output, target);
            }
        }
    }

    if graph.outputs.is_empty() {
        out.push_str("        return None\n");
    } else if graph.outputs.len() == 1 {
        out.push_str(&format!("        return {}\n", value_exprs[&graph.outputs[0]]));
    } else {
        out.push_str("        return (");
        for (index, output_id) in graph.outputs.iter().enumerate() {
            if index != 0 {
                out.push_str(", ");
            }
            out.push_str(&value_exprs[output_id]);
        }
        out.push_str(")\n");
    }

    out.push_str(&format!(
        "\ndef build_model(device=None):\n    model = {}()\n    if device is not None:\n        model = model.to(device)\n    return model\n",
        class_name
    ));
    emit_train_config(&mut out, lowered, entry);
    Ok(PyTorchCodegenResult { source: out })
}
