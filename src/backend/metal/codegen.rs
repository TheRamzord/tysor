use crate::backend::core::execution_plan::{ExecutionPlan, PlanOpKind};
use crate::backend::core::kind::BackendKind;
use crate::compiler::frontend_ir::{FeType, FeTypeKind, FeValue};
use crate::compiler::lexer::TokenType;

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetalKernelInfo {
    pub op_index: usize,
    pub function_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetalCodegenResult {
    pub source: String,
    pub kernels: Vec<MetalKernelInfo>,
}

fn metal_scalar_type(_type_: &FeType) -> &'static str {
    "float"
}

fn sanitize_kernel_name(base: &str, index: usize) -> String {
    format!("tysor_{base}_{index}")
}

fn binary_expr(op: TokenType, lhs: &str, rhs: &str) -> Result<String, String> {
    Ok(match op {
        TokenType::Plus => format!("{lhs} + {rhs}"),
        TokenType::Minus => format!("{lhs} - {rhs}"),
        TokenType::Star => format!("{lhs} * {rhs}"),
        TokenType::Slash => format!("{lhs} / {rhs}"),
        _ => return Err("Unsupported binary op in Metal codegen".to_string()),
    })
}

fn constant_literal(value: &FeValue) -> Result<String, String> {
    Ok(match value {
        FeValue::Int(value) => format!("{value}.0f"),
        FeValue::Float(value) => format!("{value:.6}f"),
        FeValue::Bool(value) => {
            if *value {
                "1.0f".to_string()
            } else {
                "0.0f".to_string()
            }
        }
        _ => return Err("Unsupported constant literal in Metal codegen".to_string()),
    })
}

fn require_producer<'a>(
    plan: &'a ExecutionPlan,
    producers: &BTreeMap<usize, usize>,
    value_id: usize,
) -> Result<&'a crate::backend::core::execution_plan::PlanOp, String> {
    let op_index = producers
        .get(&value_id)
        .ok_or_else(|| format!("Metal codegen could not resolve producer for value {value_id}"))?;
    Ok(&plan.ops[*op_index])
}

fn require_const_bool(plan: &ExecutionPlan, producers: &BTreeMap<usize, usize>, value_id: usize) -> Result<bool, String> {
    let op = require_producer(plan, producers, value_id)?;
    match &op.constant {
        FeValue::Bool(value) if op.kind == PlanOpKind::Constant => Ok(*value),
        _ => Err("Metal codegen expected a bool constant".to_string()),
    }
}

fn require_const_number(
    plan: &ExecutionPlan,
    producers: &BTreeMap<usize, usize>,
    value_id: usize,
) -> Result<f64, String> {
    let op = require_producer(plan, producers, value_id)?;
    if op.kind != PlanOpKind::Constant {
        return Err("Metal codegen expected a constant numeric value".to_string());
    }
    match &op.constant {
        FeValue::Int(value) => Ok(*value as f64),
        FeValue::Float(value) => Ok(*value),
        FeValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err("Metal codegen expected a numeric constant".to_string()),
    }
}

fn emit_matmul_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* lhs [[buffer(0)]],\n    device const {scalar_type}* rhs [[buffer(1)]],\n    device {scalar_type}* out [[buffer(2)]],\n    constant uint& m [[buffer(3)]],\n    constant uint& n [[buffer(4)]],\n    constant uint& k [[buffer(5)]],\n    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= n || gid.y >= m) return;\n  {scalar_type} acc = 0;\n  for (uint kk = 0; kk < k; ++kk) {{\n    acc += lhs[gid.y * k + kk] * rhs[kk * n + gid.x];\n  }}\n  out[gid.y * n + gid.x] = acc;\n}}\n\n"
    ));
}

fn emit_unary_elementwise_kernel(out: &mut String, kernel_name: &str, scalar_type: &str, expr: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& count [[buffer(2)]],\n    uint gid [[thread_position_in_grid]]) {{\n  if (gid >= count) return;\n  {scalar_type} x = in[gid];\n  out[gid] = {expr};\n}}\n\n"
    ));
}

fn emit_copy_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& count [[buffer(2)]],\n    uint gid [[thread_position_in_grid]]) {{\n  if (gid >= count) return;\n  out[gid] = in[gid];\n}}\n\n"
    ));
}

fn emit_embedding_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* indices [[buffer(0)]],\n    device const {scalar_type}* weight [[buffer(1)]],\n    device {scalar_type}* out [[buffer(2)]],\n    constant uint& index_count [[buffer(3)]],\n    constant uint& embedding_dim [[buffer(4)]],\n    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= embedding_dim || gid.y >= index_count) return;\n  uint index = uint(max(float(indices[gid.y]), 0.0));\n  out[gid.y * embedding_dim + gid.x] = weight[index * embedding_dim + gid.x];\n}}\n\n"
    ));
}

fn emit_repeat_kv_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& outer [[buffer(2)]],\n    constant uint& out_heads [[buffer(3)]],\n    constant uint& inner [[buffer(4)]],\n    constant uint& repeats [[buffer(5)]],\n    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= inner || gid.y >= outer * out_heads) return;\n  uint outer_idx = gid.y / out_heads;\n  uint dst_head = gid.y % out_heads;\n  uint src_head = dst_head / repeats;\n  uint in_heads = out_heads / repeats;\n  uint src = (outer_idx * in_heads + src_head) * inner + gid.x;\n  uint dst = gid.y * inner + gid.x;\n  out[dst] = in[src];\n}}\n\n"
    ));
}

fn emit_causal_mask_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& outer [[buffer(2)]],\n    constant uint& q [[buffer(3)]],\n    constant uint& k [[buffer(4)]],\n    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= k || gid.y >= outer * q) return;\n  uint outer_idx = gid.y / q;\n  uint row = gid.y % q;\n  uint index = outer_idx * q * k + row * k + gid.x;\n  {scalar_type} value = in[index];\n  if (gid.x > row) value = {scalar_type}(-10000.0f);\n  out[index] = value;\n}}\n\n"
    ));
}

fn emit_rope_kernel(out: &mut String, kernel_name: &str, scalar_type: &str, head_dim: u32, theta: f64) {
    let half = head_dim / 2;
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& outer [[buffer(2)]],\n    constant uint& seq_len [[buffer(3)]],\n    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= {half}u || gid.y >= outer * seq_len) return;\n  uint outer_idx = gid.y / seq_len;\n  uint pos = gid.y % seq_len;\n  uint base = outer_idx * seq_len * {head_dim}u + pos * {head_dim}u;\n  float inv_freq = pow({theta:.6}f, -(float(gid.x) / float({half}u)));\n  float angle = float(pos) * inv_freq;\n  float c = cos(angle);\n  float s = sin(angle);\n  float x1 = float(in[base + gid.x]);\n  float x2 = float(in[base + {half}u + gid.x]);\n  out[base + gid.x] = {scalar_type}(x1 * c - x2 * s);\n  out[base + {half}u + gid.x] = {scalar_type}(x1 * s + x2 * c);\n}}\n\n"
    ));
}

fn emit_binary_kernel(
    out: &mut String,
    kernel_name: &str,
    scalar_type: &str,
    lhs_type: &FeType,
    rhs_type: &FeType,
    binary_op: TokenType,
    rhs_const: Option<&FeValue>,
    lhs_const: Option<&FeValue>,
) -> Result<(), String> {
    let lhs_tensor = lhs_type.kind == FeTypeKind::Tensor;
    let rhs_tensor = rhs_type.kind == FeTypeKind::Tensor;
    if lhs_tensor && rhs_tensor {
        out.push_str(&format!(
            "kernel void {kernel_name}(\n    device const {scalar_type}* lhs [[buffer(0)]],\n    device const {scalar_type}* rhs [[buffer(1)]],\n    device {scalar_type}* out [[buffer(2)]],\n    constant uint& count [[buffer(3)]],\n    uint gid [[thread_position_in_grid]]) {{\n  if (gid >= count) return;\n  out[gid] = {};\n}}\n\n",
            binary_expr(binary_op, "lhs[gid]", "rhs[gid]")?
        ));
        return Ok(());
    }
    if lhs_tensor {
        let rhs_const = rhs_const.ok_or_else(|| "Metal codegen expected a rhs constant".to_string())?;
        out.push_str(&format!(
            "kernel void {kernel_name}(\n    device const {scalar_type}* lhs [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& count [[buffer(2)]],\n    uint gid [[thread_position_in_grid]]) {{\n  if (gid >= count) return;\n  out[gid] = {};\n}}\n\n",
            binary_expr(binary_op, "lhs[gid]", &constant_literal(rhs_const)?)?
        ));
        return Ok(());
    }
    if rhs_tensor {
        let lhs_const = lhs_const.ok_or_else(|| "Metal codegen expected a lhs constant".to_string())?;
        out.push_str(&format!(
            "kernel void {kernel_name}(\n    device const {scalar_type}* rhs [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& count [[buffer(2)]],\n    uint gid [[thread_position_in_grid]]) {{\n  if (gid >= count) return;\n  out[gid] = {};\n}}\n\n",
            binary_expr(binary_op, &constant_literal(lhs_const)?, "rhs[gid]")?
        ));
        return Ok(());
    }
    Err("Metal codegen does not support scalar-scalar kernels".to_string())
}

fn emit_linear_kernel(out: &mut String, kernel_name: &str, scalar_type: &str, with_bias: bool) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device const {scalar_type}* weight [[buffer(1)]],\n"
    ));
    if with_bias {
        out.push_str(&format!(
            "    device const {scalar_type}* bias [[buffer(2)]],\n    device {scalar_type}* out [[buffer(3)]],\n    constant uint& m [[buffer(4)]],\n    constant uint& n [[buffer(5)]],\n    constant uint& k [[buffer(6)]],\n"
        ));
    } else {
        out.push_str(&format!(
            "    device {scalar_type}* out [[buffer(2)]],\n    constant uint& m [[buffer(3)]],\n    constant uint& n [[buffer(4)]],\n    constant uint& k [[buffer(5)]],\n"
        ));
    }
    out.push_str(&format!(
        "    uint2 gid [[thread_position_in_grid]]) {{\n  if (gid.x >= n || gid.y >= m) return;\n  {scalar_type} acc = 0;\n  for (uint kk = 0; kk < k; ++kk) {{\n    acc += in[gid.y * k + kk] * weight[kk * n + gid.x];\n  }}\n"
    ));
    if with_bias {
        out.push_str("  acc += bias[gid.x];\n");
    }
    out.push_str("  out[gid.y * n + gid.x] = acc;\n}\n\n");
}

fn emit_rms_norm_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& rows [[buffer(2)]],\n    constant uint& width [[buffer(3)]],\n    uint row [[thread_position_in_grid]]) {{\n  if (row >= rows) return;\n  float mean_sq = 0.0f;\n  for (uint i = 0; i < width; ++i) {{\n    float x = float(in[row * width + i]);\n    mean_sq += x * x;\n  }}\n  mean_sq /= float(width);\n  float inv_rms = rsqrt(mean_sq + 1.0e-5f);\n  for (uint i = 0; i < width; ++i) {{\n    out[row * width + i] = {scalar_type}(float(in[row * width + i]) * inv_rms);\n  }}\n}}\n\n"
    ));
}

fn emit_softmax_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* in [[buffer(0)]],\n    device {scalar_type}* out [[buffer(1)]],\n    constant uint& rows [[buffer(2)]],\n    constant uint& width [[buffer(3)]],\n    uint row [[thread_position_in_grid]]) {{\n  if (row >= rows) return;\n  float max_value = float(in[row * width]);\n  for (uint i = 1; i < width; ++i) {{\n    max_value = max(max_value, float(in[row * width + i]));\n  }}\n  float sum = 0.0f;\n  for (uint i = 0; i < width; ++i) {{\n    float e = exp(float(in[row * width + i]) - max_value);\n    out[row * width + i] = {scalar_type}(e);\n    sum += e;\n  }}\n  for (uint i = 0; i < width; ++i) {{\n    out[row * width + i] = {scalar_type}(float(out[row * width + i]) / sum);\n  }}\n}}\n\n"
    ));
}

fn emit_cross_entropy_kernel(out: &mut String, kernel_name: &str, scalar_type: &str) {
    out.push_str(&format!(
        "kernel void {kernel_name}(\n    device const {scalar_type}* logits [[buffer(0)]],\n    device const {scalar_type}* target [[buffer(1)]],\n    device {scalar_type}* out [[buffer(2)]],\n    constant uint& rows [[buffer(3)]],\n    constant uint& width [[buffer(4)]],\n    uint row [[thread_position_in_grid]]) {{\n  if (row >= rows) return;\n  float max_value = float(logits[row * width]);\n  for (uint i = 1; i < width; ++i) {{\n    max_value = max(max_value, float(logits[row * width + i]));\n  }}\n  float sum = 0.0f;\n  for (uint i = 0; i < width; ++i) {{\n    sum += exp(float(logits[row * width + i]) - max_value);\n  }}\n  float loss = 0.0f;\n  for (uint i = 0; i < width; ++i) {{\n    float prob = exp(float(logits[row * width + i]) - max_value) / sum;\n    loss -= float(target[row * width + i]) * log(max(prob, 1.0e-6f));\n  }}\n  out[row] = {scalar_type}(loss);\n}}\n\n"
    ));
}

fn emit_kernel_for_op(
    out: &mut String,
    plan: &ExecutionPlan,
    op_index: usize,
    producers: &BTreeMap<usize, usize>,
) -> Result<Option<String>, String> {
    let op = &plan.ops[op_index];
    let kernel_name = sanitize_kernel_name(if op.op.is_empty() { "op" } else { &op.op }, op_index);
    let scalar_type = metal_scalar_type(&plan.values[op.output].ty);
    out.push_str(&format!("// op {op_index} output=v{} backend={}\n", op.output, plan.backend.as_str()));
    match op.kind {
        PlanOpKind::PrimitiveCall => {
            if op.op == "matmul" {
                emit_matmul_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "relu" {
                emit_unary_elementwise_kernel(out, &kernel_name, scalar_type, &format!("max(x, {scalar_type}(0))"));
                return Ok(Some(kernel_name));
            }
            if op.op == "scale" {
                let rhs = require_producer(plan, producers, op.inputs[1])?;
                emit_binary_kernel(
                    out,
                    &kernel_name,
                    scalar_type,
                    &plan.values[op.inputs[0]].ty,
                    &plan.values[op.inputs[1]].ty,
                    TokenType::Star,
                    Some(&rhs.constant),
                    None,
                )?;
                return Ok(Some(kernel_name));
            }
        }
        PlanOpKind::Binary => {
            let lhs_prod = producers.get(&op.inputs[0]).map(|index| &plan.ops[*index]);
            let rhs_prod = producers.get(&op.inputs[1]).map(|index| &plan.ops[*index]);
            let lhs_const = lhs_prod.filter(|prod| prod.kind == PlanOpKind::Constant).map(|prod| &prod.constant);
            let rhs_const = rhs_prod.filter(|prod| prod.kind == PlanOpKind::Constant).map(|prod| &prod.constant);
            emit_binary_kernel(
                out,
                &kernel_name,
                scalar_type,
                &plan.values[op.inputs[0]].ty,
                &plan.values[op.inputs[1]].ty,
                op.binary_op,
                rhs_const,
                lhs_const,
            )?;
            return Ok(Some(kernel_name));
        }
        PlanOpKind::LibraryCall => {
            if op.op == "rms_norm" {
                emit_rms_norm_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "cross_entropy" {
                emit_cross_entropy_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "reshape" || op.op == "flatten_heads" {
                emit_copy_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "repeat_kv" {
                emit_repeat_kv_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "causal_mask" {
                emit_causal_mask_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if op.op == "rope" {
                let head_dim = require_const_number(plan, producers, op.inputs[1])? as u32;
                let theta = require_const_number(plan, producers, op.inputs[2])?;
                emit_rope_kernel(out, &kernel_name, scalar_type, head_dim, theta);
                return Ok(Some(kernel_name));
            }
        }
        PlanOpKind::Apply => {
            let callee = require_producer(plan, producers, op.inputs[0])?;
            if callee.kind == PlanOpKind::LibraryCtor && callee.op == "linear" {
                let mut with_bias = true;
                if callee.inputs.len() == 3 {
                    with_bias = require_const_bool(plan, producers, callee.inputs[2])?;
                } else if callee.inputs.len() == 2 {
                    let maybe_bool = require_producer(plan, producers, callee.inputs[1])?;
                    if maybe_bool.kind == PlanOpKind::Constant && matches!(maybe_bool.constant, FeValue::Bool(_)) {
                        with_bias = require_const_bool(plan, producers, callee.inputs[1])?;
                    }
                }
                emit_linear_kernel(out, &kernel_name, scalar_type, with_bias);
                return Ok(Some(kernel_name));
            }
            if callee.kind == PlanOpKind::LibraryCtor && callee.op == "Embedding" {
                emit_embedding_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if callee.kind == PlanOpKind::LibraryCtor && callee.op == "SiLU" {
                emit_unary_elementwise_kernel(out, &kernel_name, scalar_type, &format!("x / ({scalar_type}(1) + exp(-x))"));
                return Ok(Some(kernel_name));
            }
            if callee.kind == PlanOpKind::LibraryCtor && callee.op == "Softmax" {
                emit_softmax_kernel(out, &kernel_name, scalar_type);
                return Ok(Some(kernel_name));
            }
            if callee.kind == PlanOpKind::LibraryCtor && callee.op == "Dropout" {
                let probability = require_const_number(plan, producers, callee.inputs[0])?;
                emit_unary_elementwise_kernel(
                    out,
                    &kernel_name,
                    scalar_type,
                    &format!("x * {scalar_type}({:.6}f)", 1.0 - probability),
                );
                return Ok(Some(kernel_name));
            }
        }
        PlanOpKind::Constant | PlanOpKind::LibraryCtor => return Ok(None),
    }
    Err("Metal codegen does not support this operation yet".to_string())
}

pub fn generate_metal_code(plan: &ExecutionPlan) -> Result<MetalCodegenResult, String> {
    if plan.backend != BackendKind::Metal {
        return Err("Metal codegen requires a Metal execution plan".to_string());
    }
    let producers = plan
        .ops
        .iter()
        .enumerate()
        .map(|(index, op)| (op.output, index))
        .collect::<BTreeMap<_, _>>();
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n// Generated by tysor Rust Metal codegen.\n// Host runtime remains responsible for allocation, upload, dispatch, and download.\n\n",
    );
    let mut result = MetalCodegenResult::default();
    for index in 0..plan.ops.len() {
        if let Some(name) = emit_kernel_for_op(&mut source, plan, index, &producers)? {
            result.kernels.push(MetalKernelInfo {
                op_index: index,
                function_name: name,
            });
        }
    }
    result.source = source;
    Ok(result)
}
