use crate::runtime::tensor::{add_in_place, apply_softmax, matmul, ones_like, transpose_2d, zeros_like, SimpleTensor};

#[derive(Debug, Clone, PartialEq)]
pub struct LinearClosure {
    pub in_features: Option<i64>,
    pub out_features: i64,
    pub with_bias: bool,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingClosure {
    pub num_embeddings: i64,
    pub embedding_dim: i64,
    pub dtype: String,
}

impl Default for EmbeddingClosure {
    fn default() -> Self {
        Self {
            num_embeddings: 0,
            embedding_dim: 0,
            dtype: "float32".to_string(),
        }
    }
}

impl Default for LinearClosure {
    fn default() -> Self {
        Self {
            in_features: None,
            out_features: 0,
            with_bias: true,
            dtype: "float32".to_string(),
        }
    }
}

pub fn make_linear_weight(in_features: i64, out_features: i64, dtype: &str) -> SimpleTensor {
    let shape = vec![in_features, out_features];
    let element_count = (in_features * out_features) as usize;
    let data = (0..element_count)
        .map(|index| ((index % 13) as f32 + 1.0) / 16.0)
        .collect();
    SimpleTensor {
        shape,
        data,
        dtype: dtype.to_string(),
    }
}

pub fn make_linear_bias(out_features: i64, dtype: &str) -> SimpleTensor {
    let shape = vec![out_features];
    let data = (0..out_features as usize)
        .map(|index| ((index % 7) as f32 + 1.0) / 32.0)
        .collect();
    SimpleTensor {
        shape,
        data,
        dtype: dtype.to_string(),
    }
}

pub fn make_embedding_weight(num_embeddings: i64, embedding_dim: i64, dtype: &str) -> SimpleTensor {
    let shape = vec![num_embeddings, embedding_dim];
    let element_count = (num_embeddings * embedding_dim) as usize;
    let data = (0..element_count)
        .map(|index| ((index % 17) as f32 + 1.0) / 8.0)
        .collect();
    SimpleTensor {
        shape,
        data,
        dtype: dtype.to_string(),
    }
}

pub fn apply_linear(closure: &LinearClosure, input: &SimpleTensor) -> Result<SimpleTensor, String> {
    if input.shape.len() != 2 {
        return Err("linear currently requires rank-2 input tensors".to_string());
    }
    let inferred_in_features = input.shape[1];
    let in_features = closure.in_features.unwrap_or(inferred_in_features);
    if inferred_in_features != in_features {
        return Err(format!(
            "linear expected input feature size {}, got {}",
            in_features, inferred_in_features
        ));
    }
    let weight = make_linear_weight(in_features, closure.out_features, &closure.dtype);
    let mut output = matmul(input, &weight)?;
    if closure.with_bias {
        let bias = make_linear_bias(closure.out_features, &closure.dtype);
        output = apply_linear_with_parameters(input, &weight, Some(&bias))?;
    }
    Ok(output)
}

pub fn apply_linear_with_parameters(
    input: &SimpleTensor,
    weight: &SimpleTensor,
    bias: Option<&SimpleTensor>,
) -> Result<SimpleTensor, String> {
    let mut output = matmul(input, weight)?;
    if let Some(bias) = bias {
        if bias.shape.len() != 1 || bias.shape[0] != output.shape[1] {
            return Err("linear bias shape mismatch".to_string());
        }
        let batch = output.shape[0] as usize;
        let width = output.shape[1] as usize;
        for row in 0..batch {
            for col in 0..width {
                output.data[row * width + col] += bias.data[col];
            }
        }
    }
    Ok(output)
}

pub fn apply_dropout(input: &SimpleTensor, probability: f64) -> Result<SimpleTensor, String> {
    if !(0.0..1.0).contains(&probability) {
        return Err("dropout probability must be in [0, 1)".to_string());
    }
    let keep_scale = (1.0 - probability) as f32;
    Ok(SimpleTensor {
        shape: input.shape.clone(),
        data: input.data.iter().map(|value| *value * keep_scale).collect(),
        dtype: input.dtype.clone(),
    })
}

pub fn apply_embedding_with_parameters(
    indices: &SimpleTensor,
    weight: &SimpleTensor,
    num_embeddings: i64,
    embedding_dim: i64,
) -> Result<SimpleTensor, String> {
    if weight.shape != vec![num_embeddings, embedding_dim] {
        return Err("embedding weight shape mismatch".to_string());
    }
    let mut output = SimpleTensor {
        shape: {
            let mut shape = indices.shape.clone();
            shape.push(embedding_dim);
            shape
        },
        data: vec![0.0; indices.data.len() * embedding_dim as usize],
        dtype: weight.dtype.clone(),
    };
    for (i, raw_index) in indices.data.iter().enumerate() {
        let index = *raw_index as i64;
        if index < 0 || index >= num_embeddings {
            return Err("Embedding index out of range".to_string());
        }
        for d in 0..embedding_dim as usize {
            output.data[i * embedding_dim as usize + d] =
                weight.data[index as usize * embedding_dim as usize + d];
        }
    }
    Ok(output)
}

pub fn apply_reshape(input: &SimpleTensor, shape: &[i64]) -> Result<SimpleTensor, String> {
    if input.data.len() != shape.iter().product::<i64>() as usize {
        return Err("reshape requires matching element counts".to_string());
    }
    Ok(SimpleTensor {
        shape: shape.to_vec(),
        data: input.data.clone(),
        dtype: input.dtype.clone(),
    })
}

pub fn apply_repeat_kv(input: &SimpleTensor, repeats: i64) -> Result<SimpleTensor, String> {
    if repeats <= 0 {
        return Err("repeat_kv repeats must be positive".to_string());
    }
    if input.shape.len() < 2 {
        return Err("repeat_kv expects rank >= 2".to_string());
    }
    let mut output_shape = input.shape.clone();
    output_shape[1] *= repeats;
    let inner = input.shape[2..].iter().product::<i64>() as usize;
    let outer = input.shape[0] as usize;
    let heads = input.shape[1] as usize;
    let mut data = vec![0.0; output_shape.iter().product::<i64>() as usize];
    for outer_idx in 0..outer {
        for head in 0..heads {
            let src_base = (outer_idx * heads + head) * inner;
            for rep in 0..repeats as usize {
                let dst_head = head * repeats as usize + rep;
                let dst_base = (outer_idx * output_shape[1] as usize + dst_head) * inner;
                data[dst_base..dst_base + inner].copy_from_slice(&input.data[src_base..src_base + inner]);
            }
        }
    }
    Ok(SimpleTensor {
        shape: output_shape,
        data,
        dtype: input.dtype.clone(),
    })
}

pub fn apply_flatten_heads(input: &SimpleTensor) -> Result<SimpleTensor, String> {
    if input.shape.len() < 3 {
        return Ok(input.clone());
    }
    let mut shape = input.shape[..input.shape.len() - 2].to_vec();
    let merged = input.shape[input.shape.len() - 2] * input.shape[input.shape.len() - 1];
    shape.push(merged);
    Ok(SimpleTensor {
        shape,
        data: input.data.clone(),
        dtype: input.dtype.clone(),
    })
}

pub fn apply_causal_mask(input: &SimpleTensor) -> Result<SimpleTensor, String> {
    if input.shape.len() < 2 {
        return Err("causal_mask expects rank >= 2".to_string());
    }
    let mut output = input.clone();
    let q = input.shape[input.shape.len() - 2] as usize;
    let k = input.shape[input.shape.len() - 1] as usize;
    let inner_stride = q * k;
    let outer = input.data.len() / inner_stride.max(1);
    let fill = -1.0e4f32;
    for outer_idx in 0..outer {
        let base = outer_idx * inner_stride;
        for row in 0..q {
            for col in row + 1..k {
                output.data[base + row * k + col] = fill;
            }
        }
    }
    Ok(output)
}

pub fn apply_rope(input: &SimpleTensor, head_dim: i64, theta: f64) -> Result<SimpleTensor, String> {
    if input.shape.is_empty() || *input.shape.last().unwrap() != head_dim {
        return Err("rope head_dim mismatch".to_string());
    }
    if head_dim % 2 != 0 {
        return Err("rope requires an even head_dim".to_string());
    }
    let half = (head_dim / 2) as usize;
    let seq_len = if input.shape.len() >= 2 {
        input.shape[input.shape.len() - 2] as usize
    } else {
        input.shape[0] as usize
    };
    let outer = input.data.len() / (seq_len * head_dim as usize);
    let mut output = input.clone();
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        inv_freq[i] = (theta as f32).powf(-(i as f32) / (half as f32));
    }
    for outer_idx in 0..outer {
        let outer_base = outer_idx * seq_len * head_dim as usize;
        for pos in 0..seq_len {
            let pos_base = outer_base + pos * head_dim as usize;
            for i in 0..half {
                let angle = pos as f32 * inv_freq[i];
                let c = angle.cos();
                let s = angle.sin();
                let x1 = input.data[pos_base + i];
                let x2 = input.data[pos_base + half + i];
                output.data[pos_base + i] = x1 * c - x2 * s;
                output.data[pos_base + half + i] = x1 * s + x2 * c;
            }
        }
    }
    Ok(output)
}

pub fn apply_rms_norm(input: &SimpleTensor, hidden_size: i64) -> Result<SimpleTensor, String> {
    if input.shape.is_empty() || *input.shape.last().unwrap() != hidden_size {
        return Err(format!(
            "rms_norm hidden size mismatch: expected {}, got {:?}",
            hidden_size, input.shape
        ));
    }
    let width = hidden_size as usize;
    let rows = input.data.len() / width;
    let mut output = vec![0.0f32; input.data.len()];
    for row in 0..rows {
        let start = row * width;
        let slice = &input.data[start..start + width];
        let mean_square = slice.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = 1.0 / (mean_square + 1e-5).sqrt();
        for (index, value) in slice.iter().enumerate() {
            output[start + index] = *value * scale;
        }
    }
    Ok(SimpleTensor {
        shape: input.shape.clone(),
        data: output,
        dtype: input.dtype.clone(),
    })
}

pub fn apply_cross_entropy(logits: &SimpleTensor, target: &SimpleTensor) -> Result<SimpleTensor, String> {
    if logits.shape != target.shape {
        return Err("cross_entropy requires logits and target to have identical shapes".to_string());
    }
    let probabilities = apply_softmax(logits)?;
    let width = *logits.shape.last().unwrap_or(&1) as usize;
    let rows = logits.data.len() / width.max(1);
    let mut losses = vec![0.0f32; rows];
    for row in 0..rows {
        let start = row * width;
        let probs = &probabilities.data[start..start + width];
        let targets = &target.data[start..start + width];
        let mut loss = 0.0f32;
        for (probability, target_value) in probs.iter().zip(targets) {
            loss -= *target_value * probability.max(1e-6).ln();
        }
        losses[row] = loss;
    }
    Ok(SimpleTensor {
        shape: vec![rows as i64, 1],
        data: losses,
        dtype: logits.dtype.clone(),
    })
}

pub fn backward_dropout(grad_output: &SimpleTensor, probability: f64) -> Result<SimpleTensor, String> {
    apply_dropout(grad_output, probability)
}

pub fn backward_silu(input: &SimpleTensor, grad_output: &SimpleTensor) -> Result<SimpleTensor, String> {
    if input.shape != grad_output.shape {
        return Err("backward_silu shape mismatch".to_string());
    }
    let data = input
        .data
        .iter()
        .zip(&grad_output.data)
        .map(|(input_value, grad)| {
            let sigmoid = 1.0 / (1.0 + (-*input_value).exp());
            grad * (sigmoid + *input_value * sigmoid * (1.0 - sigmoid))
        })
        .collect();
    Ok(SimpleTensor {
        shape: input.shape.clone(),
        data,
        dtype: input.dtype.clone(),
    })
}

pub fn backward_softmax(output: &SimpleTensor, grad_output: &SimpleTensor) -> Result<SimpleTensor, String> {
    if output.shape != grad_output.shape {
        return Err("backward_softmax shape mismatch".to_string());
    }
    Ok(SimpleTensor {
        shape: output.shape.clone(),
        data: output
            .data
            .iter()
            .zip(&grad_output.data)
            .map(|(out, grad)| out * (1.0 - out) * grad)
            .collect(),
        dtype: output.dtype.clone(),
    })
}

pub fn backward_rms_norm_input(input: &SimpleTensor, _hidden_size: i64, grad_output: &SimpleTensor) -> Result<SimpleTensor, String> {
    if input.shape != grad_output.shape {
        return Err("backward_rms_norm_input shape mismatch".to_string());
    }
    Ok(grad_output.clone())
}

pub fn backward_reshape_input(grad: &SimpleTensor, input: &SimpleTensor) -> Result<SimpleTensor, String> {
    if grad.data.len() != input.data.len() {
        return Err("reshape backward requires matching element counts".to_string());
    }
    Ok(SimpleTensor {
        shape: input.shape.clone(),
        data: grad.data.clone(),
        dtype: grad.dtype.clone(),
    })
}

pub fn backward_flatten_heads_input(grad: &SimpleTensor, input: &SimpleTensor) -> Result<SimpleTensor, String> {
    backward_reshape_input(grad, input)
}

pub fn backward_repeat_kv_input(grad: &SimpleTensor, input: &SimpleTensor, repeats: i64) -> Result<SimpleTensor, String> {
    if repeats <= 0 {
        return Err("repeat_kv backward requires repeats > 0".to_string());
    }
    if input.shape.len() < 2 || grad.shape.len() != input.shape.len() {
        return Err("repeat_kv backward expects matching ranks >= 2".to_string());
    }
    let inner = input.shape[2..].iter().product::<i64>() as usize;
    let outer = input.shape[0] as usize;
    let heads = input.shape[1] as usize;
    let mut result = SimpleTensor {
        shape: input.shape.clone(),
        data: vec![0.0; input.data.len()],
        dtype: grad.dtype.clone(),
    };
    for outer_idx in 0..outer {
        for head in 0..heads {
            let out_base = (outer_idx * heads + head) * inner;
            for rep in 0..repeats as usize {
                let grad_head = head * repeats as usize + rep;
                let grad_base = (outer_idx * grad.shape[1] as usize + grad_head) * inner;
                for i in 0..inner {
                    result.data[out_base + i] += grad.data[grad_base + i];
                }
            }
        }
    }
    Ok(result)
}

pub fn backward_causal_mask_input(grad: &SimpleTensor) -> Result<SimpleTensor, String> {
    if grad.shape.len() < 2 {
        return Err("causal_mask backward expects rank >= 2".to_string());
    }
    let mut result = grad.clone();
    let q = grad.shape[grad.shape.len() - 2] as usize;
    let k = grad.shape[grad.shape.len() - 1] as usize;
    let inner_stride = q * k;
    let outer = grad.data.len() / inner_stride.max(1);
    for outer_idx in 0..outer {
        let base = outer_idx * inner_stride;
        for row in 0..q {
            for col in row + 1..k {
                result.data[base + row * k + col] = 0.0;
            }
        }
    }
    Ok(result)
}

pub fn backward_rope_input(grad: &SimpleTensor, head_dim: i64, theta: f64) -> Result<SimpleTensor, String> {
    if grad.shape.is_empty() || *grad.shape.last().unwrap() != head_dim {
        return Err("rope backward head_dim mismatch".to_string());
    }
    if head_dim % 2 != 0 {
        return Err("rope backward requires even head_dim".to_string());
    }
    let half = (head_dim / 2) as usize;
    let seq_len = if grad.shape.len() >= 2 {
        grad.shape[grad.shape.len() - 2] as usize
    } else {
        grad.shape[0] as usize
    };
    let outer = grad.data.len() / (seq_len * head_dim as usize);
    let mut result = SimpleTensor {
        shape: grad.shape.clone(),
        data: vec![0.0; grad.data.len()],
        dtype: grad.dtype.clone(),
    };
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        inv_freq[i] = (theta as f32).powf(-(i as f32) / (half as f32));
    }
    for outer_idx in 0..outer {
        let outer_base = outer_idx * seq_len * head_dim as usize;
        for pos in 0..seq_len {
            let pos_base = outer_base + pos * head_dim as usize;
            for i in 0..half {
                let angle = pos as f32 * inv_freq[i];
                let c = angle.cos();
                let s = angle.sin();
                let gy1 = grad.data[pos_base + i];
                let gy2 = grad.data[pos_base + half + i];
                result.data[pos_base + i] = gy1 * c + gy2 * s;
                result.data[pos_base + half + i] = -gy1 * s + gy2 * c;
            }
        }
    }
    Ok(result)
}

pub fn backward_cross_entropy_logits(logits: &SimpleTensor, target: &SimpleTensor) -> Result<SimpleTensor, String> {
    if logits.shape != target.shape {
        return Err("backward_cross_entropy_logits shape mismatch".to_string());
    }
    let mut grad = apply_softmax(logits)?;
    for (grad_value, target_value) in grad.data.iter_mut().zip(&target.data) {
        *grad_value -= *target_value;
    }
    Ok(grad)
}

pub fn backward_cross_entropy_target(logits: &SimpleTensor, target: &SimpleTensor) -> Result<SimpleTensor, String> {
    if logits.shape != target.shape {
        return Err("backward_cross_entropy_target shape mismatch".to_string());
    }
    Ok(zeros_like(target))
}

pub fn backward_embedding_weight(
    grad: &SimpleTensor,
    indices: &SimpleTensor,
    num_embeddings: i64,
    embedding_dim: i64,
) -> Result<SimpleTensor, String> {
    let mut weight_grad = SimpleTensor {
        shape: vec![num_embeddings, embedding_dim],
        data: vec![0.0; (num_embeddings * embedding_dim) as usize],
        dtype: grad.dtype.clone(),
    };
    for (i, raw_index) in indices.data.iter().enumerate() {
        let index = *raw_index as i64;
        if index < 0 || index >= num_embeddings {
            continue;
        }
        for d in 0..embedding_dim as usize {
            weight_grad.data[index as usize * embedding_dim as usize + d] +=
                grad.data[i * embedding_dim as usize + d];
        }
    }
    Ok(weight_grad)
}

pub fn backward_linear_input(grad_output: &SimpleTensor, weight: &SimpleTensor) -> Result<SimpleTensor, String> {
    matmul(grad_output, &transpose_2d(weight)?)
}

pub fn backward_linear_weight(grad_output: &SimpleTensor, input: &SimpleTensor) -> Result<SimpleTensor, String> {
    matmul(&transpose_2d(input)?, grad_output)
}

pub fn backward_linear_bias(grad_output: &SimpleTensor) -> Result<SimpleTensor, String> {
    if grad_output.shape.len() != 2 {
        return Err("backward_linear_bias currently requires rank-2 gradients".to_string());
    }
    let batch = grad_output.shape[0] as usize;
    let width = grad_output.shape[1] as usize;
    let mut bias = SimpleTensor {
        shape: vec![width as i64],
        data: vec![0.0; width],
        dtype: grad_output.dtype.clone(),
    };
    for row in 0..batch {
        let row_tensor = SimpleTensor {
            shape: vec![width as i64],
            data: grad_output.data[row * width..(row + 1) * width].to_vec(),
            dtype: grad_output.dtype.clone(),
        };
        add_in_place(&mut bias, &row_tensor)?;
    }
    Ok(bias)
}

pub fn load_runtime_layers() -> Result<(), String> {
    let probe = SimpleTensor {
        shape: vec![1, 1],
        data: vec![1.0],
        dtype: "float32".to_string(),
    };
    let _ = ones_like(&probe);
    let _ = transpose_2d(&probe)?;
    Ok(())
}
