use crate::compiler::lexer::TokenType;

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleTensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
    pub dtype: String,
}

impl Default for SimpleTensor {
    fn default() -> Self {
        Self {
            shape: Vec::new(),
            data: Vec::new(),
            dtype: "float32".to_string(),
        }
    }
}

pub fn num_elements(shape: &[i64]) -> usize {
    shape.iter().copied().map(|dim| dim as usize).product::<usize>().max(1)
}

pub fn make_synthetic_tensor(shape: &[i64], dtype: impl Into<String>) -> SimpleTensor {
    let element_count = num_elements(shape);
    let data = (0..element_count)
        .map(|index| ((index % 17) + 1) as f32 / 8.0)
        .collect();
    SimpleTensor {
        shape: shape.to_vec(),
        data,
        dtype: dtype.into(),
    }
}

pub fn print_tensor(tensor: &SimpleTensor) {
    println!("\n--- Execution Output ---");
    print_tensor_body(tensor);
    println!("------------------------");
}

pub fn print_named_tensor(name: &str, tensor: &SimpleTensor) {
    if !name.is_empty() {
        println!("{name}:");
    }
    print_tensor_body(tensor);
}

fn print_tensor_body(tensor: &SimpleTensor) {
    println!(
        "shape=[{}] dtype={}",
        tensor
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        tensor.dtype
    );
    println!("values={}", format_tensor_values(&tensor.data));
}

fn format_tensor_values(data: &[f32]) -> String {
    let parts = data
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

pub fn zeros_like(tensor: &SimpleTensor) -> SimpleTensor {
    SimpleTensor {
        shape: tensor.shape.clone(),
        data: vec![0.0; tensor.data.len()],
        dtype: tensor.dtype.clone(),
    }
}

pub fn ones_like(tensor: &SimpleTensor) -> SimpleTensor {
    SimpleTensor {
        shape: tensor.shape.clone(),
        data: vec![1.0; tensor.data.len()],
        dtype: tensor.dtype.clone(),
    }
}

pub fn negate(tensor: &SimpleTensor) -> SimpleTensor {
    SimpleTensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| -*value).collect(),
        dtype: tensor.dtype.clone(),
    }
}

pub fn add_in_place(dst: &mut SimpleTensor, src: &SimpleTensor) -> Result<(), String> {
    if dst.shape != src.shape {
        return Err("tensor shape mismatch".to_string());
    }
    for (dst_value, src_value) in dst.data.iter_mut().zip(&src.data) {
        *dst_value += src_value;
    }
    Ok(())
}

pub fn elementwise_binary(op: TokenType, lhs: &SimpleTensor, rhs: &SimpleTensor) -> Result<SimpleTensor, String> {
    if lhs.shape != rhs.shape {
        return Err("tensor shape mismatch".to_string());
    }
    if lhs.data.len() != rhs.data.len() {
        return Err("tensor data length mismatch".to_string());
    }
    let data = lhs
        .data
        .iter()
        .zip(&rhs.data)
        .map(|(left, right)| apply_binary_scalar(op, *left, *right))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SimpleTensor {
        shape: lhs.shape.clone(),
        data,
        dtype: lhs.dtype.clone(),
    })
}

pub fn tensor_scalar_binary(op: TokenType, lhs: &SimpleTensor, rhs: f64) -> Result<SimpleTensor, String> {
    let rhs = rhs as f32;
    let data = lhs
        .data
        .iter()
        .map(|left| apply_binary_scalar(op, *left, rhs))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SimpleTensor {
        shape: lhs.shape.clone(),
        data,
        dtype: lhs.dtype.clone(),
    })
}

pub fn scalar_tensor_binary(op: TokenType, lhs: f64, rhs: &SimpleTensor) -> Result<SimpleTensor, String> {
    let lhs = lhs as f32;
    let data = rhs
        .data
        .iter()
        .map(|right| apply_binary_scalar(op, lhs, *right))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SimpleTensor {
        shape: rhs.shape.clone(),
        data,
        dtype: rhs.dtype.clone(),
    })
}

fn apply_binary_scalar(op: TokenType, lhs: f32, rhs: f32) -> Result<f32, String> {
    match op {
        TokenType::Plus => Ok(lhs + rhs),
        TokenType::Minus => Ok(lhs - rhs),
        TokenType::Star => Ok(lhs * rhs),
        TokenType::Slash => Ok(lhs / rhs),
        _ => Err(format!("unsupported tensor binary op {}", op.as_str())),
    }
}

pub fn matmul(lhs: &SimpleTensor, rhs: &SimpleTensor) -> Result<SimpleTensor, String> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err("matmul currently requires rank-2 tensors".to_string());
    }
    let m = lhs.shape[0] as usize;
    let k = lhs.shape[1] as usize;
    let rhs_k = rhs.shape[0] as usize;
    let n = rhs.shape[1] as usize;
    if k != rhs_k {
        return Err(format!(
            "matmul inner dimension mismatch: lhs={:?} rhs={:?}",
            lhs.shape, rhs.shape
        ));
    }

    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += lhs.data[row * k + inner] * rhs.data[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }

    Ok(SimpleTensor {
        shape: vec![m as i64, n as i64],
        data: out,
        dtype: lhs.dtype.clone(),
    })
}

pub fn transpose_2d(tensor: &SimpleTensor) -> Result<SimpleTensor, String> {
    if tensor.shape.len() != 2 {
        return Err("transpose_2d currently requires rank-2 tensors".to_string());
    }
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    let mut out = vec![0.0f32; tensor.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = tensor.data[row * cols + col];
        }
    }
    Ok(SimpleTensor {
        shape: vec![cols as i64, rows as i64],
        data: out,
        dtype: tensor.dtype.clone(),
    })
}

pub fn apply_relu(tensor: &SimpleTensor) -> SimpleTensor {
    SimpleTensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| value.max(0.0)).collect(),
        dtype: tensor.dtype.clone(),
    }
}

pub fn apply_silu(tensor: &SimpleTensor) -> SimpleTensor {
    SimpleTensor {
        shape: tensor.shape.clone(),
        data: tensor
            .data
            .iter()
            .map(|value| *value / (1.0 + (-*value).exp()))
            .collect(),
        dtype: tensor.dtype.clone(),
    }
}

pub fn apply_softmax(tensor: &SimpleTensor) -> Result<SimpleTensor, String> {
    if tensor.shape.is_empty() {
        return Err("softmax requires a tensor with at least one dimension".to_string());
    }
    let axis = *tensor.shape.last().unwrap() as usize;
    if axis == 0 || tensor.data.len() % axis != 0 {
        return Err("softmax requires the last dimension to divide the data length".to_string());
    }
    let rows = tensor.data.len() / axis;
    let mut out = vec![0.0f32; tensor.data.len()];
    for row in 0..rows {
        let start = row * axis;
        let slice = &tensor.data[start..start + axis];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = slice.iter().map(|value| (*value - max_value).exp()).collect::<Vec<_>>();
        let sum = exps.iter().sum::<f32>();
        for (index, value) in exps.into_iter().enumerate() {
            out[start + index] = value / sum.max(f32::EPSILON);
        }
    }
    Ok(SimpleTensor {
        shape: tensor.shape.clone(),
        data: out,
        dtype: tensor.dtype.clone(),
    })
}
