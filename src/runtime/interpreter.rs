use crate::compiler::frontend_ir::LoweredModule;
use crate::runtime::graph_executor::{execute_execution_plan, GraphExecutorOptions, GraphRuntimeValue};
use crate::backend::core::execution_plan::compile_function_execution_plan;
use crate::backend::core::kind::BackendKind;
use crate::runtime::tensor::print_tensor;

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRunOptions {
    pub entry: String,
    pub tensor_shapes: BTreeMap<String, Vec<i64>>,
}

impl Default for RuntimeRunOptions {
    fn default() -> Self {
        Self {
            entry: "model".to_string(),
            tensor_shapes: BTreeMap::new(),
        }
    }
}

pub fn run_lowered_module(lowered: &LoweredModule, options: &RuntimeRunOptions) -> Result<(), String> {
    let entry = lowered
        .functions
        .iter()
        .find(|function| function.name == options.entry)
        .ok_or_else(|| format!("Entry function '{}' not found in lowered module", options.entry))?;

    let plan = compile_function_execution_plan(entry, BackendKind::Local)?;
    let execution = execute_execution_plan(
        &plan,
        &GraphExecutorOptions {
            tensor_shapes: options.tensor_shapes.clone(),
        },
    )?;

    if plan.outputs.is_empty() {
        return Err("Entry function did not return a value".to_string());
    }
    if plan.outputs.len() != 1 {
        return Err("Runtime interpreter currently supports a single return value".to_string());
    }
    let output = execution
        .outputs
        .get(&plan.outputs[0])
        .ok_or_else(|| "Runtime interpreter could not resolve the graph output value".to_string())?;
    if let GraphRuntimeValue::Tensor(tensor) = output {
        print_tensor(tensor);
        return Ok(());
    }
    Err("Runtime interpreter only supports tensor returns".to_string())
}
