use crate::compiler::frontend_ir::LoweredModule;
use crate::runtime::graph_executor::GraphExecutionResult;
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::runtime::tensor::SimpleTensor;
use crate::training::executor::TrainRunOptions;

use std::collections::BTreeMap;

fn unsupported_cuda_message() -> String {
    "CUDA backend is scaffolded but not available on this host/build. Validate and implement it on a Linux/NVIDIA CUDA environment.".to_string()
}

pub fn execute_cuda_module(
    _lowered: &LoweredModule,
    _options: &RuntimeRunOptions,
    _parameters: Option<&BTreeMap<String, SimpleTensor>>,
) -> Result<GraphExecutionResult, String> {
    Err(unsupported_cuda_message())
}

pub fn run_cuda_module(_lowered: &LoweredModule, _options: &RuntimeRunOptions) -> Result<(), String> {
    Err(unsupported_cuda_message())
}

pub fn run_cuda_train_module(_lowered: &LoweredModule, _options: &TrainRunOptions) -> Result<(), String> {
    Err(unsupported_cuda_message())
}
