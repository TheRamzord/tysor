use crate::compiler::frontend_ir::LoweredModule;
use crate::runtime::interpreter::RuntimeRunOptions;
use crate::training::backward::BackwardRunOptions;
use crate::training::backward::run_backward_module;
use crate::training::executor::TrainRunOptions;
use crate::training::executor::run_train_module;

pub fn run_local_forward_module(lowered: &LoweredModule, options: &RuntimeRunOptions) -> Result<(), String> {
    crate::runtime::interpreter::run_lowered_module(lowered, options)
}

pub fn run_local_backward_module(
    lowered: &LoweredModule,
    options: &BackwardRunOptions,
) -> Result<(), String> {
    run_backward_module(lowered, options)
}

pub fn run_local_train_module(lowered: &LoweredModule, options: &TrainRunOptions) -> Result<(), String> {
    run_train_module(lowered, options)
}
