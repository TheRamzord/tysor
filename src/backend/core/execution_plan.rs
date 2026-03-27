//! Backend-oriented execution planning.
//!
//! A graph function is backend-agnostic. An execution plan adds placement and
//! scheduling information so a runtime/backend knows which steps to execute.

use crate::backend::core::kind::BackendKind;
use crate::compiler::frontend_ir::{FeType, FeValue};
use crate::compiler::lexer::TokenType;
use crate::ir::builder::build_graph_function;
use crate::ir::graph::{GraphFunction, GraphNodeKind, GraphValue};

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    #[default]
    Host,
    Device,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanValue {
    pub id: usize,
    pub name: String,
    pub ty: FeType,
    pub is_parameter: bool,
    pub requires_grad: bool,
    pub placement: Placement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanOpKind {
    #[default]
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanOp {
    pub kind: PlanOpKind,
    pub output: usize,
    pub op: String,
    pub binary_op: TokenType,
    pub constant: FeValue,
    pub inputs: Vec<usize>,
    pub backend: BackendKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanStepKind {
    #[default]
    AllocateHostValue,
    AllocateDeviceValue,
    ExecuteOp,
    MaterializeOutput,
    UploadToDevice,
    DispatchDeviceOp,
    DownloadToHost,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlanStep {
    pub kind: PlanStepKind,
    pub value_id: usize,
    pub op_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExecutionPlan {
    // `values` and `ops` describe dataflow; `steps` is the scheduled order the
    // runtime actually walks.
    pub backend: BackendKind,
    pub name: String,
    pub return_type: FeType,
    pub values: Vec<PlanValue>,
    pub ops: Vec<PlanOp>,
    pub steps: Vec<PlanStep>,
    pub outputs: Vec<usize>,
    pub named_values: BTreeMap<String, usize>,
}

fn lower_plan_kind(kind: GraphNodeKind) -> PlanOpKind {
    match kind {
        GraphNodeKind::Constant => PlanOpKind::Constant,
        GraphNodeKind::Binary => PlanOpKind::Binary,
        GraphNodeKind::PrimitiveCall => PlanOpKind::PrimitiveCall,
        GraphNodeKind::LibraryCall => PlanOpKind::LibraryCall,
        GraphNodeKind::LibraryCtor => PlanOpKind::LibraryCtor,
        GraphNodeKind::Apply => PlanOpKind::Apply,
    }
}

fn lower_plan_value(value: &GraphValue, placement: Placement) -> PlanValue {
    PlanValue {
        id: value.id,
        name: value.name.clone(),
        ty: value.ty.clone(),
        is_parameter: value.is_parameter,
        requires_grad: value.requires_grad,
        placement,
    }
}

fn make_base_plan(graph: &GraphFunction, backend: BackendKind) -> ExecutionPlan {
    ExecutionPlan {
        backend,
        name: graph.name.clone(),
        return_type: graph.return_type.clone(),
        outputs: graph.outputs.clone(),
        named_values: graph.named_values.clone(),
        ..ExecutionPlan::default()
    }
}

fn append_local_allocate_steps(plan: &mut ExecutionPlan) {
    for value in &plan.values {
        plan.steps.push(PlanStep {
            kind: PlanStepKind::AllocateHostValue,
            value_id: value.id,
            op_index: None,
        });
    }
}

pub fn compile_local_execution_plan(graph: &GraphFunction) -> ExecutionPlan {
    let mut plan = make_base_plan(graph, BackendKind::Local);
    for value in &graph.values {
        plan.values.push(lower_plan_value(value, Placement::Host));
    }
    append_local_allocate_steps(&mut plan);

    // Local execution is simple: keep everything on host and execute graph ops in order.
    for node in &graph.nodes {
        plan.ops.push(PlanOp {
            kind: lower_plan_kind(node.kind),
            output: node.output,
            op: node.op.clone(),
            binary_op: node.binary_op,
            constant: node.constant.clone(),
            inputs: node.inputs.clone(),
            backend: BackendKind::Local,
        });
        plan.steps.push(PlanStep {
            kind: PlanStepKind::ExecuteOp,
            value_id: node.output,
            op_index: Some(plan.ops.len() - 1),
        });
    }

    for output in &plan.outputs {
        plan.steps.push(PlanStep {
            kind: PlanStepKind::MaterializeOutput,
            value_id: *output,
            op_index: None,
        });
    }
    plan
}

pub fn compile_metal_execution_plan(graph: &GraphFunction) -> ExecutionPlan {
    let mut plan = make_base_plan(graph, BackendKind::Metal);
    // Metal plans make tensor placement explicit so both the native and simulated
    // Metal runtimes can consume the same plan format.
    for value in &graph.values {
        let placement = if value.ty.kind == crate::compiler::frontend_ir::FeTypeKind::Tensor && !value.is_parameter {
            Placement::Device
        } else {
            Placement::Host
        };
        plan.values.push(lower_plan_value(value, placement));
    }

    for value in &plan.values {
        match value.placement {
            Placement::Host => {
                plan.steps.push(PlanStep {
                    kind: PlanStepKind::AllocateHostValue,
                    value_id: value.id,
                    op_index: None,
                });
                if value.is_parameter {
                    plan.steps.push(PlanStep {
                        kind: PlanStepKind::UploadToDevice,
                        value_id: value.id,
                        op_index: None,
                    });
                }
            }
            Placement::Device => {
                plan.steps.push(PlanStep {
                    kind: PlanStepKind::AllocateDeviceValue,
                    value_id: value.id,
                    op_index: None,
                });
            }
        }
    }

    for node in &graph.nodes {
        plan.ops.push(PlanOp {
            kind: lower_plan_kind(node.kind),
            output: node.output,
            op: node.op.clone(),
            binary_op: node.binary_op,
            constant: node.constant.clone(),
            inputs: node.inputs.clone(),
            backend: BackendKind::Metal,
        });
        plan.steps.push(PlanStep {
            kind: PlanStepKind::DispatchDeviceOp,
            value_id: node.output,
            op_index: Some(plan.ops.len() - 1),
        });
    }

    for output in &plan.outputs {
        plan.steps.push(PlanStep {
            kind: PlanStepKind::DownloadToHost,
            value_id: *output,
            op_index: None,
        });
        plan.steps.push(PlanStep {
            kind: PlanStepKind::MaterializeOutput,
            value_id: *output,
            op_index: None,
        });
    }
    plan
}

pub fn compile_cuda_execution_plan(graph: &GraphFunction) -> ExecutionPlan {
    let mut plan = make_base_plan(graph, BackendKind::Cuda);
    for value in &graph.values {
        let placement = if value.ty.kind == crate::compiler::frontend_ir::FeTypeKind::Tensor && !value.is_parameter {
            Placement::Device
        } else {
            Placement::Host
        };
        plan.values.push(lower_plan_value(value, placement));
    }

    for value in &plan.values {
        match value.placement {
            Placement::Host => {
                plan.steps.push(PlanStep {
                    kind: PlanStepKind::AllocateHostValue,
                    value_id: value.id,
                    op_index: None,
                });
                if value.is_parameter {
                    plan.steps.push(PlanStep {
                        kind: PlanStepKind::UploadToDevice,
                        value_id: value.id,
                        op_index: None,
                    });
                }
            }
            Placement::Device => {
                plan.steps.push(PlanStep {
                    kind: PlanStepKind::AllocateDeviceValue,
                    value_id: value.id,
                    op_index: None,
                });
            }
        }
    }

    for node in &graph.nodes {
        plan.ops.push(PlanOp {
            kind: lower_plan_kind(node.kind),
            output: node.output,
            op: node.op.clone(),
            binary_op: node.binary_op,
            constant: node.constant.clone(),
            inputs: node.inputs.clone(),
            backend: BackendKind::Cuda,
        });
        plan.steps.push(PlanStep {
            kind: PlanStepKind::DispatchDeviceOp,
            value_id: node.output,
            op_index: Some(plan.ops.len() - 1),
        });
    }

    for output in &plan.outputs {
        plan.steps.push(PlanStep {
            kind: PlanStepKind::DownloadToHost,
            value_id: *output,
            op_index: None,
        });
        plan.steps.push(PlanStep {
            kind: PlanStepKind::MaterializeOutput,
            value_id: *output,
            op_index: None,
        });
    }
    plan
}

pub fn compile_execution_plan(graph: &GraphFunction, backend: BackendKind) -> ExecutionPlan {
    match backend {
        BackendKind::Local => compile_local_execution_plan(graph),
        BackendKind::Cuda => compile_cuda_execution_plan(graph),
        BackendKind::Metal => compile_metal_execution_plan(graph),
        BackendKind::PyTorch => compile_local_execution_plan(graph),
    }
}

pub fn compile_function_execution_plan(
    function: &crate::compiler::frontend_ir::FeFunction,
    backend: BackendKind,
) -> Result<ExecutionPlan, String> {
    let graph = build_graph_function(function)?;
    Ok(compile_execution_plan(&graph, backend))
}
