use crate::backend::core::execution_plan::ExecutionPlan;
use crate::backend::core::kind::BackendKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaCodegenResult {
    pub source: String,
}

pub fn generate_cuda_code(plan: &ExecutionPlan) -> Result<CudaCodegenResult, String> {
    if plan.backend != BackendKind::Cuda {
        return Err("CUDA codegen requires a CUDA execution plan".to_string());
    }

    Ok(CudaCodegenResult {
        source: "// CUDA backend scaffold\n// Code generation is not implemented yet.\n".to_string(),
    })
}
