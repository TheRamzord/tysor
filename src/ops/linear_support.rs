use crate::compiler::frontend_ir::{FeExpr, FeFunction, FeType};

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LinearInstance {
    pub binding_name: String,
    pub weight_param_name: String,
    pub bias_param_name: Option<String>,
    pub input_type: FeType,
    pub output_type: FeType,
}

pub fn is_linear_layer_ctor(_expr: &FeExpr) -> bool {
    false
}

pub fn collect_linear_instances(_function: &FeFunction) -> BTreeMap<String, LinearInstance> {
    BTreeMap::new()
}
