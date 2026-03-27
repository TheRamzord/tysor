use crate::compiler::frontend_ir::{FeType, FeValue};
use crate::compiler::lexer::TokenType;

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphValue {
    pub id: usize,
    pub name: String,
    pub ty: FeType,
    pub is_parameter: bool,
    pub requires_grad: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphNodeKind {
    #[default]
    Constant,
    Binary,
    PrimitiveCall,
    LibraryCall,
    LibraryCtor,
    Apply,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    pub kind: GraphNodeKind,
    pub output: usize,
    pub op: String,
    pub binary_op: TokenType,
    pub constant: FeValue,
    pub inputs: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphFunction {
    pub name: String,
    pub is_layer: bool,
    pub return_type: FeType,
    pub values: Vec<GraphValue>,
    pub nodes: Vec<GraphNode>,
    pub outputs: Vec<usize>,
    pub named_values: BTreeMap<String, usize>,
}
