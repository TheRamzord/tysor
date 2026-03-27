//! Graph builder for lowered frontend functions.
//!
//! This is the bridge between normalized frontend IR and backend planning. It accepts
//! the straight-line subset used by the current runtime paths.

use crate::compiler::frontend_ir::{
    FeExpr, FeExprKind, FeFunction, FeStmtKind, FeType, FeValue,
};
use crate::ir::graph::{GraphFunction, GraphNode, GraphNodeKind, GraphValue};
use crate::ops::library::is_primitive_tensor_op;

fn append_value(graph: &mut GraphFunction, name: String, ty: FeType, is_parameter: bool) -> usize {
    let id = graph.values.len();
    let value = GraphValue {
        id,
        name: name.clone(),
        ty: ty.clone(),
        is_parameter,
        requires_grad: is_parameter && ty.kind == crate::compiler::frontend_ir::FeTypeKind::Tensor,
    };
    graph.values.push(value);
    if !name.is_empty() {
        graph.named_values.insert(name, id);
    }
    id
}

pub struct GraphBuilder<'a> {
    function: &'a FeFunction,
}

impl<'a> GraphBuilder<'a> {
    pub fn new(function: &'a FeFunction) -> Self {
        Self { function }
    }

    pub fn build(&self) -> Result<GraphFunction, String> {
        let mut graph = GraphFunction {
            name: self.function.name.clone(),
            is_layer: self.function.is_layer,
            return_type: self.function.return_type.clone(),
            ..GraphFunction::default()
        };

        for (name, ty) in &self.function.params {
            append_value(&mut graph, name.clone(), ty.clone(), true);
        }

        // The current graph IR is expression-oriented and assumes straight-line code.
        for stmt in &self.function.body {
            match &stmt.kind {
                FeStmtKind::VarDecl { name, value, .. } => {
                    let value_id = self.lower_expr(
                        value.as_ref().ok_or_else(|| "Graph builder expected value".to_string())?,
                        &mut graph,
                    )?;
                    graph.named_values.insert(name.clone(), value_id);
                    if let Some(stored) = graph.values.get_mut(value_id) {
                        stored.name = name.clone();
                    }
                }
                FeStmtKind::Assign { name, value } => {
                    let value_id = self.lower_expr(value, &mut graph)?;
                    graph.named_values.insert(name.clone(), value_id);
                    if let Some(stored) = graph.values.get_mut(value_id) {
                        stored.name = name.clone();
                    }
                }
                FeStmtKind::Return { value } => {
                    let output_id = self.lower_expr(value, &mut graph)?;
                    graph.outputs.push(output_id);
                }
                _ => {
                    return Err("Graph builder currently supports straight-line statements only".to_string());
                }
            }
        }

        Ok(graph)
    }

    fn lookup_named_value(&self, name: &str, graph: &GraphFunction) -> Result<usize, String> {
        graph
            .named_values
            .get(name)
            .copied()
            .ok_or_else(|| format!("Graph builder could not resolve symbol '{name}'"))
    }

    fn lower_expr(&self, expr: &FeExpr, graph: &mut GraphFunction) -> Result<usize, String> {
        match &expr.kind {
            FeExprKind::Constant(value) => {
                let output = append_value(graph, String::new(), expr.ty.clone(), false);
                graph.nodes.push(GraphNode {
                    kind: GraphNodeKind::Constant,
                    output,
                    op: String::new(),
                    binary_op: crate::compiler::lexer::TokenType::Plus,
                    constant: value.clone(),
                    inputs: Vec::new(),
                });
                Ok(output)
            }
            FeExprKind::Var(symbol) => self.lookup_named_value(symbol, graph),
            FeExprKind::Binary { lhs, rhs, op } => {
                let lhs_id = self.lower_expr(lhs, graph)?;
                let rhs_id = self.lower_expr(rhs, graph)?;
                let output = append_value(graph, String::new(), expr.ty.clone(), false);
                graph.nodes.push(GraphNode {
                    kind: GraphNodeKind::Binary,
                    output,
                    op: String::new(),
                    binary_op: *op,
                    constant: FeValue::None,
                    inputs: vec![lhs_id, rhs_id],
                });
                Ok(output)
            }
            FeExprKind::Call { callee, args } => {
                let mut inputs = Vec::new();
                for arg in args {
                    inputs.push(self.lower_expr(&arg.value, graph)?);
                }
                let output = append_value(graph, String::new(), expr.ty.clone(), false);
                graph.nodes.push(GraphNode {
                    // Primitive tensor ops get their own node kind because backends often
                    // lower them differently from library-level calls and constructors.
                    kind: if is_primitive_tensor_op(callee) {
                        GraphNodeKind::PrimitiveCall
                    } else {
                        GraphNodeKind::LibraryCall
                    },
                    output,
                    op: callee.clone(),
                    binary_op: crate::compiler::lexer::TokenType::Plus,
                    constant: FeValue::None,
                    inputs,
                });
                Ok(output)
            }
            FeExprKind::LayerCtor { callee, args } => {
                let mut inputs = Vec::new();
                for arg in args {
                    inputs.push(self.lower_expr(&arg.value, graph)?);
                }
                let output = append_value(graph, String::new(), expr.ty.clone(), false);
                graph.nodes.push(GraphNode {
                    kind: GraphNodeKind::LibraryCtor,
                    output,
                    op: callee.clone(),
                    binary_op: crate::compiler::lexer::TokenType::Plus,
                    constant: FeValue::None,
                    inputs,
                });
                Ok(output)
            }
            FeExprKind::Apply { callee, args } => {
                let mut inputs = Vec::new();
                inputs.push(self.lower_expr(callee, graph)?);
                for arg in args {
                    inputs.push(self.lower_expr(&arg.value, graph)?);
                }
                let output = append_value(graph, String::new(), expr.ty.clone(), false);
                graph.nodes.push(GraphNode {
                    kind: GraphNodeKind::Apply,
                    output,
                    op: String::new(),
                    binary_op: crate::compiler::lexer::TokenType::Plus,
                    constant: FeValue::None,
                    inputs,
                });
                Ok(output)
            }
            _ => Err("Graph builder does not support this FE expression kind yet".to_string()),
        }
    }
}

pub fn build_graph_function(function: &FeFunction) -> Result<GraphFunction, String> {
    GraphBuilder::new(function).build()
}
