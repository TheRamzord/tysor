//! Frontend lowering from AST into a normalized IR.
//!
//! The parser keeps source-like structure. This module reduces that into a smaller,
//! runtime-oriented representation that later graph and backend stages can consume.

use crate::compiler::builtins::registry::all_builtin_signatures;
use crate::compiler::builtins::type_rules::infer_fe_call_result_type;
use crate::compiler::lexer::TokenType;
use crate::compiler::parser::{
    Arg, Config, Expr, ExprKind, Field, Function, Layer, Program, SourceSpan, Stmt, StmtKind,
    Train, Type, VarDecl,
};
use crate::ops::model::is_layer_constructor;

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeTypeKind {
    Int,
    Float,
    Bool,
    Tensor,
    Tuple,
    Callable,
    Void,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeType {
    // FE types preserve tensor annotations and callable returns because later stages
    // still need them for graph construction and backend planning.
    pub kind: FeTypeKind,
    pub elements: Vec<FeType>,
    pub callable_return: Option<Box<FeType>>,
    pub scalar_dtype: Option<String>,
    pub tensor_dtype: Option<String>,
    pub tensor_shape_expr: Option<String>,
    pub tensor_rank: Option<usize>,
}

impl Default for FeType {
    fn default() -> Self {
        Self::void()
    }
}

impl FeType {
    pub fn int() -> Self {
        Self {
            kind: FeTypeKind::Int,
            scalar_dtype: None,
            ..Self::void()
        }
    }

    pub fn int16() -> Self {
        Self {
            kind: FeTypeKind::Int,
            scalar_dtype: Some("int16".to_string()),
            ..Self::void()
        }
    }

    pub fn int32() -> Self {
        Self {
            kind: FeTypeKind::Int,
            scalar_dtype: Some("int32".to_string()),
            ..Self::void()
        }
    }

    pub fn int64() -> Self {
        Self {
            kind: FeTypeKind::Int,
            scalar_dtype: Some("int64".to_string()),
            ..Self::void()
        }
    }

    pub fn float() -> Self {
        Self {
            kind: FeTypeKind::Float,
            scalar_dtype: None,
            ..Self::void()
        }
    }

    pub fn float16() -> Self {
        Self {
            kind: FeTypeKind::Float,
            scalar_dtype: Some("float16".to_string()),
            ..Self::void()
        }
    }

    pub fn float32() -> Self {
        Self {
            kind: FeTypeKind::Float,
            scalar_dtype: Some("float32".to_string()),
            ..Self::void()
        }
    }

    pub fn float64() -> Self {
        Self {
            kind: FeTypeKind::Float,
            scalar_dtype: Some("float64".to_string()),
            ..Self::void()
        }
    }

    pub fn bool() -> Self {
        Self {
            kind: FeTypeKind::Bool,
            ..Self::void()
        }
    }

    pub fn tensor(dtype: Option<String>, shape_expr: Option<String>, rank: Option<usize>) -> Self {
        Self {
            kind: FeTypeKind::Tensor,
            tensor_dtype: dtype,
            tensor_shape_expr: shape_expr,
            tensor_rank: rank,
            ..Self::void()
        }
    }

    pub fn void() -> Self {
        Self {
            kind: FeTypeKind::Void,
            elements: Vec::new(),
            callable_return: None,
            scalar_dtype: None,
            tensor_dtype: None,
            tensor_shape_expr: None,
            tensor_rank: None,
        }
    }

    pub fn none() -> Self {
        Self {
            kind: FeTypeKind::None,
            ..Self::void()
        }
    }

    pub fn tuple(elements: Vec<FeType>) -> Self {
        Self {
            kind: FeTypeKind::Tuple,
            elements,
            ..Self::void()
        }
    }

    pub fn callable(return_type: FeType) -> Self {
        Self {
            kind: FeTypeKind::Callable,
            callable_return: Some(Box::new(return_type)),
            ..Self::void()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeValue {
    None,
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

impl Default for FeValue {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeCallArg {
    pub name: Option<String>,
    pub value: FeExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeExprKind {
    Constant(FeValue),
    Var(String),
    Call {
        callee: String,
        args: Vec<FeCallArg>,
    },
    LayerCtor {
        callee: String,
        args: Vec<FeCallArg>,
    },
    Apply {
        callee: Box<FeExpr>,
        args: Vec<FeCallArg>,
    },
    Tuple(Vec<FeExpr>),
    Binary {
        op: TokenType,
        lhs: Box<FeExpr>,
        rhs: Box<FeExpr>,
    },
    IfThenElse {
        condition: Box<FeExpr>,
        then_expr: Box<FeExpr>,
        else_expr: Box<FeExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeExpr {
    pub ty: FeType,
    pub kind: FeExprKind,
}

impl FeExpr {
    pub fn constant(value: FeValue, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Constant(value),
        }
    }

    pub fn var(symbol: String, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Var(symbol),
        }
    }

    pub fn call(callee: String, args: Vec<FeCallArg>, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Call { callee, args },
        }
    }

    pub fn layer_ctor(callee: String, args: Vec<FeCallArg>, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::LayerCtor { callee, args },
        }
    }

    pub fn apply(callee: FeExpr, args: Vec<FeCallArg>, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Apply {
                callee: Box::new(callee),
                args,
            },
        }
    }

    pub fn tuple(elements: Vec<FeExpr>, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Tuple(elements),
        }
    }

    pub fn binary(op: TokenType, lhs: FeExpr, rhs: FeExpr, ty: FeType) -> Self {
        Self {
            ty,
            kind: FeExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
        }
    }

    pub fn if_then_else(
        condition: FeExpr,
        then_expr: FeExpr,
        else_expr: FeExpr,
        ty: FeType,
    ) -> Self {
        Self {
            ty,
            kind: FeExprKind::IfThenElse {
                condition: Box::new(condition),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeStmtKind {
    VarDecl {
        name: String,
        ty: FeType,
        value: Option<FeExpr>,
    },
    Assign {
        name: String,
        value: FeExpr,
    },
    Return {
        value: FeExpr,
    },
    Expr {
        value: FeExpr,
    },
    If {
        condition: FeExpr,
        then_body: Vec<FeStmt>,
        elif_bodies: Vec<(FeExpr, Vec<FeStmt>)>,
        else_body: Vec<FeStmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeStmt {
    pub kind: FeStmtKind,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeFunction {
    pub name: String,
    pub is_layer: bool,
    pub return_type: FeType,
    pub params: Vec<(String, FeType)>,
    pub named_outputs: Vec<(String, FeType)>,
    pub body: Vec<FeStmt>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeConfig {
    pub name: String,
    pub fields: BTreeMap<String, FeValue>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeTrain {
    pub name: String,
    pub backends: Vec<FeValue>,
    pub optimizers: Vec<FeValue>,
    pub learning_rates: Vec<FeValue>,
    pub objective_symbols: Vec<String>,
    pub iterations: Vec<FeValue>,
    pub variant_count: usize,
    pub extra_properties: BTreeMap<String, FeValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObjectiveSource {
    Param,
    Output,
    Local,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeExecutionRun {
    pub run_name: String,
    pub model_name: String,
    pub train_name: String,
    pub backend: Option<FeValue>,
    pub optimizer: Option<FeValue>,
    pub learning_rate: Option<FeValue>,
    pub objective_symbol: Option<String>,
    pub objective_source: ObjectiveSource,
    pub objective_type: FeType,
    pub iteration: Option<FeValue>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeExecutionPlan {
    pub model_entry: String,
    pub runs: Vec<FeExecutionRun>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LoweredModule {
    pub configs: Vec<FeConfig>,
    pub trains: Vec<FeTrain>,
    pub functions: Vec<FeFunction>,
    pub globals: Vec<FeStmt>,
    pub execution_plan: Option<FeExecutionPlan>,
}

#[derive(Debug, Clone, Default)]
struct EvaluatedConfigField {
    in_progress: bool,
    computed: bool,
    value: FeValue,
}

pub struct FrontendLowerer<'a> {
    program: &'a Program,
    config_defs: BTreeMap<String, &'a Config>,
    global_symbols: BTreeMap<String, FeType>,
    config_field_cache: BTreeMap<String, BTreeMap<String, EvaluatedConfigField>>,
    current_symbols: BTreeMap<String, FeType>,
}

impl<'a> FrontendLowerer<'a> {
    pub fn new(program: &'a Program) -> Self {
        let mut config_defs = BTreeMap::new();
        let mut config_field_cache = BTreeMap::new();
        for config in &program.configs {
            config_defs.insert(config.name.clone(), config);
            let mut fields = BTreeMap::new();
            for field in &config.fields {
                fields.insert(field.name.clone(), EvaluatedConfigField::default());
            }
            config_field_cache.insert(config.name.clone(), fields);
        }

        let mut global_symbols = BTreeMap::new();
        for builtin in all_builtin_signatures() {
            global_symbols.insert(builtin.name, lower_type(&builtin.return_type));
        }
        for layer in &program.layers {
            global_symbols.insert(layer.name.clone(), lower_type(&layer.return_type));
        }
        for function in &program.functions {
            global_symbols.insert(function.name.clone(), lower_type(&function.return_type));
        }

        Self {
            program,
            config_defs,
            global_symbols,
            config_field_cache,
            current_symbols: BTreeMap::new(),
        }
    }

    pub fn lower(&mut self) -> Result<LoweredModule, String> {
        let mut module = LoweredModule::default();
        for config in &self.program.configs {
            module.configs.push(self.lower_config(config)?);
        }
        for train in &self.program.trains {
            module.trains.push(self.lower_train(train)?);
        }
        self.current_symbols.clear();
        for stmt in &self.program.globals {
            module.globals.push(self.lower_stmt(stmt)?);
        }
        for layer in &self.program.layers {
            module.functions.push(self.lower_layer(layer)?);
        }
        for function in &self.program.functions {
            module.functions.push(self.lower_function(function)?);
        }
        let has_model_layer = module
            .functions
            .iter()
            .any(|function| function.is_layer && function.name == "model");
        let has_model_train = module.trains.iter().any(|train| train.name == "model");
        if has_model_layer && has_model_train {
            module.execution_plan = Some(self.build_execution_plan(&module)?);
        }
        Ok(module)
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<FeExpr, String> {
        match &expr.kind {
            ExprKind::IntLit(value) => Ok(FeExpr::constant(FeValue::Int(*value), FeType::int())),
            ExprKind::FloatLit(value) => {
                Ok(FeExpr::constant(FeValue::Float(*value), FeType::float()))
            }
            ExprKind::BoolLit(value) => Ok(FeExpr::constant(FeValue::Bool(*value), FeType::bool())),
            ExprKind::StringLit(value) => Ok(FeExpr::constant(
                FeValue::String(value.clone()),
                FeType::void(),
            )),
            ExprKind::Identifier(name) => {
                if name == "None" {
                    return Ok(FeExpr::constant(FeValue::None, FeType::none()));
                }
                if let Some(symbol) = self.find_symbol(name) {
                    return Ok(FeExpr::var(name.clone(), symbol));
                }
                if self.config_defs.contains_key(name) {
                    return Err(format!(
                        "Config object '{}' cannot appear directly in lowered IR",
                        name
                    ));
                }
                Ok(FeExpr::var(name.clone(), FeType::void()))
            }
            ExprKind::Call { callee, args } => {
                let mut lowered_args = Vec::new();
                let mut arg_types = Vec::new();
                for arg in args {
                    let lowered = self.lower_expr(&arg.value)?;
                    arg_types.push(lowered.ty.clone());
                    lowered_args.push(FeCallArg {
                        name: arg.name.clone(),
                        value: lowered,
                    });
                }
                let mut result_type = FeType::void();
                if let Some(symbol) = self.find_symbol(callee) {
                    result_type = if symbol.kind == FeTypeKind::Callable {
                        symbol
                            .callable_return
                            .as_deref()
                            .cloned()
                            .unwrap_or_else(FeType::void)
                    } else {
                        symbol.clone()
                    };
                    result_type = self.infer_call_result_type(callee, &result_type, &arg_types);
                    if symbol.kind == FeTypeKind::Callable {
                        return Ok(FeExpr::apply(
                            FeExpr::var(callee.clone(), symbol),
                            lowered_args,
                            result_type,
                        ));
                    }
                } else if let Some(symbol) = self.global_symbols.get(callee) {
                    result_type = symbol.clone();
                }
                result_type = self.infer_call_result_type(callee, &result_type, &arg_types);
                if is_layer_constructor(callee) && result_type.kind == FeTypeKind::Callable {
                    Ok(FeExpr::layer_ctor(
                        callee.clone(),
                        lowered_args,
                        result_type,
                    ))
                } else {
                    Ok(FeExpr::call(callee.clone(), lowered_args, result_type))
                }
            }
            ExprKind::Binary { lhs, rhs, op } => {
                if *op == TokenType::Dot {
                    if let (ExprKind::Identifier(config_name), ExprKind::Identifier(field_name)) =
                        (&lhs.kind, &rhs.kind)
                    {
                        let value = self.eval_config_field(config_name, field_name, &expr.span)?;
                        let ty = match value {
                            FeValue::Int(_) => FeType::int(),
                            FeValue::Float(_) => FeType::float(),
                            FeValue::Bool(_) => FeType::bool(),
                            _ => FeType::void(),
                        };
                        return Ok(FeExpr::constant(value, ty));
                    }
                    return Err("Only config field access can be lowered from '.'".to_string());
                }
                let lhs = self.lower_expr(lhs)?;
                let rhs = self.lower_expr(rhs)?;
                let ty = if lhs.ty.kind == FeTypeKind::Tensor || rhs.ty.kind == FeTypeKind::Tensor {
                    self.merge_tensor_types(&lhs.ty, &rhs.ty)
                } else if lhs.ty.kind == FeTypeKind::Float || rhs.ty.kind == FeTypeKind::Float {
                    merge_float_types(&lhs.ty, &rhs.ty)
                } else if lhs.ty.kind == FeTypeKind::Int || rhs.ty.kind == FeTypeKind::Int {
                    merge_int_types(&lhs.ty, &rhs.ty)
                } else if matches!(
                    op,
                    TokenType::EqEq
                        | TokenType::Neq
                        | TokenType::Lt
                        | TokenType::Gt
                        | TokenType::LtEq
                        | TokenType::GtEq
                        | TokenType::AmpAmp
                        | TokenType::PipePipe
                ) {
                    FeType::bool()
                } else {
                    FeType::int()
                };
                Ok(FeExpr::binary(*op, lhs, rhs, ty))
            }
            ExprKind::Tuple(elements) => {
                let mut lowered = Vec::new();
                let mut tys = Vec::new();
                for element in elements {
                    let expr = self.lower_expr(element)?;
                    tys.push(expr.ty.clone());
                    lowered.push(expr);
                }
                Ok(FeExpr::tuple(lowered, FeType::tuple(tys)))
            }
            ExprKind::Ternary {
                then_expr,
                condition,
                else_expr,
            } => {
                let cond = self.lower_expr(condition)?;
                let then_expr = self.lower_expr(then_expr)?;
                let else_expr = self.lower_expr(else_expr)?;
                let ty = if then_expr.ty.kind == FeTypeKind::None {
                    else_expr.ty.clone()
                } else {
                    then_expr.ty.clone()
                };
                Ok(FeExpr::if_then_else(cond, then_expr, else_expr, ty))
            }
            ExprKind::Arrow { .. } => self.lower_arrow_expr(expr),
            ExprKind::Unary { operand, op } => {
                let operand = self.lower_expr(operand)?;
                let operand_ty = operand.ty.clone();
                if *op == TokenType::Minus {
                    Ok(FeExpr::binary(
                        TokenType::Minus,
                        FeExpr::constant(FeValue::Int(0), FeType::int()),
                        operand,
                        operand_ty,
                    ))
                } else {
                    Ok(FeExpr::binary(
                        *op,
                        operand,
                        FeExpr::constant(FeValue::Bool(false), FeType::bool()),
                        FeType::bool(),
                    ))
                }
            }
        }
    }

    fn lower_arrow_expr(&mut self, expr: &Expr) -> Result<FeExpr, String> {
        let ExprKind::Arrow { source, stages } = &expr.kind else {
            return self.lower_expr(expr);
        };
        let mut current = self.lower_expr(source)?;
        for stage in stages {
            current = self.lower_arrow_stage_expr(stage, current)?;
        }
        Ok(current)
    }

    fn lower_arrow_call_stage(
        &mut self,
        callee: &str,
        args: &[crate::compiler::parser::CallArgument],
        current: FeExpr,
    ) -> Result<FeExpr, String> {
        let local_symbol = self.find_symbol(callee);
        let mut call_type = local_symbol.clone().unwrap_or_else(|| {
            self.global_symbols
                .get(callee)
                .cloned()
                .unwrap_or_else(FeType::void)
        });

        if call_type.kind == FeTypeKind::Callable {
            if let Some(local) = local_symbol {
                let callee_expr = FeExpr::var(callee.to_string(), local.clone());
                let mut apply_args = Vec::new();
                apply_args.push(FeCallArg {
                    name: None,
                    value: current,
                });
                for arg in args {
                    apply_args.push(FeCallArg {
                        name: arg.name.clone(),
                        value: self.lower_expr(&arg.value)?,
                    });
                }
                let mut result_type = call_type
                    .callable_return
                    .as_deref()
                    .cloned()
                    .unwrap_or_else(|| FeType::tensor(None, None, None));
                if let Some(first) = apply_args.first() {
                    if first.value.ty.kind == FeTypeKind::Tensor
                        && result_type.kind == FeTypeKind::Tensor
                    {
                        result_type.tensor_dtype = first.value.ty.tensor_dtype.clone();
                        result_type.tensor_rank = first.value.ty.tensor_rank;
                    }
                }
                return Ok(FeExpr::apply(callee_expr, apply_args, result_type));
            }
            let callee_expr = self.lower_expr(&Expr {
                span: SourceSpan::default(),
                kind: ExprKind::Call {
                    callee: callee.to_string(),
                    args: args.to_vec(),
                },
            })?;
            let mut apply_args = Vec::new();
            apply_args.push(FeCallArg {
                name: None,
                value: current,
            });
            let mut result_type = call_type
                .callable_return
                .as_deref()
                .cloned()
                .unwrap_or_else(|| FeType::tensor(None, None, None));
            if apply_args[0].value.ty.kind == FeTypeKind::Tensor
                && result_type.kind == FeTypeKind::Tensor
            {
                result_type.tensor_dtype = apply_args[0].value.ty.tensor_dtype.clone();
                result_type.tensor_rank = apply_args[0].value.ty.tensor_rank;
            }
            return Ok(FeExpr::apply(callee_expr, apply_args, result_type));
        }

        let mut lowered_args = Vec::new();
        lowered_args.push(FeCallArg {
            name: None,
            value: current,
        });
        for arg in args {
            lowered_args.push(FeCallArg {
                name: arg.name.clone(),
                value: self.lower_expr(&arg.value)?,
            });
        }
        let mut arg_types = Vec::new();
        for arg in &lowered_args {
            arg_types.push(arg.value.ty.clone());
        }
        if call_type.kind == FeTypeKind::Void {
            call_type = FeType::tensor(None, None, None);
        }
        let result_type = self.infer_call_result_type(callee, &call_type, &arg_types);
        Ok(FeExpr::call(callee.to_string(), lowered_args, result_type))
    }

    fn lower_arrow_stage_expr(&mut self, expr: &Expr, current: FeExpr) -> Result<FeExpr, String> {
        match &expr.kind {
            ExprKind::Call { callee, args } => self.lower_arrow_call_stage(callee, args, current),
            ExprKind::Binary { lhs, rhs, op } if *op == TokenType::StarEq => {
                let ExprKind::Call { callee, args } = &lhs.kind else {
                    return Err("Repeated arrow stage must begin with a call".to_string());
                };
                let repeat_value = self.eval_constant_expr(rhs)?;
                let repeat = self.expect_int(repeat_value, &rhs.span, "arrow repetition count")?;
                let mut value = current;
                for _ in 0..repeat {
                    value = self.lower_arrow_call_stage(callee, args, value)?;
                }
                Ok(value)
            }
            ExprKind::Binary { lhs, rhs, op } => {
                let lhs_sites = count_arrow_stage_sites(lhs);
                let rhs_sites = count_arrow_stage_sites(rhs);
                let lhs_expr = if lhs_sites > 0 {
                    self.lower_arrow_stage_expr(lhs, current.clone())?
                } else {
                    self.lower_expr(lhs)?
                };
                let rhs_expr = if rhs_sites > 0 {
                    self.lower_arrow_stage_expr(rhs, current.clone())?
                } else {
                    self.lower_expr(rhs)?
                };
                let ty = if lhs_expr.ty.kind == FeTypeKind::Tensor
                    || rhs_expr.ty.kind == FeTypeKind::Tensor
                {
                    self.merge_tensor_types(&lhs_expr.ty, &rhs_expr.ty)
                } else if lhs_expr.ty.kind == FeTypeKind::Float
                    || rhs_expr.ty.kind == FeTypeKind::Float
                {
                    merge_float_types(&lhs_expr.ty, &rhs_expr.ty)
                } else if lhs_expr.ty.kind == FeTypeKind::Int
                    || rhs_expr.ty.kind == FeTypeKind::Int
                {
                    merge_int_types(&lhs_expr.ty, &rhs_expr.ty)
                } else if matches!(
                    op,
                    TokenType::EqEq
                        | TokenType::Neq
                        | TokenType::Lt
                        | TokenType::Gt
                        | TokenType::LtEq
                        | TokenType::GtEq
                        | TokenType::AmpAmp
                        | TokenType::PipePipe
                ) {
                    FeType::bool()
                } else {
                    FeType::int()
                };
                Ok(FeExpr::binary(*op, lhs_expr, rhs_expr, ty))
            }
            ExprKind::Unary { operand, op } => {
                let operand = if count_arrow_stage_sites(operand) > 0 {
                    self.lower_arrow_stage_expr(operand, current)?
                } else {
                    self.lower_expr(operand)?
                };
                let operand_ty = operand.ty.clone();
                if *op == TokenType::Minus {
                    Ok(FeExpr::binary(
                        TokenType::Minus,
                        FeExpr::constant(FeValue::Int(0), FeType::int()),
                        operand,
                        operand_ty,
                    ))
                } else {
                    Ok(FeExpr::binary(
                        *op,
                        operand,
                        FeExpr::constant(FeValue::Bool(false), FeType::bool()),
                        FeType::bool(),
                    ))
                }
            }
            ExprKind::Ternary {
                then_expr,
                condition,
                else_expr,
            } => {
                let cond = if count_arrow_stage_sites(condition) > 0 {
                    self.lower_arrow_stage_expr(condition, current.clone())?
                } else {
                    self.lower_expr(condition)?
                };
                let then_expr = if count_arrow_stage_sites(then_expr) > 0 {
                    self.lower_arrow_stage_expr(then_expr, current.clone())?
                } else {
                    self.lower_expr(then_expr)?
                };
                let else_expr = if count_arrow_stage_sites(else_expr) > 0 {
                    self.lower_arrow_stage_expr(else_expr, current)?
                } else {
                    self.lower_expr(else_expr)?
                };
                let ty = if then_expr.ty.kind == FeTypeKind::None {
                    else_expr.ty.clone()
                } else {
                    then_expr.ty.clone()
                };
                Ok(FeExpr::if_then_else(cond, then_expr, else_expr, ty))
            }
            ExprKind::Tuple(elements) => {
                let mut lowered = Vec::new();
                let mut tys = Vec::new();
                for element in elements {
                    let expr = if count_arrow_stage_sites(element) > 0 {
                        self.lower_arrow_stage_expr(element, current.clone())?
                    } else {
                        self.lower_expr(element)?
                    };
                    tys.push(expr.ty.clone());
                    lowered.push(expr);
                }
                Ok(FeExpr::tuple(lowered, FeType::tuple(tys)))
            }
            _ => self.lower_expr(expr),
        }
    }

    fn lower_scope(&mut self, stmt: &Stmt) -> Result<Vec<FeStmt>, String> {
        let StmtKind::Scope(stmts) = &stmt.kind else {
            return Err("Expected scope statement while lowering".to_string());
        };
        let saved = self.current_symbols.clone();
        let mut lowered = Vec::new();
        for stmt in stmts {
            lowered.push(self.lower_stmt(stmt)?);
        }
        self.current_symbols = saved;
        Ok(lowered)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<FeStmt, String> {
        match &stmt.kind {
            StmtKind::Return(expr) => Ok(FeStmt {
                kind: FeStmtKind::Return {
                    value: self.lower_expr(expr)?,
                },
            }),
            StmtKind::Expr(expr) => Ok(FeStmt {
                kind: FeStmtKind::Expr {
                    value: self.lower_expr(expr)?,
                },
            }),
            StmtKind::VarDecl(VarDecl { name, ty, init, .. }) => {
                let fe_type = lower_type(ty);
                self.bind_symbol(name, fe_type.clone());
                Ok(FeStmt {
                    kind: FeStmtKind::VarDecl {
                        name: name.clone(),
                        ty: fe_type,
                        value: init
                            .as_ref()
                            .map(|expr| self.lower_expr(expr))
                            .transpose()?,
                    },
                })
            }
            StmtKind::Assign(crate::compiler::parser::AssignStmt { name, value }) => {
                let lowered_value = self.lower_expr(value)?;
                let value_type = lowered_value.ty.clone();
                let already_bound = self.find_symbol(name).is_some();
                self.bind_symbol(name, value_type.clone());
                if !already_bound {
                    Ok(FeStmt {
                        kind: FeStmtKind::VarDecl {
                            name: name.clone(),
                            ty: value_type,
                            value: Some(lowered_value),
                        },
                    })
                } else {
                    Ok(FeStmt {
                        kind: FeStmtKind::Assign {
                            name: name.clone(),
                            value: lowered_value,
                        },
                    })
                }
            }
            StmtKind::Scope(_) => Err(
                "Nested standalone scope statements are not supported in FE lowering".to_string(),
            ),
            StmtKind::If {
                condition,
                then_stmt,
                elifs,
                else_stmt,
            } => {
                let mut lowered_elifs = Vec::new();
                for branch in elifs {
                    lowered_elifs.push((
                        self.lower_expr(&branch.condition)?,
                        self.lower_scope(&branch.body)?,
                    ));
                }
                let else_body = if let Some(otherwise) = else_stmt {
                    self.lower_scope(otherwise)?
                } else {
                    Vec::new()
                };
                Ok(FeStmt {
                    kind: FeStmtKind::If {
                        condition: self.lower_expr(condition)?,
                        then_body: self.lower_scope(then_stmt)?,
                        elif_bodies: lowered_elifs,
                        else_body,
                    },
                })
            }
        }
    }

    fn lower_function(&mut self, function: &Function) -> Result<FeFunction, String> {
        let mut lowered = FeFunction {
            name: function.name.clone(),
            is_layer: false,
            return_type: lower_type(&function.return_type),
            ..FeFunction::default()
        };
        let saved = self.current_symbols.clone();
        self.current_symbols.clear();
        for Arg { name, ty, .. } in &function.args {
            let fe_type = lower_type(ty);
            lowered.params.push((name.clone(), fe_type.clone()));
            self.bind_symbol(name, fe_type);
        }
        lowered.body = self.lower_scope(&function.body)?;
        let mut callable_symbols = self.current_symbols.clone();
        self.normalize_callable_calls(&mut lowered.body, &mut callable_symbols);
        self.current_symbols = saved;
        Ok(lowered)
    }

    fn lower_layer(&mut self, layer: &Layer) -> Result<FeFunction, String> {
        let mut lowered = FeFunction {
            name: layer.name.clone(),
            is_layer: true,
            return_type: lower_type(&layer.return_type),
            ..FeFunction::default()
        };
        let saved = self.current_symbols.clone();
        self.current_symbols.clear();
        for Arg { name, ty, .. } in &layer.args {
            let fe_type = lower_type(ty);
            lowered.params.push((name.clone(), fe_type.clone()));
            self.bind_symbol(name, fe_type);
        }
        lowered.body = self.lower_scope(&layer.body)?;
        let mut callable_symbols = self.current_symbols.clone();
        self.normalize_callable_calls(&mut lowered.body, &mut callable_symbols);
        for stmt in &lowered.body {
            if let FeStmtKind::Return { value } = &stmt.kind {
                match &value.kind {
                    FeExprKind::Var(symbol) => lowered
                        .named_outputs
                        .push((symbol.clone(), value.ty.clone())),
                    FeExprKind::Tuple(elements) => {
                        for element in elements {
                            if let FeExprKind::Var(symbol) = &element.kind {
                                lowered
                                    .named_outputs
                                    .push((symbol.clone(), element.ty.clone()));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        self.current_symbols = saved;
        Ok(lowered)
    }

    fn lower_config(&mut self, config: &Config) -> Result<FeConfig, String> {
        let mut lowered = FeConfig {
            name: config.name.clone(),
            ..FeConfig::default()
        };
        for Field { name, .. } in &config.fields {
            lowered.fields.insert(
                name.clone(),
                self.eval_config_field(&config.name, name, &config.span)?,
            );
        }
        Ok(lowered)
    }

    fn lower_train(&mut self, train: &Train) -> Result<FeTrain, String> {
        let mut lowered = FeTrain {
            name: train.name.clone(),
            variant_count: 1,
            ..FeTrain::default()
        };

        for field in &train.fields {
            let values = if let Some(init) = &field.init {
                self.eval_constant_field_values(init)?
            } else {
                vec![FeValue::None]
            };
            let assigned = match field.name.as_str() {
                "backend" | "target" | "device" => {
                    lowered.backends = values.clone();
                    true
                }
                "optimizer" => {
                    lowered.optimizers = values.clone();
                    true
                }
                "lr" | "learning_rate" => {
                    lowered.learning_rates = values.clone();
                    true
                }
                "iteration" => {
                    lowered.iterations = values.clone();
                    true
                }
                "objective" => {
                    lowered.objective_symbols.clear();
                    for value in &values {
                        if let FeValue::String(name) = value {
                            lowered.objective_symbols.push(name.clone());
                        } else {
                            return Err(
                                "Train field 'objective' must reference a named tensor root"
                                    .to_string(),
                            );
                        }
                    }
                    true
                }
                _ => false,
            };
            if assigned {
                if values.len() > lowered.variant_count {
                    lowered.variant_count = values.len();
                }
                continue;
            }
            if values.len() != 1 {
                return Err(format!(
                    "Train field '{}' does not support tuple variants; use a scalar value",
                    field.name
                ));
            }
            lowered
                .extra_properties
                .insert(field.name.clone(), values[0].clone());
        }
        Ok(lowered)
    }

    fn build_execution_plan(&self, module: &LoweredModule) -> Result<FeExecutionPlan, String> {
        let model_function = module
            .functions
            .iter()
            .find(|function| function.is_layer && function.name == "model")
            .ok_or_else(|| {
                "Train lowering requires 'layer model' as the model entrypoint".to_string()
            })?;

        let train_model = module
            .trains
            .iter()
            .find(|train| train.name == "model")
            .ok_or_else(|| {
                "Train lowering requires 'train model' as the training entrypoint".to_string()
            })?;

        let mut plan = FeExecutionPlan {
            model_entry: model_function.name.clone(),
            ..FeExecutionPlan::default()
        };

        for i in 0..train_model.variant_count {
            let mut run = FeExecutionRun {
                run_name: if train_model.variant_count == 1 {
                    "model".to_string()
                } else {
                    format!("model_{}", i + 1)
                },
                model_name: plan.model_entry.clone(),
                train_name: train_model.name.clone(),
                backend: pick_broadcast_value(&train_model.backends, i)?,
                optimizer: pick_broadcast_value(&train_model.optimizers, i)?,
                learning_rate: pick_broadcast_value(&train_model.learning_rates, i)?,
                objective_symbol: pick_broadcast_string(&train_model.objective_symbols, i)?,
                iteration: pick_broadcast_value(&train_model.iterations, i)?,
                ..FeExecutionRun::default()
            };

            if let Some(objective_symbol) = run.objective_symbol.clone() {
                let mut matched = false;
                for (name, ty) in &model_function.named_outputs {
                    if name == &objective_symbol {
                        run.objective_source = ObjectiveSource::Output;
                        run.objective_type = ty.clone();
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    for (name, ty) in &model_function.params {
                        if name == &objective_symbol {
                            run.objective_source = ObjectiveSource::Param;
                            run.objective_type = ty.clone();
                            matched = true;
                            break;
                        }
                    }
                }
                if !matched {
                    for stmt in &model_function.body {
                        if resolve_objective_stmt(stmt, &objective_symbol, &mut run) {
                            matched = true;
                            break;
                        }
                    }
                }
                if !matched {
                    return Err(format!(
                        "Execution plan could not resolve objective symbol '{}' against layer model",
                        objective_symbol
                    ));
                }
            }

            plan.runs.push(run);
        }

        Ok(plan)
    }

    fn eval_constant_expr(&mut self, expr: &Expr) -> Result<FeValue, String> {
        match &expr.kind {
            ExprKind::IntLit(value) => Ok(FeValue::Int(*value)),
            ExprKind::FloatLit(value) => Ok(FeValue::Float(*value)),
            ExprKind::BoolLit(value) => Ok(FeValue::Bool(*value)),
            ExprKind::Identifier(name) if name == "None" => Ok(FeValue::None),
            ExprKind::Identifier(name) => Ok(FeValue::String(name.clone())),
            ExprKind::StringLit(value) => Ok(FeValue::String(value.clone())),
            ExprKind::Unary { operand, op } => {
                let operand_value = self.eval_constant_expr(operand)?;
                self.eval_unary(*op, operand_value, &expr.span)
            }
            ExprKind::Binary { lhs, rhs, op } if *op == TokenType::Dot => {
                if let (ExprKind::Identifier(config_name), ExprKind::Identifier(field_name)) =
                    (&lhs.kind, &rhs.kind)
                {
                    self.eval_config_field(config_name, field_name, &expr.span)
                } else {
                    Err("Only config.field is supported in constant evaluation".to_string())
                }
            }
            ExprKind::Binary { lhs, rhs, op } => {
                let lhs_value = self.eval_constant_expr(lhs)?;
                let rhs_value = self.eval_constant_expr(rhs)?;
                self.eval_binary(*op, lhs_value, rhs_value, &expr.span)
            }
            ExprKind::Ternary {
                then_expr,
                condition,
                else_expr,
            } => {
                let condition = as_bool(&self.eval_constant_expr(condition)?)?;
                if condition {
                    self.eval_constant_expr(then_expr)
                } else {
                    self.eval_constant_expr(else_expr)
                }
            }
            _ => Err("Expression is not compile-time constant".to_string()),
        }
    }

    fn eval_constant_field_values(&mut self, expr: &Expr) -> Result<Vec<FeValue>, String> {
        if let ExprKind::Tuple(elements) = &expr.kind {
            let mut values = Vec::new();
            for element in elements {
                values.push(self.eval_constant_expr(element)?);
            }
            Ok(values)
        } else {
            Ok(vec![self.eval_constant_expr(expr)?])
        }
    }

    fn eval_binary(
        &self,
        op: TokenType,
        lhs: FeValue,
        rhs: FeValue,
        span: &SourceSpan,
    ) -> Result<FeValue, String> {
        let result = match op {
            TokenType::Plus => {
                if matches!(lhs, FeValue::Int(_)) && matches!(rhs, FeValue::Int(_)) {
                    FeValue::Int(as_int(&lhs)? + as_int(&rhs)?)
                } else {
                    FeValue::Float(as_double(&lhs)? + as_double(&rhs)?)
                }
            }
            TokenType::Minus => {
                if matches!(lhs, FeValue::Int(_)) && matches!(rhs, FeValue::Int(_)) {
                    FeValue::Int(as_int(&lhs)? - as_int(&rhs)?)
                } else {
                    FeValue::Float(as_double(&lhs)? - as_double(&rhs)?)
                }
            }
            TokenType::Star => {
                if matches!(lhs, FeValue::Int(_)) && matches!(rhs, FeValue::Int(_)) {
                    FeValue::Int(as_int(&lhs)? * as_int(&rhs)?)
                } else {
                    FeValue::Float(as_double(&lhs)? * as_double(&rhs)?)
                }
            }
            TokenType::Slash => FeValue::Float(as_double(&lhs)? / as_double(&rhs)?),
            TokenType::EqEq => {
                if is_none(&lhs) || is_none(&rhs) {
                    FeValue::Bool(is_none(&lhs) && is_none(&rhs))
                } else if matches!(lhs, FeValue::Bool(_)) && matches!(rhs, FeValue::Bool(_)) {
                    FeValue::Bool(as_bool(&lhs)? == as_bool(&rhs)?)
                } else {
                    FeValue::Bool((as_double(&lhs)? - as_double(&rhs)?).abs() < 1e-12)
                }
            }
            TokenType::Neq => {
                let FeValue::Bool(value) = self.eval_binary(TokenType::EqEq, lhs, rhs, span)?
                else {
                    unreachable!()
                };
                FeValue::Bool(!value)
            }
            TokenType::Lt => FeValue::Bool(as_double(&lhs)? < as_double(&rhs)?),
            TokenType::Gt => FeValue::Bool(as_double(&lhs)? > as_double(&rhs)?),
            TokenType::LtEq => FeValue::Bool(as_double(&lhs)? <= as_double(&rhs)?),
            TokenType::GtEq => FeValue::Bool(as_double(&lhs)? >= as_double(&rhs)?),
            TokenType::AmpAmp => FeValue::Bool(as_bool(&lhs)? && as_bool(&rhs)?),
            TokenType::PipePipe => FeValue::Bool(as_bool(&lhs)? || as_bool(&rhs)?),
            _ => {
                return Err(format!(
                    "Constant evaluation failed at {}:{}: unsupported binary operator",
                    span.line, span.column
                ))
            }
        };
        Ok(result)
    }

    fn eval_unary(
        &self,
        op: TokenType,
        operand: FeValue,
        span: &SourceSpan,
    ) -> Result<FeValue, String> {
        match op {
            TokenType::Minus => {
                if matches!(operand, FeValue::Int(_)) {
                    Ok(FeValue::Int(-as_int(&operand)?))
                } else {
                    Ok(FeValue::Float(-as_double(&operand)?))
                }
            }
            TokenType::Bang => Ok(FeValue::Bool(!as_bool(&operand)?)),
            _ => Err(format!(
                "Constant evaluation failed at {}:{}: unsupported unary operator",
                span.line, span.column
            )),
        }
    }

    fn eval_config_field(
        &mut self,
        config_name: &str,
        field_name: &str,
        _span: &SourceSpan,
    ) -> Result<FeValue, String> {
        let config = self
            .config_defs
            .get(config_name)
            .copied()
            .ok_or_else(|| format!("Unknown config '{}'", config_name))?;

        {
            let cache = self
                .config_field_cache
                .get(config_name)
                .and_then(|fields| fields.get(field_name))
                .ok_or_else(|| format!("Unknown config field '{}.{}'", config_name, field_name))?;
            if cache.computed {
                return Ok(cache.value.clone());
            }
            if cache.in_progress {
                return Err(format!(
                    "Cycle detected while evaluating config field '{}.{}'",
                    config_name, field_name
                ));
            }
        }

        let field = config
            .fields
            .iter()
            .find(|candidate| candidate.name == field_name)
            .ok_or_else(|| {
                format!(
                    "Config field '{}.{}' does not exist",
                    config_name, field_name
                )
            })?;
        let Some(init) = &field.init else {
            return Err(format!(
                "Config field '{}.{}' does not have a compile-time value",
                config_name, field_name
            ));
        };

        self.config_field_cache
            .get_mut(config_name)
            .unwrap()
            .get_mut(field_name)
            .unwrap()
            .in_progress = true;
        let value = self.eval_constant_expr(init)?;
        let cache = self
            .config_field_cache
            .get_mut(config_name)
            .unwrap()
            .get_mut(field_name)
            .unwrap();
        cache.value = value.clone();
        cache.computed = true;
        cache.in_progress = false;
        Ok(value)
    }

    fn expect_int(&self, value: FeValue, span: &SourceSpan, context: &str) -> Result<i64, String> {
        as_int(&value).map_err(|_| {
            format!(
                "{context} must evaluate to an int at {}:{}",
                span.line, span.column
            )
        })
    }

    fn merge_tensor_types(&self, lhs: &FeType, rhs: &FeType) -> FeType {
        if lhs.kind != FeTypeKind::Tensor {
            return rhs.clone();
        }
        if rhs.kind != FeTypeKind::Tensor {
            return lhs.clone();
        }
        FeType::tensor(
            lhs.tensor_dtype
                .clone()
                .or_else(|| rhs.tensor_dtype.clone()),
            lhs.tensor_shape_expr
                .clone()
                .or_else(|| rhs.tensor_shape_expr.clone()),
            lhs.tensor_rank.or(rhs.tensor_rank),
        )
    }

    fn infer_call_result_type(
        &self,
        callee: &str,
        declared_type: &FeType,
        arg_types: &[FeType],
    ) -> FeType {
        infer_fe_call_result_type(callee, declared_type, arg_types)
    }

    fn normalize_callable_calls(
        &self,
        body: &mut [FeStmt],
        symbols: &mut BTreeMap<String, FeType>,
    ) {
        for stmt in body {
            self.normalize_callable_calls_in_stmt(stmt, symbols);
        }
    }

    fn normalize_callable_calls_in_stmt(
        &self,
        stmt: &mut FeStmt,
        symbols: &mut BTreeMap<String, FeType>,
    ) {
        match &mut stmt.kind {
            FeStmtKind::VarDecl { name, ty, value } => {
                if let Some(value) = value {
                    self.normalize_callable_calls_in_expr(value, symbols);
                }
                symbols.insert(name.clone(), ty.clone());
            }
            FeStmtKind::Assign { name, value } => {
                self.normalize_callable_calls_in_expr(value, symbols);
                symbols.insert(name.clone(), value.ty.clone());
            }
            FeStmtKind::Return { value } | FeStmtKind::Expr { value } => {
                self.normalize_callable_calls_in_expr(value, symbols);
            }
            FeStmtKind::If {
                condition,
                then_body,
                elif_bodies,
                else_body,
            } => {
                self.normalize_callable_calls_in_expr(condition, symbols);
                let mut then_symbols = symbols.clone();
                self.normalize_callable_calls(then_body, &mut then_symbols);
                for (condition, body) in elif_bodies {
                    self.normalize_callable_calls_in_expr(condition, symbols);
                    let mut elif_symbols = symbols.clone();
                    self.normalize_callable_calls(body, &mut elif_symbols);
                }
                let mut else_symbols = symbols.clone();
                self.normalize_callable_calls(else_body, &mut else_symbols);
            }
        }
    }

    fn normalize_callable_calls_in_expr(
        &self,
        expr: &mut FeExpr,
        symbols: &BTreeMap<String, FeType>,
    ) {
        match &mut expr.kind {
            FeExprKind::Call { callee, args } => {
                for arg in args.iter_mut() {
                    self.normalize_callable_calls_in_expr(&mut arg.value, symbols);
                }
                if let Some(symbol) = symbols.get(callee) {
                    if symbol.kind == FeTypeKind::Callable {
                        let callee_name = callee.clone();
                        let args = std::mem::take(args);
                        let ty = expr.ty.clone();
                        *expr = FeExpr::apply(FeExpr::var(callee_name, symbol.clone()), args, ty);
                    }
                }
            }
            FeExprKind::Apply { callee, args } => {
                self.normalize_callable_calls_in_expr(callee, symbols);
                for arg in args.iter_mut() {
                    self.normalize_callable_calls_in_expr(&mut arg.value, symbols);
                }
            }
            FeExprKind::LayerCtor { args, .. } => {
                for arg in args.iter_mut() {
                    self.normalize_callable_calls_in_expr(&mut arg.value, symbols);
                }
            }
            FeExprKind::Tuple(elements) => {
                for element in elements {
                    self.normalize_callable_calls_in_expr(element, symbols);
                }
            }
            FeExprKind::Binary { lhs, rhs, .. } => {
                self.normalize_callable_calls_in_expr(lhs, symbols);
                self.normalize_callable_calls_in_expr(rhs, symbols);
            }
            FeExprKind::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                self.normalize_callable_calls_in_expr(condition, symbols);
                self.normalize_callable_calls_in_expr(then_expr, symbols);
                self.normalize_callable_calls_in_expr(else_expr, symbols);
            }
            FeExprKind::Constant(_) | FeExprKind::Var(_) => {}
        }
    }

    fn bind_symbol(&mut self, name: &str, ty: FeType) {
        self.current_symbols.insert(name.to_string(), ty);
    }

    fn find_symbol(&self, name: &str) -> Option<FeType> {
        self.current_symbols.get(name).cloned()
    }
}

pub fn lower_type(ty: &Type) -> FeType {
    match ty.base {
        crate::compiler::parser::TypeBase::Int => match ty.scalar_dtype.as_deref() {
            Some("int16") => FeType::int16(),
            Some("int32") => FeType::int32(),
            Some("int64") => FeType::int64(),
            _ => FeType::int(),
        },
        crate::compiler::parser::TypeBase::Float => match ty.scalar_dtype.as_deref() {
            Some("float16") => FeType::float16(),
            Some("float32") => FeType::float32(),
            Some("float64") => FeType::float64(),
            _ => FeType::float(),
        },
        crate::compiler::parser::TypeBase::Bool => FeType::bool(),
        crate::compiler::parser::TypeBase::Tensor => FeType::tensor(
            ty.tensor_dtype.clone(),
            ty.tensor_shape_expr.clone(),
            ty.tensor_rank,
        ),
        crate::compiler::parser::TypeBase::Void => FeType::void(),
        crate::compiler::parser::TypeBase::Tuple => {
            FeType::tuple(ty.elements.iter().map(lower_type).collect::<Vec<_>>())
        }
        crate::compiler::parser::TypeBase::Callable => FeType::callable(
            ty.callable_return
                .as_deref()
                .map(lower_type)
                .unwrap_or_else(FeType::void),
        ),
    }
}

fn merge_float_types(lhs: &FeType, rhs: &FeType) -> FeType {
    let rank = |ty: &FeType| match ty.scalar_dtype.as_deref() {
        Some("float64") => 3,
        Some("float32") | None => 2,
        Some("float16") => 1,
        _ => 2,
    };

    match rank(lhs).max(rank(rhs)) {
        3 => FeType::float64(),
        1 => FeType::float16(),
        _ if lhs.scalar_dtype.is_none() && rhs.scalar_dtype.is_none() => FeType::float(),
        _ => FeType::float32(),
    }
}

fn merge_int_types(lhs: &FeType, rhs: &FeType) -> FeType {
    let rank = |ty: &FeType| match ty.scalar_dtype.as_deref() {
        Some("int64") => 3,
        Some("int32") | None => 2,
        Some("int16") => 1,
        _ => 2,
    };

    match rank(lhs).max(rank(rhs)) {
        3 => FeType::int64(),
        1 => FeType::int16(),
        _ if lhs.scalar_dtype.is_none() && rhs.scalar_dtype.is_none() => FeType::int(),
        _ => FeType::int32(),
    }
}

fn count_arrow_stage_sites(expr: &Expr) -> i32 {
    match &expr.kind {
        ExprKind::Call { .. } => 1,
        ExprKind::Binary { lhs, rhs, .. } => {
            count_arrow_stage_sites(lhs) + count_arrow_stage_sites(rhs)
        }
        ExprKind::Unary { operand, .. } => count_arrow_stage_sites(operand),
        ExprKind::Ternary {
            then_expr,
            condition,
            else_expr,
        } => {
            count_arrow_stage_sites(then_expr)
                + count_arrow_stage_sites(condition)
                + count_arrow_stage_sites(else_expr)
        }
        ExprKind::Tuple(elements) => elements.iter().map(count_arrow_stage_sites).sum(),
        _ => 0,
    }
}

fn is_none(value: &FeValue) -> bool {
    matches!(value, FeValue::None)
}

fn as_double(value: &FeValue) -> Result<f64, String> {
    match value {
        FeValue::Int(value) => Ok(*value as f64),
        FeValue::Float(value) => Ok(*value),
        _ => Err("Expected numeric constant".to_string()),
    }
}

fn as_int(value: &FeValue) -> Result<i64, String> {
    match value {
        FeValue::Int(value) => Ok(*value),
        _ => Err("Expected integer constant".to_string()),
    }
}

fn as_bool(value: &FeValue) -> Result<bool, String> {
    match value {
        FeValue::Bool(value) => Ok(*value),
        _ => Err("Expected boolean constant".to_string()),
    }
}

fn pick_broadcast_value<T: Clone>(values: &[T], index: usize) -> Result<Option<T>, String> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() == 1 {
        return Ok(Some(values[0].clone()));
    }
    values
        .get(index)
        .cloned()
        .map(Some)
        .ok_or_else(|| "Train variant index out of range during execution planning".to_string())
}

fn pick_broadcast_string(values: &[String], index: usize) -> Result<Option<String>, String> {
    pick_broadcast_value(values, index)
}

fn resolve_objective_stmt(stmt: &FeStmt, objective_symbol: &str, run: &mut FeExecutionRun) -> bool {
    match &stmt.kind {
        FeStmtKind::VarDecl { name, ty, .. } => {
            if name == objective_symbol {
                run.objective_source = ObjectiveSource::Local;
                run.objective_type = ty.clone();
                true
            } else {
                false
            }
        }
        FeStmtKind::Assign { name, value } => {
            if name == objective_symbol {
                run.objective_source = ObjectiveSource::Local;
                run.objective_type = value.ty.clone();
                true
            } else {
                false
            }
        }
        FeStmtKind::If {
            then_body,
            elif_bodies,
            else_body,
            ..
        } => {
            then_body
                .iter()
                .any(|stmt| resolve_objective_stmt(stmt, objective_symbol, run))
                || elif_bodies.iter().any(|(_, body)| {
                    body.iter()
                        .any(|stmt| resolve_objective_stmt(stmt, objective_symbol, run))
                })
                || else_body
                    .iter()
                    .any(|stmt| resolve_objective_stmt(stmt, objective_symbol, run))
        }
        FeStmtKind::Return { .. } | FeStmtKind::Expr { .. } => false,
    }
}
