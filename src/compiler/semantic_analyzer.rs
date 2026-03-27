use crate::compiler::builtins::registry::all_builtin_signatures;
use crate::compiler::builtins::type_rules::infer_call_result_type;
use crate::compiler::parser::{
    Arg, AssignStmt, CallArgument, Expr, ExprKind, Function, IfBranch, Layer, Program, SourceSpan, Stmt, StmtKind,
    Train, Type, TypeBase, VarDecl,
};

use std::collections::{BTreeMap, BTreeSet};

const SUPPORTED_TENSOR_DTYPES: &[&str] = &[
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int16",
    "int32",
    "int64",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub ty: Type,
    pub array_size: Option<usize>,
    pub is_callable: bool,
    pub callable_return_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub name: String,
    pub return_type: Type,
    pub arg_types: Vec<Type>,
    pub min_arity: usize,
    pub max_arity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallableKind {
    None,
    Function,
    Layer,
}

#[derive(Debug, Default)]
pub struct SemanticAnalyzer {
    pub scopes: Vec<BTreeMap<String, Symbol>>,
    pub functions: BTreeMap<String, Signature>,
    pub layers: BTreeMap<String, Signature>,
    pub configs: BTreeMap<String, BTreeMap<String, Type>>,
    last_expr_type: Type,
    current_func_return_type: Option<Type>,
    current_callable_has_return: bool,
    current_callable_kind: CallableKind,
}

impl Default for CallableKind {
    fn default() -> Self {
        Self::None
    }
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self::default();
        analyzer.register_builtins();
        analyzer
    }

    pub fn analyze(&mut self, program: &Program) -> Result<(), String> {
        self.collect_configs(program)?;
        self.collect_layers(program)?;
        self.collect_functions(program)?;
        self.collect_trains(program)?;

        self.push_scope();
        for stmt in &program.globals {
            self.analyze_stmt(stmt, program)?;
        }
        for train in &program.trains {
            self.visit_train(train, program)?;
        }
        for layer in &program.layers {
            self.visit_layer(layer, program)?;
        }
        for function in &program.functions {
            self.visit_function(function, program)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn register_builtins(&mut self) {
        for builtin in all_builtin_signatures() {
            self.functions.insert(
                builtin.name.clone(),
                Signature {
                    name: builtin.name,
                    return_type: builtin.return_type,
                    arg_types: builtin.arg_types,
                    min_arity: builtin.min_arity,
                    max_arity: builtin.max_arity,
                },
            );
        }
    }

    fn collect_configs(&mut self, program: &Program) -> Result<(), String> {
        let mut config_names = BTreeSet::new();
        for config in &program.configs {
            if !config_names.insert(config.name.clone()) {
                return Err(self.error_span(&config.span, &format!("Duplicate config '{}'", config.name)));
            }
            let mut fields = BTreeMap::new();
            for field in &config.fields {
                self.validate_declared_type(&field.ty, &config.span)?;
                if fields.contains_key(&field.name) {
                    return Err(self.error_span(
                        &config.span,
                        &format!("Duplicate field '{}' in config '{}'", field.name, config.name),
                    ));
                }
                fields.insert(field.name.clone(), field.ty.clone());
            }
            self.configs.insert(config.name.clone(), fields);
        }
        Ok(())
    }

    fn collect_layers(&mut self, program: &Program) -> Result<(), String> {
        let mut layer_names = BTreeSet::new();
        for layer in &program.layers {
            self.validate_declared_type(&layer.return_type, &layer.span)?;
            for arg in &layer.args {
                self.validate_declared_type(&arg.ty, &layer.span)?;
            }
            if !layer_names.insert(layer.name.clone()) {
                return Err(self.error_span(&layer.span, &format!("Duplicate layer '{}'", layer.name)));
            }
            if self.functions.contains_key(&layer.name) {
                return Err(self.error_span(
                    &layer.span,
                    &format!("Layer '{}' conflicts with an existing function or builtin", layer.name),
                ));
            }
            let min_arity = layer.args.iter().filter(|arg| arg.default_value.is_none()).count();
            let signature = Signature {
                name: layer.name.clone(),
                return_type: layer.return_type.clone(),
                arg_types: layer.args.iter().map(|arg| arg.ty.clone()).collect(),
                min_arity,
                max_arity: layer.args.len(),
            };
            self.layers.insert(layer.name.clone(), signature);
        }
        Ok(())
    }

    fn collect_functions(&mut self, program: &Program) -> Result<(), String> {
        for function in &program.functions {
            self.validate_declared_type(&function.return_type, &function.span)?;
            for arg in &function.args {
                self.validate_declared_type(&arg.ty, &function.span)?;
            }
            if self.functions.contains_key(&function.name) || self.layers.contains_key(&function.name) {
                return Err(self.error_span(
                    &function.span,
                    &format!("Function '{}' conflicts with an existing declaration", function.name),
                ));
            }
            let min_arity = function.args.iter().filter(|arg| arg.default_value.is_none()).count();
            let signature = Signature {
                name: function.name.clone(),
                return_type: function.return_type.clone(),
                arg_types: function.args.iter().map(|arg| arg.ty.clone()).collect(),
                min_arity,
                max_arity: function.args.len(),
            };
            self.functions.insert(function.name.clone(), signature);
        }
        Ok(())
    }

    fn collect_trains(&mut self, program: &Program) -> Result<(), String> {
        let mut has_train_model = false;
        for train in &program.trains {
            if train.name == "model" {
                if has_train_model {
                    return Err(self.error_span(&train.span, "Duplicate 'train model' declaration"));
                }
                has_train_model = true;
            }
            self.validate_train_decl(train, program)?;
        }
        Ok(())
    }

    fn analyze_stmt(&mut self, stmt: &Stmt, program: &Program) -> Result<(), String> {
        match &stmt.kind {
            StmtKind::Return(expr) => {
                let value_type = self.analyze_expr(expr, program)?;
                self.current_callable_has_return = true;
                if let Some(expected) = &self.current_func_return_type {
                    if !self.is_compatible(expected, &value_type) {
                        return Err(self.error_span(
                            &stmt.span,
                            &format!(
                                "Return type mismatch. Expected {}, got {}",
                                type_to_string(expected),
                                type_to_string(&value_type)
                            ),
                        ));
                    }
                }
            }
            StmtKind::Expr(expr) => {
                self.analyze_expr(expr, program)?;
            }
            StmtKind::VarDecl(decl) => {
                self.validate_declared_type(&decl.ty, &stmt.span)?;
                if let Some(init) = &decl.init {
                    let init_type = self.analyze_expr(init, program)?;
                    if !self.is_compatible(&decl.ty, &init_type) {
                        return Err(self.error_span(
                            &stmt.span,
                            &format!("Initialization type mismatch for '{}'", decl.name),
                        ));
                    }
                }
                self.declare_var(&decl.name, decl.ty.clone(), &stmt.span)?;
            }
            StmtKind::Assign(assign) => {
                let value_type = self.analyze_expr(&assign.value, program)?;
                if let Some(var) = self.find_var(&assign.name) {
                    if !self.is_compatible(&var.ty, &value_type) {
                        return Err(self.error_span(
                            &stmt.span,
                            &format!("Assignment type mismatch for '{}'", assign.name),
                        ));
                    }
                } else {
                    self.declare_var(&assign.name, value_type, &stmt.span)?;
                }
            }
            StmtKind::Scope(stmts) => {
                self.push_scope();
                for inner in stmts {
                    self.analyze_stmt(inner, program)?;
                }
                self.pop_scope();
            }
            StmtKind::If {
                condition,
                then_stmt,
                elifs,
                else_stmt,
            } => {
                let cond_type = self.analyze_expr(condition, program)?;
                self.ensure_condition_type(&cond_type, &condition.span, "If condition")?;
                self.analyze_stmt(then_stmt, program)?;
                for branch in elifs {
                    let elif_type = self.analyze_expr(&branch.condition, program)?;
                    self.ensure_condition_type(&elif_type, &branch.condition.span, "Elif condition")?;
                    self.analyze_stmt(&branch.body, program)?;
                }
                if let Some(otherwise) = else_stmt {
                    self.analyze_stmt(otherwise, program)?;
                }
            }
        }
        Ok(())
    }

    fn analyze_expr(&mut self, expr: &Expr, program: &Program) -> Result<Type, String> {
        let ty = match &expr.kind {
            ExprKind::IntLit(_) => Type::int(),
            ExprKind::FloatLit(_) => Type::float(),
            ExprKind::BoolLit(_) => Type::bool(),
            ExprKind::StringLit(_) => Type::void(),
            ExprKind::Identifier(name) => self.visit_identifier(name, &expr.span)?,
            ExprKind::Call { callee, args } => self.visit_call(callee, args, &expr.span, program)?,
            ExprKind::Unary { operand, op } => self.visit_unary(operand, *op, &expr.span, program)?,
            ExprKind::Binary { lhs, rhs, op } => self.visit_binary(lhs, rhs, *op, &expr.span, program)?,
            ExprKind::Ternary {
                then_expr,
                condition,
                else_expr,
            } => self.visit_ternary(then_expr, condition, else_expr, &expr.span, program)?,
            ExprKind::Tuple(elements) => {
                let mut tys = Vec::with_capacity(elements.len());
                for element in elements {
                    tys.push(self.analyze_expr(element, program)?);
                }
                Type::tuple(tys)
            }
            ExprKind::Arrow { source, stages } => self.visit_arrow(source, stages, &expr.span, program)?,
        };
        self.last_expr_type = ty.clone();
        Ok(ty)
    }

    fn visit_identifier(&mut self, name: &str, span: &SourceSpan) -> Result<Type, String> {
        if name == "None" {
            return Ok(Type::void());
        }
        if let Some(var) = self.find_var(name) {
            return Ok(var.ty.clone());
        }
        if self.configs.contains_key(name) {
            return Ok(Type::void());
        }
        let _ = span;
        Ok(Type::void())
    }

    fn visit_call(
        &mut self,
        callee: &str,
        args: &[CallArgument],
        span: &SourceSpan,
        program: &Program,
    ) -> Result<Type, String> {
        if let Some(sig) = self.functions.get(callee).cloned() {
            self.ensure_call_allowed(callee, false, span)?;
            self.validate_signature_arity(&sig, args.len(), span)?;
            let mut arg_types = Vec::new();
            for (index, arg) in args.iter().enumerate() {
                let actual = self.analyze_expr(&arg.value, program)?;
                arg_types.push(actual.clone());
                if index < sig.arg_types.len() && !self.is_compatible(&sig.arg_types[index], &actual) {
                    return Err(self.error_span(
                        span,
                        &format!(
                            "Argument {} to '{}' has incompatible type. Expected {}, got {}",
                            index + 1,
                            callee,
                            type_to_string(&sig.arg_types[index]),
                            type_to_string(&actual)
                        ),
                    ));
                }
            }
            return Ok(infer_call_result_type(callee, &sig.return_type, &arg_types));
        }

        if let Some(sig) = self.layers.get(callee).cloned() {
            self.ensure_call_allowed(callee, true, span)?;
            self.validate_signature_arity(&sig, args.len(), span)?;
            let mut arg_types = Vec::new();
            for (index, arg) in args.iter().enumerate() {
                let actual = self.analyze_expr(&arg.value, program)?;
                arg_types.push(actual.clone());
                if index < sig.arg_types.len() && !self.is_compatible(&sig.arg_types[index], &actual) {
                    return Err(self.error_span(
                        span,
                        &format!(
                            "Argument {} to '{}' has incompatible type. Expected {}, got {}",
                            index + 1,
                            callee,
                            type_to_string(&sig.arg_types[index]),
                            type_to_string(&actual)
                        ),
                    ));
                }
            }
            return Ok(infer_call_result_type(callee, &sig.return_type, &arg_types));
        }

        if let Some(var) = self.find_var(callee).cloned() {
            if !var.ty.is_callable() {
                return Err(self.error_span(span, &format!("Variable '{callee}' is not callable")));
            }
            if args.len() != 1 {
                return Err(self.error_span(
                    span,
                    &format!("Callable value '{callee}' must be invoked with exactly one pipeline/input argument"),
                ));
            }
            let input_type = self.analyze_expr(&args[0].value, program)?;
            if let Some(callable_return) = &var.ty.callable_return {
                if callable_return.base == TypeBase::Tensor && input_type.base == TypeBase::Tensor {
                    return Ok(Type::tensor(
                        input_type.tensor_dtype.clone(),
                        callable_return.tensor_shape_expr.clone(),
                        input_type.tensor_rank,
                    ));
                }
                return Ok((**callable_return).clone());
            }
            return Ok(Type::tensor(None, None, None));
        }

        Err(self.error_span(span, &format!("Undefined function or layer '{callee}'")))
    }

    fn visit_unary(
        &mut self,
        operand: &Expr,
        op: crate::compiler::lexer::TokenType,
        span: &SourceSpan,
        program: &Program,
    ) -> Result<Type, String> {
        let operand_type = self.analyze_expr(operand, program)?;
        match op {
            crate::compiler::lexer::TokenType::Bang => {
                self.ensure_condition_type(&operand_type, span, "Unary '!'")?;
                Ok(Type::bool())
            }
            crate::compiler::lexer::TokenType::Minus => {
                if !matches!(operand_type.base, TypeBase::Int | TypeBase::Float | TypeBase::Tensor | TypeBase::Void) {
                    return Err(self.error_span(span, "Unary '-' expects int, float, or tensor operand"));
                }
                Ok(operand_type)
            }
            _ => Ok(operand_type),
        }
    }

    fn visit_binary(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        op: crate::compiler::lexer::TokenType,
        span: &SourceSpan,
        program: &Program,
    ) -> Result<Type, String> {
        let lhs_ty = self.analyze_expr(lhs, program)?;
        if op == crate::compiler::lexer::TokenType::Dot {
            if let (ExprKind::Identifier(config_name), ExprKind::Identifier(field_name)) = (&lhs.kind, &rhs.kind) {
                if let Some(config) = self.configs.get(config_name) {
                    if let Some(field_ty) = config.get(field_name) {
                        return Ok(field_ty.clone());
                    }
                    return Err(self.error_span(
                        span,
                        &format!("Unknown field '{}' on config '{}'", field_name, config_name),
                    ));
                }
            }
            return Err(self.error_span(
                span,
                "Only config field access of the form config_name.field is supported",
            ));
        }
        let rhs_ty = self.analyze_expr(rhs, program)?;
        use crate::compiler::lexer::TokenType as T;
        match op {
            T::Plus | T::Minus | T::Star | T::Slash => {
                if lhs_ty.base == TypeBase::Tensor || rhs_ty.base == TypeBase::Tensor {
                    Ok(self.merge_tensor_types(&lhs_ty, &rhs_ty))
                } else if lhs_ty.base == TypeBase::Void || rhs_ty.base == TypeBase::Void {
                    Ok(Type::void())
                } else if lhs_ty.base == TypeBase::Float || rhs_ty.base == TypeBase::Float {
                    Ok(merge_scalar_float_types(&lhs_ty, &rhs_ty))
                } else if lhs_ty.base == TypeBase::Int && rhs_ty.base == TypeBase::Int {
                    Ok(merge_scalar_int_types(&lhs_ty, &rhs_ty))
                } else {
                    Err(self.error_span(span, "Arithmetic operation has incompatible operand types"))
                }
            }
            T::EqEq | T::Neq => {
                if !self.is_compatible(&lhs_ty, &rhs_ty) {
                    return Err(self.error_span(span, "Comparison expects compatible operand types"));
                }
                Ok(Type::bool())
            }
            T::Lt | T::Gt | T::LtEq | T::GtEq => {
                if !(matches!(lhs_ty.base, TypeBase::Int | TypeBase::Float | TypeBase::Void)
                    && matches!(rhs_ty.base, TypeBase::Int | TypeBase::Float | TypeBase::Void))
                {
                    return Err(self.error_span(span, "Ordered comparisons require int or float operands"));
                }
                Ok(Type::bool())
            }
            T::AmpAmp | T::PipePipe => {
                self.ensure_condition_type(&lhs_ty, &lhs.span, "Logical operand")?;
                self.ensure_condition_type(&rhs_ty, &rhs.span, "Logical operand")?;
                Ok(Type::bool())
            }
            T::StarEq => Ok(lhs_ty),
            _ => Err(self.error_span(span, "Unsupported binary operator in semantic analysis")),
        }
    }

    fn visit_ternary(
        &mut self,
        then_expr: &Expr,
        condition: &Expr,
        else_expr: &Expr,
        span: &SourceSpan,
        program: &Program,
    ) -> Result<Type, String> {
        let cond_ty = self.analyze_expr(condition, program)?;
        self.ensure_condition_type(&cond_ty, &condition.span, "Ternary condition")?;
        let then_ty = self.analyze_expr(then_expr, program)?;
        let else_ty = self.analyze_expr(else_expr, program)?;
        if !self.is_compatible(&then_ty, &else_ty) && !self.is_compatible(&else_ty, &then_ty) {
            return Err(self.error_span(span, "Ternary branches must have compatible types"));
        }
        if then_ty.base == TypeBase::Void {
            Ok(else_ty)
        } else {
            Ok(then_ty)
        }
    }

    fn visit_arrow(
        &mut self,
        source: &Expr,
        stages: &[Expr],
        span: &SourceSpan,
        program: &Program,
    ) -> Result<Type, String> {
        if self.current_callable_kind == CallableKind::Function {
            return Err(self.error_span(
                span,
                "The '->' operator is only allowed inside layer bodies; fn must use explicit calls",
            ));
        }
        let mut current = self.analyze_expr(source, program)?;
        for stage in stages {
            current = self.analyze_stage(stage, &current, program)?;
        }
        Ok(current)
    }

    fn analyze_stage(&mut self, expr: &Expr, input_type: &Type, program: &Program) -> Result<Type, String> {
        match &expr.kind {
            ExprKind::Call { callee, args } => self.analyze_arrow_call(callee, args, &expr.span, input_type, program),
            ExprKind::Unary { operand, op } => {
                let operand_ty = if count_stage_sites(operand) > 0 {
                    self.analyze_stage(operand, input_type, program)?
                } else {
                    self.analyze_expr(operand, program)?
                };
                match op {
                    crate::compiler::lexer::TokenType::Bang => {
                        self.ensure_condition_type(&operand_ty, &expr.span, "Unary '!'")?;
                        Ok(Type::bool())
                    }
                    crate::compiler::lexer::TokenType::Minus => {
                        if !matches!(operand_ty.base, TypeBase::Int | TypeBase::Float | TypeBase::Tensor) {
                            return Err(self.error_span(&expr.span, "Unary '-' expects int, float, or tensor operand"));
                        }
                        Ok(operand_ty)
                    }
                    _ => Ok(operand_ty),
                }
            }
            ExprKind::Binary { lhs, rhs, op } if *op == crate::compiler::lexer::TokenType::StarEq => {
                if !matches!(lhs.kind, ExprKind::Call { .. }) {
                    return Err(self.error_span(&expr.span, "Repeated arrow stage must begin with a call"));
                }
                let repeat_ty = self.analyze_expr(rhs, program)?;
                if repeat_ty.base != TypeBase::Int {
                    return Err(self.error_span(&rhs.span, "Repeated arrow stage count must have type int"));
                }
                if let ExprKind::Call { callee, args } = &lhs.kind {
                    self.analyze_arrow_call(callee, args, &lhs.span, input_type, program)
                } else {
                    unreachable!()
                }
            }
            ExprKind::Binary { lhs, rhs, op } => {
                let lhs_ty = if count_stage_sites(lhs) > 0 {
                    self.analyze_stage(lhs, input_type, program)?
                } else {
                    self.analyze_expr(lhs, program)?
                };
                let rhs_ty = if count_stage_sites(rhs) > 0 {
                    self.analyze_stage(rhs, input_type, program)?
                } else {
                    self.analyze_expr(rhs, program)?
                };
                use crate::compiler::lexer::TokenType as T;
                match op {
                    T::Plus | T::Minus | T::Star | T::Slash => {
                        if lhs_ty.base == TypeBase::Tensor || rhs_ty.base == TypeBase::Tensor {
                            Ok(self.merge_tensor_types(&lhs_ty, &rhs_ty))
                        } else if lhs_ty.base == TypeBase::Float || rhs_ty.base == TypeBase::Float {
                            Ok(merge_scalar_float_types(&lhs_ty, &rhs_ty))
                        } else if lhs_ty.base == TypeBase::Int && rhs_ty.base == TypeBase::Int {
                            Ok(merge_scalar_int_types(&lhs_ty, &rhs_ty))
                        } else {
                            Err(self.error_span(&expr.span, "Arithmetic operation has incompatible operand types"))
                        }
                    }
                    T::EqEq | T::Neq => {
                        if !self.is_compatible(&lhs_ty, &rhs_ty) {
                            return Err(self.error_span(&expr.span, "Comparison expects compatible operand types"));
                        }
                        Ok(Type::bool())
                    }
                    T::Lt | T::Gt | T::LtEq | T::GtEq => {
                        if !matches!(lhs_ty.base, TypeBase::Int | TypeBase::Float)
                            || !matches!(rhs_ty.base, TypeBase::Int | TypeBase::Float)
                        {
                            return Err(self.error_span(&expr.span, "Ordered comparisons require int or float operands"));
                        }
                        Ok(Type::bool())
                    }
                    T::AmpAmp | T::PipePipe => {
                        self.ensure_condition_type(&lhs_ty, &lhs.span, "Logical operand")?;
                        self.ensure_condition_type(&rhs_ty, &rhs.span, "Logical operand")?;
                        Ok(Type::bool())
                    }
                    _ => self.analyze_expr(expr, program),
                }
            }
            ExprKind::Ternary {
                then_expr,
                condition,
                else_expr,
            } => {
                let cond_ty = if count_stage_sites(condition) > 0 {
                    self.analyze_stage(condition, input_type, program)?
                } else {
                    self.analyze_expr(condition, program)?
                };
                self.ensure_condition_type(&cond_ty, &condition.span, "Ternary condition")?;
                let then_ty = if count_stage_sites(then_expr) > 0 {
                    self.analyze_stage(then_expr, input_type, program)?
                } else {
                    self.analyze_expr(then_expr, program)?
                };
                let else_ty = if count_stage_sites(else_expr) > 0 {
                    self.analyze_stage(else_expr, input_type, program)?
                } else {
                    self.analyze_expr(else_expr, program)?
                };
                if !self.is_compatible(&then_ty, &else_ty) && !self.is_compatible(&else_ty, &then_ty) {
                    return Err(self.error_span(&expr.span, "Ternary branches must have compatible types"));
                }
                if then_ty.base == TypeBase::Void {
                    Ok(else_ty)
                } else {
                    Ok(then_ty)
                }
            }
            ExprKind::Tuple(elements) => {
                let mut tys = Vec::with_capacity(elements.len());
                for element in elements {
                    tys.push(if count_stage_sites(element) > 0 {
                        self.analyze_stage(element, input_type, program)?
                    } else {
                        self.analyze_expr(element, program)?
                    });
                }
                Ok(Type::tuple(tys))
            }
            _ => self.analyze_expr(expr, program),
        }
    }

    fn analyze_arrow_call(
        &mut self,
        callee: &str,
        args: &[CallArgument],
        span: &SourceSpan,
        input_type: &Type,
        program: &Program,
    ) -> Result<Type, String> {
        let mut arg_types = vec![input_type.clone()];
        for arg in args {
            arg_types.push(self.analyze_expr(&arg.value, program)?);
        }

        if let Some(sig) = self.functions.get(callee).cloned() {
            let stage_type = if sig.return_type.base == TypeBase::Callable {
                let mut ctor_arg_types = Vec::new();
                for arg in args {
                    ctor_arg_types.push(self.analyze_expr(&arg.value, program)?);
                }
                self.validate_signature_arity(&sig, ctor_arg_types.len(), span)?;
                for (index, actual) in ctor_arg_types.iter().enumerate() {
                    if index < sig.arg_types.len() && !self.is_compatible(&sig.arg_types[index], actual) {
                        return Err(self.error_span(
                            span,
                            &format!(
                                "Argument {} to '{}' has incompatible type. Expected {}, got {}",
                                index + 1,
                                callee,
                                type_to_string(&sig.arg_types[index]),
                                type_to_string(actual)
                            ),
                        ));
                    }
                }
                infer_call_result_type(callee, &sig.return_type, &ctor_arg_types)
            } else {
                let piped_arity = arg_types.len();
                if piped_arity < sig.min_arity || piped_arity > sig.max_arity {
                    return Err(self.error_span(span, &format!("Arrow stage '{}' received invalid arity", callee)));
                }
                infer_call_result_type(callee, &sig.return_type, &arg_types)
            };
            return self.unwrap_callable_stage(stage_type, input_type);
        }
        if let Some(sig) = self.layers.get(callee).cloned() {
            self.ensure_call_allowed(callee, true, span)?;
            return Ok(sig.return_type);
        }
        if let Some(var) = self.find_var(callee) {
            if !var.ty.is_callable() {
                return Err(self.error_span(span, &format!("Arrow stage '{}' is not callable", callee)));
            }
            return self.unwrap_callable_stage(var.ty.clone(), input_type);
        }
        Err(self.error_span(span, &format!("Arrow stage '{}' is not callable", callee)))
    }

    fn unwrap_callable_stage(&self, stage_type: Type, input_type: &Type) -> Result<Type, String> {
        if stage_type.base == TypeBase::Callable {
            if let Some(callable_return) = &stage_type.callable_return {
                if callable_return.base == TypeBase::Tensor && input_type.base == TypeBase::Tensor {
                    return Ok(Type::tensor(
                        input_type.tensor_dtype.clone(),
                        callable_return.tensor_shape_expr.clone(),
                        input_type.tensor_rank,
                    ));
                }
                return Ok((**callable_return).clone());
            }
            return Ok(Type::tensor(None, None, None));
        }
        Ok(stage_type)
    }

    fn visit_function(&mut self, function: &Function, program: &Program) -> Result<(), String> {
        self.visit_callable(
            &function.args,
            &function.return_type,
            &function.body,
            &function.span,
            &function.name,
            CallableKind::Function,
            "Function",
            program,
        )
    }

    fn visit_layer(&mut self, layer: &Layer, program: &Program) -> Result<(), String> {
        self.visit_callable(
            &layer.args,
            &layer.return_type,
            &layer.body,
            &layer.span,
            &layer.name,
            CallableKind::Layer,
            "Layer",
            program,
        )
    }

    fn visit_callable(
        &mut self,
        args: &[Arg],
        return_type: &Type,
        body: &Stmt,
        span: &SourceSpan,
        name: &str,
        kind: CallableKind,
        label: &str,
        program: &Program,
    ) -> Result<(), String> {
        let previous_kind = self.current_callable_kind;
        self.current_func_return_type = Some(return_type.clone());
        self.current_callable_has_return = false;
        self.current_callable_kind = kind;
        self.push_scope();
        for arg in args {
            if let Some(default) = &arg.default_value {
                let default_type = self.analyze_expr(default, program)?;
                if !self.is_compatible(&arg.ty, &default_type) {
                    return Err(self.error_span(
                        span,
                        &format!("Default value for argument '{}' has incompatible type", arg.name),
                    ));
                }
            }
            self.declare_var(&arg.name, arg.ty.clone(), span)?;
        }
        self.analyze_stmt(body, program)?;
        self.pop_scope();
        if return_type.base != TypeBase::Void && !self.current_callable_has_return {
            return Err(self.error_span(span, &format!("{label} '{name}' is missing a return statement")));
        }
        self.current_func_return_type = None;
        self.current_callable_kind = previous_kind;
        Ok(())
    }

    fn visit_train(&mut self, train: &Train, program: &Program) -> Result<(), String> {
        self.validate_train_decl(train, program)
    }

    fn validate_train_decl(&self, train: &Train, program: &Program) -> Result<(), String> {
        let mut field_names = BTreeSet::new();
        let mut variant_count: Option<usize> = None;
        for field in &train.fields {
            if !field_names.insert(field.name.clone()) {
                return Err(self.error_span(
                    &train.span,
                    &format!("Duplicate field '{}' in train '{}'", field.name, train.name),
                ));
            }
            if field.name == "subtrain" {
                return Err(self.error_span(
                    field.init.as_ref().map(|expr| &expr.span).unwrap_or(&train.span),
                    "Field 'subtrain' is no longer supported; use tuple-valued fields on 'train model' instead",
                ));
            }
            if matches!(
                field.name.as_str(),
                "backend" | "target" | "device" | "optimizer" | "lr" | "learning_rate" | "objective" | "iteration"
            ) {
                if let Some(init) = &field.init {
                    if let Some(arity) = tuple_arity(init) {
                        if arity == 0 {
                            return Err(self.error_span(
                                &init.span,
                                &format!("Train field '{}' cannot use an empty tuple", field.name),
                            ));
                        }
                        if let Some(count) = variant_count {
                            if count != arity {
                                return Err(self.error_span(
                                    &init.span,
                                    "Tuple-valued train fields must have the same length; use scalar values to broadcast",
                                ));
                            }
                        } else {
                            variant_count = Some(arity);
                        }
                    }
                }
            }
        }

        let model_symbols = self.collect_model_symbols(program);
        for field in &train.fields {
            if field.name != "objective" {
                continue;
            }
            let Some(init) = &field.init else { continue };
            let objective_exprs: Vec<&Expr> = match &init.kind {
                ExprKind::Tuple(elements) => elements.iter().collect(),
                _ => vec![init],
            };
            for expr in objective_exprs {
                if let ExprKind::Identifier(name) = &expr.kind {
                    if !model_symbols.is_empty() && !model_symbols.contains(name) {
                        return Err(self.error_span(
                            &expr.span,
                            &format!("Field 'objective' references unknown model root '{}'", name),
                        ));
                    }
                } else {
                    return Err(self.error_span(
                        &expr.span,
                        "Field 'objective' must reference a named tensor root, not a string literal or arbitrary expression",
                    ));
                }
            }
        }

        Ok(())
    }

    fn find_model_layer<'a>(&self, program: &'a Program) -> Option<&'a Layer> {
        program.layers.iter().find(|layer| layer.name == "model")
    }

    fn collect_model_symbols(&self, program: &Program) -> BTreeSet<String> {
        let Some(model) = self.find_model_layer(program) else {
            return BTreeSet::new();
        };
        let mut symbols = BTreeSet::new();
        for arg in &model.args {
            symbols.insert(arg.name.clone());
        }
        collect_stmt_symbols(&model.body, &mut symbols);
        symbols
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_var(&mut self, name: &str, ty: Type, span: &SourceSpan) -> Result<(), String> {
        if self.scopes.is_empty() {
            self.push_scope();
        }
        if self.scopes.last().is_some_and(|scope| scope.contains_key(name)) {
            return Err(self.error_span(span, &format!("Variable '{}' already declared in this scope", name)));
        }
        let symbol = Symbol {
            ty: ty.clone(),
            array_size: None,
            is_callable: ty.is_callable(),
            callable_return_type: ty.callable_return.as_ref().map(|inner| (**inner).clone()),
        };
        self.scopes.last_mut().unwrap().insert(name.to_string(), symbol);
        Ok(())
    }

    fn find_var(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    fn validate_signature_arity(&self, sig: &Signature, actual_arity: usize, span: &SourceSpan) -> Result<(), String> {
        if actual_arity < sig.min_arity || actual_arity > sig.max_arity {
            let expected = if sig.min_arity == sig.max_arity {
                sig.min_arity.to_string()
            } else {
                format!("{} to {}", sig.min_arity, sig.max_arity)
            };
            return Err(self.error_span(
                span,
                &format!("Call to '{}' expects {} argument(s), but got {}", sig.name, expected, actual_arity),
            ));
        }
        Ok(())
    }

    fn ensure_condition_type(&self, ty: &Type, span: &SourceSpan, context: &str) -> Result<(), String> {
        if ty.base != TypeBase::Bool && ty.base != TypeBase::Void {
            return Err(self.error_span(
                span,
                &format!(
                    "{} must have type bool or an optional/None-like value, but got {}",
                    context,
                    type_to_string(ty)
                ),
            ));
        }
        Ok(())
    }

    fn ensure_call_allowed(&self, callee: &str, is_layer: bool, span: &SourceSpan) -> Result<(), String> {
        if is_layer && self.current_callable_kind == CallableKind::Function {
            return Err(self.error_span(
                span,
                &format!(
                    "Function '{}' is a layer and cannot be called from fn; move this logic into a layer if you want tracked state/graph semantics",
                    callee
                ),
            ));
        }
        Ok(())
    }

    fn is_compatible(&self, target: &Type, source: &Type) -> bool {
        if target.base == TypeBase::Void || source.base == TypeBase::Void {
            return true;
        }
        if target.base != source.base {
            return false;
        }
        if (target.base == TypeBase::Float || target.base == TypeBase::Int)
            && target.scalar_dtype.is_some()
            && source.scalar_dtype.is_some()
            && target.scalar_dtype != source.scalar_dtype
        {
            return false;
        }
        if target.base == TypeBase::Tuple {
            if target.elements.len() != source.elements.len() {
                return false;
            }
            for (lhs, rhs) in target.elements.iter().zip(&source.elements) {
                if !self.is_compatible(lhs, rhs) {
                    return false;
                }
            }
        }
        if target.base == TypeBase::Tensor {
            if target.tensor_dtype.is_some() && source.tensor_dtype.is_some() && target.tensor_dtype != source.tensor_dtype
            {
                return false;
            }
            if target.tensor_shape_expr.is_some()
                && source.tensor_shape_expr.is_some()
                && target.tensor_shape_expr != source.tensor_shape_expr
            {
                return false;
            }
            if target.tensor_rank.is_some() && source.tensor_rank.is_some() && target.tensor_rank != source.tensor_rank {
                return false;
            }
        }
        if target.base == TypeBase::Callable {
            match (&target.callable_return, &source.callable_return) {
                (Some(lhs), Some(rhs)) => self.is_compatible(lhs, rhs),
                (None, None) => true,
                _ => false,
            }
        } else {
            true
        }
    }

    fn merge_tensor_types(&self, lhs: &Type, rhs: &Type) -> Type {
        if lhs.base != TypeBase::Tensor {
            return rhs.clone();
        }
        if rhs.base != TypeBase::Tensor {
            return lhs.clone();
        }
        Type::tensor(
            lhs.tensor_dtype.clone().or_else(|| rhs.tensor_dtype.clone()),
            lhs.tensor_shape_expr.clone().or_else(|| rhs.tensor_shape_expr.clone()),
            lhs.tensor_rank.or(rhs.tensor_rank),
        )
    }

    fn error_span(&self, span: &SourceSpan, message: &str) -> String {
        format!("Semantic Error: {message} at {}:{}", span.line, span.column)
    }

    fn validate_declared_type(&self, ty: &Type, span: &SourceSpan) -> Result<(), String> {
        match ty.base {
            TypeBase::Int => {
                if let Some(dtype) = &ty.scalar_dtype {
                    if !matches!(dtype.as_str(), "int16" | "int32" | "int64") {
                        return Err(self.error_span(span, &format!("Unsupported scalar integer type '{dtype}'")));
                    }
                }
            }
            TypeBase::Float => {
                if let Some(dtype) = &ty.scalar_dtype {
                    if !matches!(dtype.as_str(), "float16" | "float32" | "float64") {
                        return Err(self.error_span(span, &format!("Unsupported scalar float type '{dtype}'")));
                    }
                }
            }
            TypeBase::Tensor => {
                if let Some(dtype) = &ty.tensor_dtype {
                    if !SUPPORTED_TENSOR_DTYPES.contains(&dtype.as_str()) {
                        return Err(self.error_span(span, &format!("Unsupported tensor dtype '{dtype}'")));
                    }
                }
            }
            TypeBase::Tuple => {
                for element in &ty.elements {
                    self.validate_declared_type(element, span)?;
                }
            }
            TypeBase::Callable => {
                if let Some(result) = &ty.callable_return {
                    self.validate_declared_type(result, span)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn tuple_arity(expr: &Expr) -> Option<usize> {
    match &expr.kind {
        ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::Identifier(_)
        | ExprKind::Unary { .. }
        | ExprKind::Binary { .. }
        | ExprKind::Ternary { .. } => None,
        ExprKind::Tuple(elements) => Some(elements.len()),
        _ => None,
    }
}

fn collect_expr_identifiers(expr: &Expr, symbols: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Identifier(name) => {
            if name != "None" {
                symbols.insert(name.clone());
            }
        }
        ExprKind::Call { args, .. } => {
            for arg in args {
                collect_expr_identifiers(&arg.value, symbols);
            }
        }
        ExprKind::Unary { operand, .. } => collect_expr_identifiers(operand, symbols),
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_expr_identifiers(lhs, symbols);
            collect_expr_identifiers(rhs, symbols);
        }
        ExprKind::Ternary {
            then_expr,
            condition,
            else_expr,
        } => {
            collect_expr_identifiers(then_expr, symbols);
            collect_expr_identifiers(condition, symbols);
            collect_expr_identifiers(else_expr, symbols);
        }
        ExprKind::Tuple(elements) => {
            for element in elements {
                collect_expr_identifiers(element, symbols);
            }
        }
        ExprKind::Arrow { source, stages } => {
            collect_expr_identifiers(source, symbols);
            for stage in stages {
                collect_expr_identifiers(stage, symbols);
            }
        }
        ExprKind::IntLit(_) | ExprKind::FloatLit(_) | ExprKind::BoolLit(_) | ExprKind::StringLit(_) => {}
    }
}

fn collect_stmt_symbols(stmt: &Stmt, symbols: &mut BTreeSet<String>) {
    match &stmt.kind {
        StmtKind::VarDecl(VarDecl { name, init, .. }) => {
            symbols.insert(name.clone());
            if let Some(init) = init {
                collect_expr_identifiers(init, symbols);
            }
        }
        StmtKind::Assign(AssignStmt { name, value }) => {
            symbols.insert(name.clone());
            collect_expr_identifiers(value, symbols);
        }
        StmtKind::Return(expr) | StmtKind::Expr(expr) => collect_expr_identifiers(expr, symbols),
        StmtKind::Scope(stmts) => {
            for inner in stmts {
                collect_stmt_symbols(inner, symbols);
            }
        }
        StmtKind::If {
            condition,
            then_stmt,
            elifs,
            else_stmt,
        } => {
            collect_expr_identifiers(condition, symbols);
            collect_stmt_symbols(then_stmt, symbols);
            for IfBranch { condition, body } in elifs {
                collect_expr_identifiers(condition, symbols);
                collect_stmt_symbols(body, symbols);
            }
            if let Some(otherwise) = else_stmt {
                collect_stmt_symbols(otherwise, symbols);
            }
        }
    }
}

fn count_stage_sites(expr: &Expr) -> i32 {
    match &expr.kind {
        ExprKind::Call { .. } => 1,
        ExprKind::Binary { lhs, rhs, .. } => count_stage_sites(lhs) + count_stage_sites(rhs),
        ExprKind::Unary { operand, .. } => count_stage_sites(operand),
        ExprKind::Ternary {
            then_expr,
            condition,
            else_expr,
        } => count_stage_sites(then_expr) + count_stage_sites(condition) + count_stage_sites(else_expr),
        ExprKind::Tuple(elements) => elements.iter().map(count_stage_sites).sum(),
        _ => 0,
    }
}

pub fn type_to_string(ty: &Type) -> String {
    match ty.base {
        TypeBase::Int => ty.scalar_dtype.clone().unwrap_or_else(|| "int".to_string()),
        TypeBase::Bool => "bool".to_string(),
        TypeBase::Float => ty.scalar_dtype.clone().unwrap_or_else(|| "float".to_string()),
        TypeBase::Tensor => match (&ty.tensor_dtype, &ty.tensor_shape_expr) {
            (None, None) => "tensor".to_string(),
            (Some(dtype), None) => format!("tensor[{dtype}]"),
            (None, Some(shape)) => format!("tensor[{shape}]"),
            (Some(dtype), Some(shape)) => format!("tensor[{dtype}, {shape}]"),
        },
        TypeBase::Void => "void".to_string(),
        TypeBase::Tuple => {
            let inner = ty
                .elements
                .iter()
                .map(type_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inner})")
        }
        TypeBase::Callable => {
            let inner = ty
                .callable_return
                .as_ref()
                .map(|inner| type_to_string(inner))
                .unwrap_or_else(|| "void".to_string());
            format!("callable -> {inner}")
        }
    }
}

fn merge_scalar_float_types(lhs: &Type, rhs: &Type) -> Type {
    let rank = |ty: &Type| match ty.scalar_dtype.as_deref() {
        Some("float64") => 3,
        Some("float32") | None => 2,
        Some("float16") => 1,
        _ => 2,
    };

    match rank(lhs).max(rank(rhs)) {
        3 => Type::float64(),
        1 => Type::float16(),
        _ if lhs.scalar_dtype.is_none() && rhs.scalar_dtype.is_none() => Type::float(),
        _ => Type::float32(),
    }
}

fn merge_scalar_int_types(lhs: &Type, rhs: &Type) -> Type {
    let rank = |ty: &Type| match ty.scalar_dtype.as_deref() {
        Some("int64") => 3,
        Some("int32") | None => 2,
        Some("int16") => 1,
        _ => 2,
    };

    match rank(lhs).max(rank(rhs)) {
        3 => Type::int64(),
        1 => Type::int16(),
        _ if lhs.scalar_dtype.is_none() && rhs.scalar_dtype.is_none() => Type::int(),
        _ => Type::int32(),
    }
}

trait TypeExt {
    fn is_callable(&self) -> bool;
}

impl TypeExt for Type {
    fn is_callable(&self) -> bool {
        self.base == TypeBase::Callable
    }
}
