use crate::compiler::lexer::{Token, TokenType, TokenValue};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceSpan {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeBase {
    Int,
    Void,
    Bool,
    Float,
    Tensor,
    Tuple,
    Callable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub base: TypeBase,
    pub elements: Vec<Type>,
    pub callable_return: Option<Box<Type>>,
    pub scalar_dtype: Option<String>,
    pub tensor_dtype: Option<String>,
    pub tensor_shape_expr: Option<String>,
    pub tensor_rank: Option<usize>,
}

impl Default for Type {
    fn default() -> Self {
        Self::void()
    }
}

impl Type {
    pub fn int() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: None,
            ..Self::void()
        }
    }
    pub fn int16() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int16".to_string()),
            ..Self::void()
        }
    }
    pub fn int32() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int32".to_string()),
            ..Self::void()
        }
    }
    pub fn int64() -> Self {
        Self {
            base: TypeBase::Int,
            scalar_dtype: Some("int64".to_string()),
            ..Self::void()
        }
    }
    pub fn float16() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float16".to_string()),
            ..Self::void()
        }
    }
    pub fn float32() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float32".to_string()),
            ..Self::void()
        }
    }
    pub fn float64() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: Some("float64".to_string()),
            ..Self::void()
        }
    }
    pub fn float() -> Self {
        Self {
            base: TypeBase::Float,
            scalar_dtype: None,
            ..Self::void()
        }
    }
    pub fn bool() -> Self {
        Self {
            base: TypeBase::Bool,
            ..Self::void()
        }
    }
    pub fn void() -> Self {
        Self {
            base: TypeBase::Void,
            elements: Vec::new(),
            callable_return: None,
            scalar_dtype: None,
            tensor_dtype: None,
            tensor_shape_expr: None,
            tensor_rank: None,
        }
    }
    pub fn tensor(dtype: Option<String>, shape_expr: Option<String>, rank: Option<usize>) -> Self {
        Self {
            base: TypeBase::Tensor,
            tensor_dtype: dtype,
            tensor_shape_expr: shape_expr,
            tensor_rank: rank,
            ..Self::void()
        }
    }
    pub fn tuple(elements: Vec<Type>) -> Self {
        Self {
            base: TypeBase::Tuple,
            elements,
            ..Self::void()
        }
    }
    pub fn callable(return_type: Type) -> Self {
        Self {
            base: TypeBase::Callable,
            callable_return: Some(Box::new(return_type)),
            ..Self::void()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallArgument {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),
    Identifier(String),
    Call { callee: String, args: Vec<CallArgument> },
    Unary { operand: Box<Expr>, op: TokenType },
    Binary { lhs: Box<Expr>, rhs: Box<Expr>, op: TokenType },
    Ternary { then_expr: Box<Expr>, condition: Box<Expr>, else_expr: Box<Expr> },
    Tuple(Vec<Expr>),
    Arrow { source: Box<Expr>, stages: Vec<Expr> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub span: SourceSpan,
    pub kind: ExprKind,
}

impl Expr {
    fn span(&self) -> SourceSpan {
        self.span.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
    pub array_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignStmt {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfBranch {
    pub condition: Expr,
    pub body: Box<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Return(Expr),
    Expr(Expr),
    VarDecl(VarDecl),
    Assign(AssignStmt),
    Scope(Vec<Stmt>),
    If { condition: Expr, then_stmt: Box<Stmt>, elifs: Vec<IfBranch>, else_stmt: Option<Box<Stmt>> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub span: SourceSpan,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub name: String,
    pub ty: Type,
    pub default_value: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub span: SourceSpan,
    pub name: String,
    pub args: Vec<Arg>,
    pub return_type: Type,
    pub body: Stmt,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub span: SourceSpan,
    pub name: String,
    pub args: Vec<Arg>,
    pub return_type: Type,
    pub body: Stmt,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub span: SourceSpan,
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Train {
    pub span: SourceSpan,
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub configs: Vec<Config>,
    pub trains: Vec<Train>,
    pub layers: Vec<Layer>,
    pub functions: Vec<Function>,
    pub globals: Vec<Stmt>,
}

pub struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, index: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut program = Program::default();
        while let Some(token) = self.peek(0).cloned() {
            match token.kind {
                TokenType::Newline | TokenType::Dedent => {
                    self.consume();
                }
                TokenType::Eof => break,
                TokenType::Layer => program.layers.push(self.parse_layer()?),
                TokenType::Fn => program.functions.push(self.parse_function()?),
                TokenType::Config => program.configs.push(self.parse_config()?),
                TokenType::Train => program.trains.push(self.parse_train()?),
                TokenType::Ident | TokenType::Int | TokenType::Float | TokenType::Bool => {
                    program.globals.push(self.parse_stmt()?);
                }
                _ => return Err(self.error_here("Unexpected token at top level")),
            }
        }
        Ok(program)
    }

    pub fn token_to_string(token: &Token) -> String {
        token.to_string()
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_tuple()
    }

    fn parse_tuple(&mut self) -> Result<Expr, String> {
        let first = self.parse_ternary()?;
        if self.peek_kind(0) == Some(TokenType::Comma) {
            let span = first.span();
            let mut elements = vec![first];
            while self.peek_kind(0) == Some(TokenType::Comma) {
                self.consume();
                elements.push(self.parse_ternary()?);
            }
            Ok(Expr { span, kind: ExprKind::Tuple(elements) })
        } else {
            Ok(first)
        }
    }

    fn parse_ternary(&mut self) -> Result<Expr, String> {
        let expr = self.parse_arrow()?;
        if self.peek_kind(0) == Some(TokenType::If) {
            let if_tok = self.consume();
            let condition = self.parse_arrow()?;
            self.expect(TokenType::Else, "Expected 'else' in ternary expression")?;
            let else_expr = self.parse_ternary()?;
            Ok(Expr {
                span: span_of(&if_tok),
                kind: ExprKind::Ternary {
                    then_expr: Box::new(expr),
                    condition: Box::new(condition),
                    else_expr: Box::new(else_expr),
                },
            })
        } else {
            Ok(expr)
        }
    }

    fn parse_arrow(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_logical_or()?;
        if self.peek_kind(0) != Some(TokenType::Arrow) {
            return Ok(lhs);
        }
        let span = self.peek(0).map(span_of).unwrap_or_default();
        let mut stages = Vec::new();
        while self.peek_kind(0) == Some(TokenType::Arrow) {
            let op = self.consume();
            let rhs = self.parse_logical_or()?;
            if count_stage_sites(&rhs) != 1 {
                return Err(self.error_token(
                    "RHS of '->' must contain exactly one callable stage site. Use 'stage()[n]' for repetition.",
                    &op,
                ));
            }
            stages.push(rhs);
        }
        Ok(Expr { span, kind: ExprKind::Arrow { source: Box::new(lhs), stages } })
    }

    fn parse_logical_or(&mut self) -> Result<Expr, String> {
        self.parse_left_assoc(Self::parse_logical_and, &[TokenType::PipePipe])
    }

    fn parse_logical_and(&mut self) -> Result<Expr, String> {
        self.parse_left_assoc(Self::parse_comparison, &[TokenType::AmpAmp])
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        self.parse_left_assoc(
            Self::parse_additive,
            &[TokenType::EqEq, TokenType::Neq, TokenType::Lt, TokenType::Gt, TokenType::LtEq, TokenType::GtEq],
        )
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        self.parse_left_assoc(Self::parse_term, &[TokenType::Plus, TokenType::Minus])
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        self.parse_left_assoc(Self::parse_unary, &[TokenType::Star, TokenType::Slash])
    }

    fn parse_left_assoc(
        &mut self,
        next: fn(&mut Self) -> Result<Expr, String>,
        operators: &[TokenType],
    ) -> Result<Expr, String> {
        let mut lhs = next(self)?;
        while let Some(kind) = self.peek_kind(0) {
            if !operators.contains(&kind) {
                break;
            }
            let op = self.consume();
            let rhs = next(self)?;
            lhs = Expr {
                span: span_of(&op),
                kind: ExprKind::Binary {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    op: op.kind,
                },
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek_kind(0) {
            Some(TokenType::Bang) | Some(TokenType::Minus) => {
                let op = self.consume();
                let operand = self.parse_unary()?;
                Ok(Expr { span: span_of(&op), kind: ExprKind::Unary { operand: Box::new(operand), op: op.kind } })
            }
            _ => self.parse_dot(),
        }
    }

    fn parse_dot(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_factor()?;
        while self.peek_kind(0) == Some(TokenType::Dot) {
            let op = self.consume();
            let ident_tok = self.peek(0).cloned().ok_or_else(|| self.error_here("Expected identifier after '.'"))?;
            let name = self.consume_ident("Expected identifier after '.'")?;
            let rhs = Expr {
                span: span_of(&ident_tok),
                kind: ExprKind::Identifier(name),
            };
            lhs = Expr {
                span: span_of(&op),
                kind: ExprKind::Binary {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    op: op.kind,
                },
            };
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        let mut result = match self.peek_kind(0) {
            Some(TokenType::IntLit) => {
                let tok = self.consume();
                Expr {
                    span: span_of(&tok),
                    kind: ExprKind::IntLit(expect_int(&tok)?),
                }
            }
            Some(TokenType::FloatLit) => {
                let tok = self.consume();
                Expr {
                    span: span_of(&tok),
                    kind: ExprKind::FloatLit(expect_float(&tok)?),
                }
            }
            Some(TokenType::True) => {
                let tok = self.consume();
                Expr { span: span_of(&tok), kind: ExprKind::BoolLit(true) }
            }
            Some(TokenType::False) => {
                let tok = self.consume();
                Expr { span: span_of(&tok), kind: ExprKind::BoolLit(false) }
            }
            Some(TokenType::StringLit) => {
                let tok = self.consume();
                Expr {
                    span: span_of(&tok),
                    kind: ExprKind::StringLit(expect_string(&tok)?.to_string()),
                }
            }
            Some(TokenType::Ident)
            | Some(TokenType::Config)
            | Some(TokenType::Int)
            | Some(TokenType::Float)
            | Some(TokenType::Bool)
            | Some(TokenType::Layer)
            | Some(TokenType::Fn)
            | Some(TokenType::Train) => {
                if self.peek_kind(1) == Some(TokenType::OpenParen) {
                    let tok = self.peek(0).cloned().ok_or_else(|| self.error_here("Expected callee"))?;
                    let name = self.consume_ident("Expected callee")?;
                    self.expect(TokenType::OpenParen, "Expected '('")?;
                    let mut args = Vec::new();
                    if self.peek_kind(0) != Some(TokenType::CloseParen) {
                        loop {
                            let is_named = self.peek_kind(1) == Some(TokenType::Colon)
                                && self.peek_kind(0).map(is_ident_like).unwrap_or(false);
                            let arg_name = if is_named {
                                let name = self.consume_ident("Expected named argument")?;
                                self.expect(TokenType::Colon, "Expected ':' after named argument")?;
                                Some(name)
                            } else {
                                None
                            };
                            let value = self.parse_ternary()?;
                            args.push(CallArgument { name: arg_name, value });
                            if self.peek_kind(0) == Some(TokenType::Comma) {
                                self.consume();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(TokenType::CloseParen, "Expected ')' after function call arguments")?;
                    Expr { span: span_of(&tok), kind: ExprKind::Call { callee: name, args } }
                } else {
                    let tok = self.peek(0).cloned().ok_or_else(|| self.error_here("Expected identifier"))?;
                    let name = self.consume_ident("Expected identifier")?;
                    Expr { span: span_of(&tok), kind: ExprKind::Identifier(name) }
                }
            }
            Some(TokenType::OpenParen) => {
                self.consume();
                let expr = self.parse_expr()?;
                self.expect(TokenType::CloseParen, "Expected ')'")?;
                expr
            }
            _ => return Err(self.error_here("Expected expression")),
        };

        while self.peek_kind(0) == Some(TokenType::OpenBracket) {
            if !matches!(result.kind, ExprKind::Call { .. }) {
                let token = self.peek(0).cloned().unwrap_or_default();
                return Err(self.error_token("Repeat suffix '[n]' must follow a call expression", &token));
            }
            let bracket = self.consume();
            let repeat_count = self.parse_ternary()?;
            self.expect(TokenType::CloseBracket, "Expected ']' after repetition count")?;
            result = Expr {
                span: span_of(&bracket),
                kind: ExprKind::Binary {
                    lhs: Box::new(result),
                    rhs: Box::new(repeat_count),
                    op: TokenType::StarEq,
                },
            };
        }

        Ok(result)
    }

    fn parse_scope(&mut self) -> Result<Stmt, String> {
        while self.peek_kind(0) == Some(TokenType::Newline) {
            self.consume();
        }
        let indent = self.expect(TokenType::Indent, "Expected indentation for block")?;
        let mut stmts = Vec::new();
        while self.peek_kind(0) != Some(TokenType::Dedent) {
            if self.peek_kind(0) == Some(TokenType::Newline) {
                self.consume();
                continue;
            }
            stmts.push(self.parse_stmt()?);
        }
        self.expect(TokenType::Dedent, "Expected DEDENT")?;
        Ok(Stmt { span: span_of(&indent), kind: StmtKind::Scope(stmts) })
    }

    fn parse_type(&mut self) -> Result<Type, String> {
        let token = self.peek(0).cloned().ok_or_else(|| self.error_here("Expected type"))?;
        match token.kind {
            TokenType::Int => Err(self.error_token(
                "Use an explicit integer width such as int16, int32, or int64",
                &token,
            )),
            TokenType::Bool => {
                self.consume();
                Ok(Type::bool())
            }
            TokenType::Float => Err(self.error_token(
                "Use an explicit float width such as float16, float32, or float64",
                &token,
            )),
            TokenType::Ident => {
                let name = expect_string(&token)?;
                match name {
                    "int16" => {
                        self.consume();
                        Ok(Type::int16())
                    }
                    "int32" => {
                        self.consume();
                        Ok(Type::int32())
                    }
                    "int64" => {
                        self.consume();
                        Ok(Type::int64())
                    }
                    "float16" => {
                        self.consume();
                        Ok(Type::float16())
                    }
                    "float32" => {
                        self.consume();
                        Ok(Type::float32())
                    }
                    "float64" => {
                        self.consume();
                        Ok(Type::float64())
                    }
                    _ => Err(self.error_token("Expected type", &token)),
                }
            }
            TokenType::Tensor => {
                self.consume();
                if self.peek_kind(0) != Some(TokenType::OpenBracket) {
                    return Ok(Type::tensor(None, None, None));
                }
                self.consume();
                let first = self.parse_type_token_sequence(&[TokenType::Comma, TokenType::CloseBracket])?;
                let mut dtype = None;
                let mut shape_expr = None;
                if self.peek_kind(0) == Some(TokenType::Comma) {
                    dtype = if first.is_empty() { None } else { Some(first) };
                    self.consume();
                    shape_expr = Some(self.parse_type_token_sequence(&[TokenType::CloseBracket])?);
                } else if !first.is_empty() {
                    dtype = Some(first);
                }
                self.expect(TokenType::CloseBracket, "Expected ']' after tensor type annotation")?;
                let rank = shape_expr.as_ref().and_then(|shape| derive_rank_from_shape_expr(shape));
                Ok(Type::tensor(dtype, shape_expr, rank))
            }
            TokenType::OpenParen => {
                self.consume();
                let mut elements = vec![self.parse_type()?];
                while self.peek_kind(0) == Some(TokenType::Comma) {
                    self.consume();
                    elements.push(self.parse_type()?);
                }
                self.expect(TokenType::CloseParen, "Expected ')' after tuple type")?;
                Ok(Type::tuple(elements))
            }
            _ => Err(self.error_token("Expected type", &token)),
        }
    }

    fn parse_type_token_sequence(&mut self, terminators: &[TokenType]) -> Result<String, String> {
        let mut result = String::new();
        let mut bracket_depth = 0i32;
        while let Some(token) = self.peek(0).cloned() {
            if bracket_depth == 0 && terminators.contains(&token.kind) {
                break;
            }
            self.consume();
            match token.kind {
                TokenType::Ident | TokenType::StringLit => result.push_str(expect_string(&token)?),
                TokenType::IntLit => result.push_str(&expect_int(&token)?.to_string()),
                TokenType::FloatLit => result.push_str(&expect_float(&token)?.to_string()),
                TokenType::Int => result.push_str("int"),
                TokenType::Float => result.push_str("float"),
                TokenType::Bool => result.push_str("bool"),
                TokenType::Tensor => result.push_str("tensor"),
                TokenType::Colon => result.push(':'),
                TokenType::Comma => result.push_str(", "),
                TokenType::Dot => result.push('.'),
                TokenType::Plus => result.push('+'),
                TokenType::Minus => result.push('-'),
                TokenType::Star => result.push('*'),
                TokenType::Slash => result.push('/'),
                TokenType::OpenBracket => {
                    result.push('[');
                    bracket_depth += 1;
                }
                TokenType::CloseBracket => {
                    result.push(']');
                    bracket_depth -= 1;
                }
                _ => return Err(self.error_token("Unsupported token in tensor type annotation", &token)),
            }
        }
        Ok(result)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        let token = self.peek(0).cloned().ok_or_else(|| self.error_here("Expected statement"))?;
        match token.kind {
            TokenType::Return => {
                self.consume();
                let expr = self.parse_expr()?;
                self.consume_terminator()?;
                Ok(Stmt { span: span_of(&token), kind: StmtKind::Return(expr) })
            }
            TokenType::Ident => {
                if self.peek_kind(1) == Some(TokenType::Colon) {
                    let name = self.consume_ident("Expected variable name")?;
                    self.consume();
                    let ty = self.parse_type()?;
                    let init = if self.peek_kind(0) == Some(TokenType::Eq) {
                        self.consume();
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    self.consume_terminator()?;
                    Ok(Stmt {
                        span: span_of(&token),
                        kind: StmtKind::VarDecl(VarDecl { name, ty, init, array_size: None }),
                    })
                } else if self.peek_kind(1) == Some(TokenType::Eq) {
                    let name = self.consume_ident("Expected assignment target")?;
                    self.consume();
                    let value = self.parse_expr()?;
                    self.consume_terminator()?;
                    Ok(Stmt { span: span_of(&token), kind: StmtKind::Assign(AssignStmt { name, value }) })
                } else if self.peek_kind(1) == Some(TokenType::OpenParen) {
                    let expr = self.parse_expr()?;
                    self.consume_terminator()?;
                    Ok(Stmt { span: span_of(&token), kind: StmtKind::Expr(expr) })
                } else {
                    Err(self.error_token("Unexpected identifier or missing assignment.", &token))
                }
            }
            TokenType::Indent => self.parse_scope(),
            TokenType::If => {
                self.consume();
                let condition = self.parse_expr()?;
                self.expect(TokenType::Colon, "Expected ':' after if condition")?;
                let then_stmt = Box::new(self.parse_scope()?);
                let mut elifs = Vec::new();
                while self.peek_kind(0) == Some(TokenType::Elif) {
                    self.consume();
                    let elif_cond = self.parse_expr()?;
                    self.expect(TokenType::Colon, "Expected ':' after elif condition")?;
                    let elif_stmt = self.parse_scope()?;
                    elifs.push(IfBranch { condition: elif_cond, body: Box::new(elif_stmt) });
                }
                let else_stmt = if self.peek_kind(0) == Some(TokenType::Else) {
                    self.consume();
                    if self.peek_kind(0) == Some(TokenType::Colon) {
                        self.consume();
                    }
                    Some(Box::new(self.parse_scope()?))
                } else {
                    None
                };
                Ok(Stmt {
                    span: span_of(&token),
                    kind: StmtKind::If { condition, then_stmt, elifs, else_stmt },
                })
            }
            _ => Err(self.error_token("Unexpected token in statement", &token)),
        }
    }

    fn parse_layer(&mut self) -> Result<Layer, String> {
        let start = self.expect(TokenType::Layer, "Expected 'layer' keyword")?;
        let name = self.consume_ident("Expected layer name")?;
        let args = self.parse_callable_args()?;
        let return_type = self.parse_callable_return_type("Expected ':' after layer signature")?;
        let body = self.parse_scope()?;
        Ok(Layer { span: span_of(&start), name, args, return_type, body })
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        let start = self.expect(TokenType::Fn, "Expected 'fn' keyword")?;
        let name = self.consume_ident("Expected function name")?;
        let args = self.parse_callable_args()?;
        let return_type = self.parse_callable_return_type("Expected ':' after function signature")?;
        let body = self.parse_scope()?;
        Ok(Function { span: span_of(&start), name, args, return_type, body })
    }

    fn parse_config(&mut self) -> Result<Config, String> {
        let start = self.expect(TokenType::Config, "Expected 'config' keyword")?;
        let name = self.consume_ident("Expected config name")?;
        self.expect(TokenType::Colon, "Expected ':' after config name")?;
        while self.peek_kind(0) == Some(TokenType::Newline) {
            self.consume();
        }
        let mut fields = Vec::new();
        if self.peek_kind(0) == Some(TokenType::Indent) {
            self.consume();
            while self.peek_kind(0) != Some(TokenType::Dedent) {
                if self.peek_kind(0) == Some(TokenType::Newline) {
                    self.consume();
                    continue;
                }
                let field_name = self.consume_ident("Expected field name")?;
                let ty = if self.peek_kind(0) == Some(TokenType::Colon) {
                    self.consume();
                    self.parse_type()?
                } else {
                    Type::void()
                };
                let init = if self.peek_kind(0) == Some(TokenType::Eq) {
                    self.consume();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                fields.push(Field { name: field_name, ty, init });
                self.consume_terminator()?;
            }
            self.expect(TokenType::Dedent, "Expected DEDENT")?;
        }
        Ok(Config { span: span_of(&start), name, fields })
    }

    fn parse_train(&mut self) -> Result<Train, String> {
        let start = self.expect(TokenType::Train, "Expected 'train' keyword")?;
        let name = self.consume_ident("Expected train name")?;
        self.expect(TokenType::Colon, "Expected ':' after train target")?;
        while self.peek_kind(0) == Some(TokenType::Newline) {
            self.consume();
        }
        let mut fields = Vec::new();
        if self.peek_kind(0) == Some(TokenType::Indent) {
            self.consume();
            while self.peek_kind(0) != Some(TokenType::Dedent) {
                if self.peek_kind(0) == Some(TokenType::Newline) {
                    self.consume();
                    continue;
                }
                let prop_name = self.consume_ident("Expected property name")?;
                let init = if self.peek_kind(0) == Some(TokenType::Eq) {
                    self.consume();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                fields.push(Field { name: prop_name, ty: Type::void(), init });
                self.consume_terminator()?;
            }
            self.expect(TokenType::Dedent, "Expected DEDENT")?;
        }
        Ok(Train { span: span_of(&start), name, fields })
    }

    fn parse_callable_args(&mut self) -> Result<Vec<Arg>, String> {
        self.expect(TokenType::OpenParen, "Expected '('")?;
        let mut args = Vec::new();
        if self.peek_kind(0) != Some(TokenType::CloseParen) {
            loop {
                let arg_name = self.consume_ident("Expected argument name")?;
                let ty = if self.peek_kind(0) == Some(TokenType::Colon) {
                    self.consume();
                    self.parse_type()?
                } else {
                    Type::void()
                };
                let default_value = if self.peek_kind(0) == Some(TokenType::Eq) {
                    self.consume();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                args.push(Arg { name: arg_name, ty, default_value });
                if self.peek_kind(0) == Some(TokenType::Comma) {
                    self.consume();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenType::CloseParen, "Expected ')'")?;
        Ok(args)
    }

    fn parse_callable_return_type(&mut self, colon_error: &str) -> Result<Type, String> {
        self.expect(TokenType::Colon, colon_error)?;
        if matches!(self.peek_kind(0), Some(TokenType::Newline | TokenType::Indent)) {
            return Ok(Type::void());
        }
        let return_type = self.parse_type()?;
        self.expect(TokenType::Colon, "Expected ':' after return type")?;
        Ok(return_type)
    }

    fn consume_ident(&mut self, error: &str) -> Result<String, String> {
        let token = self.consume();
        match token.kind {
            TokenType::Ident => Ok(expect_string(&token)?.to_string()),
            TokenType::Config => Ok("config".to_string()),
            TokenType::True => Ok("true".to_string()),
            TokenType::False => Ok("false".to_string()),
            TokenType::Int => Ok("int".to_string()),
            TokenType::Float => Ok("float".to_string()),
            TokenType::Bool => Ok("bool".to_string()),
            TokenType::Layer => Ok("layer".to_string()),
            TokenType::Fn => Ok("fn".to_string()),
            TokenType::Train => Ok("train".to_string()),
            TokenType::If => Ok("if".to_string()),
            TokenType::Else => Ok("else".to_string()),
            TokenType::Elif => Ok("elif".to_string()),
            TokenType::Return => Ok("return".to_string()),
            _ => Err(self.error_token(error, &token)),
        }
    }

    fn consume_terminator(&mut self) -> Result<(), String> {
        let mut found = false;
        while matches!(self.peek_kind(0), Some(TokenType::Semi | TokenType::Newline)) {
            self.consume();
            found = true;
        }
        if !found && !matches!(self.peek_kind(0), Some(TokenType::Dedent | TokenType::Eof)) {
            return Err(self.error_here("Expected ';' or newline at end of statement"));
        }
        Ok(())
    }

    fn peek(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.index + offset)
    }

    fn peek_kind(&self, offset: usize) -> Option<TokenType> {
        self.peek(offset).map(|token| token.kind)
    }

    fn consume(&mut self) -> Token {
        let token = self.tokens[self.index].clone();
        self.index += 1;
        token
    }

    fn expect(&mut self, kind: TokenType, error: &str) -> Result<Token, String> {
        let token = self.peek(0).cloned().ok_or_else(|| self.error_here(error))?;
        if token.kind != kind {
            return Err(self.error_token(error, &token));
        }
        Ok(self.consume())
    }

    fn error_token(&self, message: &str, token: &Token) -> String {
        format!("Parser Error: {message} at {}:{}", token.line, token.column)
    }

    fn error_here(&self, message: &str) -> String {
        if let Some(token) = self.peek(0) {
            self.error_token(message, token)
        } else if let Some(last) = self.tokens.last() {
            format!("Parser Error: {message} at end of file (after {}:{})", last.line, last.column)
        } else {
            format!("Parser Error: {message} at start of file")
        }
    }
}

fn span_of(token: &Token) -> SourceSpan {
    SourceSpan {
        line: token.line,
        column: token.column,
    }
}

fn expect_int(token: &Token) -> Result<i64, String> {
    match &token.value {
        Some(TokenValue::Int(value)) => Ok(*value),
        _ => Err(format!("Expected int literal at {}:{}", token.line, token.column)),
    }
}

fn expect_float(token: &Token) -> Result<f64, String> {
    match &token.value {
        Some(TokenValue::Float(value)) => Ok(*value),
        _ => Err(format!("Expected float literal at {}:{}", token.line, token.column)),
    }
}

fn expect_string(token: &Token) -> Result<&str, String> {
    match &token.value {
        Some(TokenValue::String(value)) => Ok(value),
        _ => Err(format!("Expected string token at {}:{}", token.line, token.column)),
    }
}

fn is_ident_like(kind: TokenType) -> bool {
    matches!(
        kind,
        TokenType::Ident
            | TokenType::Config
            | TokenType::True
            | TokenType::False
            | TokenType::Int
            | TokenType::Float
            | TokenType::Bool
            | TokenType::Layer
            | TokenType::Fn
            | TokenType::Train
            | TokenType::If
            | TokenType::Else
            | TokenType::Elif
            | TokenType::Return
    )
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

fn derive_rank_from_shape_expr(shape_expr: &str) -> Option<usize> {
    let colon = shape_expr.find(':')?;
    let close = shape_expr.rfind(']')?;
    let open = shape_expr.rfind('[')?;
    if colon < open || colon > close {
        return None;
    }
    let start = &shape_expr[open + 1..colon];
    let end = &shape_expr[colon + 1..close];
    if end.is_empty() {
        return None;
    }
    let start_value = if start.is_empty() { 0 } else { start.parse::<i32>().ok()? };
    let end_value = end.parse::<i32>().ok()?;
    (end_value - start_value).try_into().ok()
}
