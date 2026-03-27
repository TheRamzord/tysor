use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenType {
    Return,
    Int,
    Float,
    Bool,
    Tensor,
    Tuple,
    True,
    False,
    Layer,
    Fn,
    Train,
    Config,
    If,
    Elif,
    Else,
    IntLit,
    FloatLit,
    BoolLit,
    StringLit,
    Ident,
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    EqEq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Amp,
    AmpAmp,
    PipePipe,
    Bang,
    Eq,
    Hash,
    Underscore,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    Semi,
    Colon,
    Comma,
    Dot,
    DotDot,
    Pipe,
    Arrow,
    StarEq,
    Indent,
    Dedent,
    Newline,
    #[default]
    Eof,
}

impl TokenType {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenType::Return => "RETURN",
            TokenType::Int => "INT",
            TokenType::Float => "FLOAT",
            TokenType::Bool => "BOOL",
            TokenType::Tensor => "TENSOR",
            TokenType::Tuple => "TUPLE",
            TokenType::True => "TRUE",
            TokenType::False => "FALSE",
            TokenType::Layer => "LAYER",
            TokenType::Fn => "FN",
            TokenType::Train => "TRAIN",
            TokenType::Config => "CONFIG",
            TokenType::If => "IF",
            TokenType::Elif => "ELIF",
            TokenType::Else => "ELSE",
            TokenType::IntLit => "INT_LIT",
            TokenType::FloatLit => "FLOAT_LIT",
            TokenType::BoolLit => "BOOL_LIT",
            TokenType::StringLit => "STRING_LIT",
            TokenType::Ident => "IDENT",
            TokenType::Plus => "PLUS",
            TokenType::Minus => "MINUS",
            TokenType::Star => "STAR",
            TokenType::Slash => "SLASH",
            TokenType::DoubleSlash => "DOUBLE_SLASH",
            TokenType::EqEq => "EQ_EQ",
            TokenType::Neq => "NEQ",
            TokenType::Lt => "LT",
            TokenType::Gt => "GT",
            TokenType::LtEq => "LT_EQ",
            TokenType::GtEq => "GT_EQ",
            TokenType::Amp => "AMP",
            TokenType::AmpAmp => "AMP_AMP",
            TokenType::PipePipe => "PIPE_PIPE",
            TokenType::Bang => "BANG",
            TokenType::Eq => "EQUALS",
            TokenType::Hash => "HASH",
            TokenType::Underscore => "UNDERSCORE",
            TokenType::OpenParen => "OPEN_PAREN",
            TokenType::CloseParen => "CLOSE_PAREN",
            TokenType::OpenBracket => "OPEN_BRACKET",
            TokenType::CloseBracket => "CLOSE_BRACKET",
            TokenType::Semi => "SEMI",
            TokenType::Colon => "COLON",
            TokenType::Comma => "COMMA",
            TokenType::Dot => "DOT",
            TokenType::DotDot => "DOT_DOT",
            TokenType::Pipe => "PIPE",
            TokenType::Arrow => "ARROW",
            TokenType::StarEq => "STAR_EQ",
            TokenType::Indent => "INDENT",
            TokenType::Dedent => "DEDENT",
            TokenType::Newline => "NEWLINE",
            TokenType::Eof => "EOF",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Token {
    pub kind: TokenType,
    pub value: Option<TokenValue>,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            Some(TokenValue::Int(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            Some(TokenValue::Float(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            Some(TokenValue::String(value)) => {
                write!(
                    f,
                    "{}({value}) ({}:{})",
                    self.kind.as_str(),
                    self.line,
                    self.column
                )
            }
            None => write!(f, "{} ({}:{})", self.kind.as_str(), self.line, self.column),
        }
    }
}

fn push_token(
    tokens: &mut Vec<Token>,
    kind: TokenType,
    value: Option<TokenValue>,
    line: usize,
    column: usize,
) {
    tokens.push(Token {
        kind,
        value,
        line,
        column,
    });
}

pub fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut line = 1usize;
    let mut col = 1usize;
    let mut indent_stack = vec![0usize];
    let mut start_of_line = true;
    let mut paren_level = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        if start_of_line {
            let mut current_indent = 0usize;
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                if chars[i] == ' ' {
                    current_indent += 1;
                } else {
                    current_indent = (current_indent / 8 + 1) * 8;
                }
                i += 1;
            }

            if i < chars.len()
                && chars[i] != '\n'
                && !(chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/')
            {
                let is_continuation = chars[i] == '-' && i + 1 < chars.len() && chars[i + 1] == '>';
                if paren_level == 0 && !is_continuation {
                    if current_indent > *indent_stack.last().unwrap() {
                        indent_stack.push(current_indent);
                        push_token(
                            &mut tokens,
                            TokenType::Indent,
                            None,
                            line,
                            current_indent + 1,
                        );
                    } else {
                        while current_indent < *indent_stack.last().unwrap() {
                            indent_stack.pop();
                            push_token(
                                &mut tokens,
                                TokenType::Dedent,
                                None,
                                line,
                                current_indent + 1,
                            );
                        }
                        if current_indent != *indent_stack.last().unwrap() {
                            return Err(format!(
                                "Indentation error at {line}:{}",
                                current_indent + 1
                            ));
                        }
                    }
                }
            }

            col = current_indent + 1;
            start_of_line = false;
        }

        if i >= chars.len() {
            break;
        }

        let c = chars[i];
        let start_col = col;

        if c.is_ascii_alphabetic() {
            let mut buf = String::new();
            buf.push(c);
            while i + 1 < chars.len()
                && (chars[i + 1].is_ascii_alphanumeric() || chars[i + 1] == '_')
            {
                i += 1;
                col += 1;
                buf.push(chars[i]);
            }

            match buf.as_str() {
                "return" => push_token(&mut tokens, TokenType::Return, None, line, start_col),
                "int" => push_token(&mut tokens, TokenType::Int, None, line, start_col),
                "bool" => push_token(&mut tokens, TokenType::Bool, None, line, start_col),
                "float" => push_token(&mut tokens, TokenType::Float, None, line, start_col),
                "tensor" => push_token(&mut tokens, TokenType::Tensor, None, line, start_col),
                "tuple" => push_token(&mut tokens, TokenType::Tuple, None, line, start_col),
                "true" => push_token(&mut tokens, TokenType::True, None, line, start_col),
                "false" => push_token(&mut tokens, TokenType::False, None, line, start_col),
                "layer" => push_token(&mut tokens, TokenType::Layer, None, line, start_col),
                "fn" => push_token(&mut tokens, TokenType::Fn, None, line, start_col),
                "train" => push_token(&mut tokens, TokenType::Train, None, line, start_col),
                "config" => push_token(&mut tokens, TokenType::Config, None, line, start_col),
                "if" => push_token(&mut tokens, TokenType::If, None, line, start_col),
                "elif" => push_token(&mut tokens, TokenType::Elif, None, line, start_col),
                "else" => push_token(&mut tokens, TokenType::Else, None, line, start_col),
                "None" => push_token(
                    &mut tokens,
                    TokenType::Ident,
                    Some(TokenValue::String("None".to_string())),
                    line,
                    start_col,
                ),
                _ => push_token(
                    &mut tokens,
                    TokenType::Ident,
                    Some(TokenValue::String(buf)),
                    line,
                    start_col,
                ),
            }
            i += 1;
            col += 1;
            continue;
        }

        if c == '"' {
            let mut buf = String::new();
            i += 1;
            col += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\n' {
                    return Err(format!("Unterminated string literal at {line}:{col}"));
                }
                buf.push(chars[i]);
                i += 1;
                col += 1;
            }
            if i >= chars.len() {
                return Err(format!("Unterminated string literal at {line}:{col}"));
            }
            push_token(
                &mut tokens,
                TokenType::StringLit,
                Some(TokenValue::String(buf)),
                line,
                start_col,
            );
            i += 1;
            col += 1;
            continue;
        }

        if c.is_ascii_digit() {
            let mut buf = String::new();
            buf.push(c);
            while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                i += 1;
                col += 1;
                buf.push(chars[i]);
            }
            let mut is_float = false;
            if i + 1 < chars.len() && chars[i + 1] == '.' {
                is_float = true;
                i += 1;
                col += 1;
                buf.push(chars[i]);
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
            }
            if i + 1 < chars.len() && (chars[i + 1] == 'e' || chars[i + 1] == 'E') {
                is_float = true;
                i += 1;
                col += 1;
                buf.push(chars[i]);
                if i + 1 < chars.len() && (chars[i + 1] == '+' || chars[i + 1] == '-') {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    col += 1;
                    buf.push(chars[i]);
                }
            }
            if is_float {
                let value = buf
                    .parse::<f64>()
                    .map_err(|err| format!("Invalid float literal '{buf}': {err}"))?;
                push_token(
                    &mut tokens,
                    TokenType::FloatLit,
                    Some(TokenValue::Float(value)),
                    line,
                    start_col,
                );
            } else {
                let value = buf
                    .parse::<i64>()
                    .map_err(|err| format!("Invalid int literal '{buf}': {err}"))?;
                push_token(
                    &mut tokens,
                    TokenType::IntLit,
                    Some(TokenValue::Int(value)),
                    line,
                    start_col,
                );
            }
            i += 1;
            col += 1;
            continue;
        }

        match c {
            ';' => {
                push_token(&mut tokens, TokenType::Semi, None, line, col);
                i += 1;
                col += 1;
            }
            ':' => {
                push_token(&mut tokens, TokenType::Colon, None, line, col);
                i += 1;
                col += 1;
            }
            '.' => {
                push_token(&mut tokens, TokenType::Dot, None, line, col);
                i += 1;
                col += 1;
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::EqEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Eq, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::Neq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Bang, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    push_token(&mut tokens, TokenType::AmpAmp, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Amp, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    push_token(&mut tokens, TokenType::PipePipe, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Pipe, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::LtEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Lt, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    push_token(&mut tokens, TokenType::GtEq, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Gt, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '+' => {
                push_token(&mut tokens, TokenType::Plus, None, line, col);
                i += 1;
                col += 1;
            }
            '-' => {
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    push_token(&mut tokens, TokenType::Arrow, None, line, col);
                    i += 2;
                    col += 2;
                } else {
                    push_token(&mut tokens, TokenType::Minus, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '*' => {
                push_token(&mut tokens, TokenType::Star, None, line, col);
                i += 1;
                col += 1;
            }
            '/' => {
                if i + 1 < chars.len() && chars[i + 1] == '/' {
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                        col += 1;
                    }
                } else {
                    push_token(&mut tokens, TokenType::Slash, None, line, col);
                    i += 1;
                    col += 1;
                }
            }
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
            }
            '(' => {
                paren_level += 1;
                push_token(&mut tokens, TokenType::OpenParen, None, line, col);
                i += 1;
                col += 1;
            }
            ')' => {
                if paren_level > 0 {
                    paren_level -= 1;
                }
                push_token(&mut tokens, TokenType::CloseParen, None, line, col);
                i += 1;
                col += 1;
            }
            '[' => {
                paren_level += 1;
                push_token(&mut tokens, TokenType::OpenBracket, None, line, col);
                i += 1;
                col += 1;
            }
            ']' => {
                if paren_level > 0 {
                    paren_level -= 1;
                }
                push_token(&mut tokens, TokenType::CloseBracket, None, line, col);
                i += 1;
                col += 1;
            }
            ',' => {
                push_token(&mut tokens, TokenType::Comma, None, line, col);
                i += 1;
                col += 1;
            }
            _ if c.is_whitespace() => {
                if c == '\n' {
                    if paren_level == 0 {
                        let mut next_is_continuation = false;
                        let mut j = i + 1;
                        while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                            j += 1;
                        }
                        if j + 1 < chars.len() && chars[j] == '-' && chars[j + 1] == '>' {
                            next_is_continuation = true;
                        }
                        if !next_is_continuation {
                            push_token(&mut tokens, TokenType::Newline, None, line, col);
                        }
                    }
                    line += 1;
                    col = 1;
                    start_of_line = true;
                } else {
                    col += 1;
                }
                i += 1;
            }
            _ => return Err(format!("Unknown character '{c}' at {line}:{col}")),
        }
    }

    while indent_stack.len() > 1 {
        indent_stack.pop();
        push_token(&mut tokens, TokenType::Dedent, None, line, col);
    }
    push_token(&mut tokens, TokenType::Eof, None, line, col);
    Ok(tokens)
}
