use crate::compiler::parser::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimePrimitiveKind {
    #[default]
    Unsupported,
    Matmul,
    Relu,
    Scale,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinSignature {
    pub name: String,
    pub return_type: Type,
    pub arg_types: Vec<Type>,
    pub min_arity: usize,
    pub max_arity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpDefinition {
    pub signature: BuiltinSignature,
    pub is_primitive_tensor_op: bool,
    pub is_library_op: bool,
    pub is_callable_library_op: bool,
    pub preserves_first_tensor_arg: bool,
    pub runtime_supports_library_op: bool,
    pub runtime_primitive_kind: RuntimePrimitiveKind,
}

fn core_ops() -> Vec<OpDefinition> {
    vec![
        OpDefinition {
            signature: BuiltinSignature {
                name: "linear".to_string(),
                return_type: Type::callable(Type::tensor(None, None, None)),
                arg_types: vec![Type::int(), Type::int(), Type::bool()],
                min_arity: 1,
                max_arity: 3,
            },
            is_primitive_tensor_op: false,
            is_library_op: true,
            is_callable_library_op: true,
            preserves_first_tensor_arg: false,
            runtime_supports_library_op: true,
            runtime_primitive_kind: RuntimePrimitiveKind::Unsupported,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "matmul".to_string(),
                return_type: Type::tensor(None, None, None),
                arg_types: vec![Type::tensor(None, None, None), Type::tensor(None, None, None)],
                min_arity: 2,
                max_arity: 2,
            },
            is_primitive_tensor_op: true,
            is_library_op: false,
            is_callable_library_op: false,
            preserves_first_tensor_arg: false,
            runtime_supports_library_op: false,
            runtime_primitive_kind: RuntimePrimitiveKind::Matmul,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "relu".to_string(),
                return_type: Type::tensor(None, None, None),
                arg_types: vec![Type::tensor(None, None, None)],
                min_arity: 1,
                max_arity: 1,
            },
            is_primitive_tensor_op: true,
            is_library_op: false,
            is_callable_library_op: false,
            preserves_first_tensor_arg: true,
            runtime_supports_library_op: false,
            runtime_primitive_kind: RuntimePrimitiveKind::Relu,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "scale".to_string(),
                return_type: Type::tensor(None, None, None),
                arg_types: vec![Type::tensor(None, None, None), Type::float()],
                min_arity: 2,
                max_arity: 2,
            },
            is_primitive_tensor_op: true,
            is_library_op: true,
            is_callable_library_op: false,
            preserves_first_tensor_arg: true,
            runtime_supports_library_op: false,
            runtime_primitive_kind: RuntimePrimitiveKind::Scale,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "SiLU".to_string(),
                return_type: Type::callable(Type::tensor(None, None, None)),
                arg_types: vec![],
                min_arity: 0,
                max_arity: 0,
            },
            is_primitive_tensor_op: false,
            is_library_op: true,
            is_callable_library_op: true,
            preserves_first_tensor_arg: false,
            runtime_supports_library_op: true,
            runtime_primitive_kind: RuntimePrimitiveKind::Unsupported,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "Softmax".to_string(),
                return_type: Type::callable(Type::tensor(None, None, None)),
                arg_types: vec![],
                min_arity: 0,
                max_arity: 0,
            },
            is_primitive_tensor_op: false,
            is_library_op: true,
            is_callable_library_op: true,
            preserves_first_tensor_arg: false,
            runtime_supports_library_op: true,
            runtime_primitive_kind: RuntimePrimitiveKind::Unsupported,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "rms_norm".to_string(),
                return_type: Type::tensor(None, None, None),
                arg_types: vec![Type::tensor(None, None, None), Type::int()],
                min_arity: 2,
                max_arity: 2,
            },
            is_primitive_tensor_op: false,
            is_library_op: true,
            is_callable_library_op: false,
            preserves_first_tensor_arg: true,
            runtime_supports_library_op: false,
            runtime_primitive_kind: RuntimePrimitiveKind::Unsupported,
        },
        OpDefinition {
            signature: BuiltinSignature {
                name: "cross_entropy".to_string(),
                return_type: Type::tensor(None, None, None),
                arg_types: vec![Type::tensor(None, None, None), Type::tensor(None, None, None)],
                min_arity: 2,
                max_arity: 2,
            },
            is_primitive_tensor_op: false,
            is_library_op: true,
            is_callable_library_op: false,
            preserves_first_tensor_arg: false,
            runtime_supports_library_op: false,
            runtime_primitive_kind: RuntimePrimitiveKind::Unsupported,
        },
    ]
}

pub fn lookup_op(name: &str) -> Option<OpDefinition> {
    core_ops().into_iter().find(|op| op.signature.name == name)
}

pub fn core_library_builtin_signatures() -> Vec<BuiltinSignature> {
    core_ops().into_iter().map(|op| op.signature).collect()
}

pub fn is_primitive_tensor_op(name: &str) -> bool {
    lookup_op(name).map(|op| op.is_primitive_tensor_op).unwrap_or(false)
}

pub fn is_library_op(name: &str) -> bool {
    if let Some(op) = lookup_op(name) {
        return op.is_library_op;
    }
    matches!(
        name,
        "Embedding" | "Dropout" | "rope" | "reshape" | "causal_mask" | "flatten_heads" | "repeat_kv"
    )
}

pub fn is_callable_library_op(name: &str) -> bool {
    if let Some(op) = lookup_op(name) {
        return op.is_callable_library_op;
    }
    matches!(name, "Embedding" | "Dropout")
}

pub fn preserves_first_tensor_arg(name: &str) -> bool {
    if let Some(op) = lookup_op(name) {
        return op.preserves_first_tensor_arg;
    }
    matches!(name, "rope" | "repeat_kv" | "flatten_heads" | "causal_mask")
}

pub fn runtime_supports_library_op(name: &str) -> bool {
    lookup_op(name).map(|op| op.runtime_supports_library_op).unwrap_or(false)
}

pub fn runtime_primitive(name: &str) -> RuntimePrimitiveKind {
    lookup_op(name)
        .map(|op| op.runtime_primitive_kind)
        .unwrap_or(RuntimePrimitiveKind::Unsupported)
}
