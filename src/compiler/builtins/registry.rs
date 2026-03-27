use crate::compiler::parser::Type;
use crate::ops::library::{core_library_builtin_signatures, BuiltinSignature};

pub fn all_builtin_signatures() -> Vec<BuiltinSignature> {
    let mut signatures = core_library_builtin_signatures();
    signatures.extend([
        BuiltinSignature {
            name: "Embedding".to_string(),
            return_type: Type::callable(Type::tensor(None, None, None)),
            arg_types: vec![Type::int(), Type::int()],
            min_arity: 2,
            max_arity: 2,
        },
        BuiltinSignature {
            name: "Dropout".to_string(),
            return_type: Type::callable(Type::tensor(None, None, None)),
            arg_types: vec![Type::float()],
            min_arity: 1,
            max_arity: 1,
        },
        BuiltinSignature {
            name: "rope".to_string(),
            return_type: Type::tensor(None, None, None),
            arg_types: vec![Type::tensor(None, None, None), Type::int(), Type::float()],
            min_arity: 3,
            max_arity: 3,
        },
        BuiltinSignature {
            name: "reshape".to_string(),
            return_type: Type::tensor(None, None, None),
            arg_types: vec![
                Type::tensor(None, None, None),
                Type::int(),
                Type::int(),
                Type::int(),
                Type::int(),
                Type::int(),
                Type::int(),
                Type::int(),
            ],
            min_arity: 2,
            max_arity: 8,
        },
        BuiltinSignature {
            name: "causal_mask".to_string(),
            return_type: Type::tensor(None, None, None),
            arg_types: vec![Type::tensor(None, None, None)],
            min_arity: 1,
            max_arity: 1,
        },
        BuiltinSignature {
            name: "flatten_heads".to_string(),
            return_type: Type::tensor(None, None, None),
            arg_types: vec![Type::tensor(None, None, None)],
            min_arity: 1,
            max_arity: 1,
        },
        BuiltinSignature {
            name: "repeat_kv".to_string(),
            return_type: Type::tensor(None, None, None),
            arg_types: vec![Type::tensor(None, None, None), Type::int()],
            min_arity: 2,
            max_arity: 2,
        },
        BuiltinSignature {
            name: "print".to_string(),
            return_type: Type::void(),
            arg_types: vec![],
            min_arity: 0,
            max_arity: 0,
        },
    ]);
    signatures
}
