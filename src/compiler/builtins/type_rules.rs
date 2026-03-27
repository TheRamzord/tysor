use crate::compiler::frontend_ir::FeType;
use crate::compiler::frontend_ir::FeTypeKind;
use crate::compiler::parser::Type;
use crate::compiler::parser::TypeBase;
use crate::ops::model::preserves_first_tensor_arg;

pub fn infer_call_result_type(callee: &str, declared_type: &Type, arg_types: &[Type]) -> Type {
    if declared_type.base == TypeBase::Callable && !arg_types.is_empty() {
        if let Some(callable_return) = &declared_type.callable_return {
            if callable_return.base == TypeBase::Tensor && arg_types[0].base == TypeBase::Tensor {
                return Type::callable(Type::tensor(
                    arg_types[0].tensor_dtype.clone(),
                    callable_return.tensor_shape_expr.clone(),
                    arg_types[0].tensor_rank,
                ));
            }
        }
        return declared_type.clone();
    }

    if declared_type.base != TypeBase::Tensor || arg_types.is_empty() || arg_types[0].base != TypeBase::Tensor {
        return declared_type.clone();
    }

    let arg0 = &arg_types[0];
    if callee == "reshape" {
        return Type::tensor(arg0.tensor_dtype.clone(), None, None);
    }
    if callee == "matmul" {
        return Type::tensor(
            arg0.tensor_dtype.clone(),
            declared_type.tensor_shape_expr.clone(),
            arg0.tensor_rank.or(declared_type.tensor_rank),
        );
    }
    if callee == "cross_entropy" {
        return declared_type.clone();
    }
    if preserves_first_tensor_arg(callee) {
        return arg0.clone();
    }
    Type::tensor(
        arg0.tensor_dtype.clone(),
        declared_type.tensor_shape_expr.clone(),
        arg0.tensor_rank,
    )
}

pub fn infer_fe_call_result_type(callee: &str, declared_type: &FeType, arg_types: &[FeType]) -> FeType {
    if declared_type.kind == FeTypeKind::Callable && !arg_types.is_empty() {
        if let Some(callable_return) = &declared_type.callable_return {
            if callable_return.kind == FeTypeKind::Tensor && arg_types[0].kind == FeTypeKind::Tensor {
                let mut result = callable_return.as_ref().clone();
                result.tensor_dtype = arg_types[0].tensor_dtype.clone();
                result.tensor_rank = arg_types[0].tensor_rank;
                return FeType::callable(result);
            }
        }
        return declared_type.clone();
    }

    if declared_type.kind != FeTypeKind::Tensor || arg_types.is_empty() || arg_types[0].kind != FeTypeKind::Tensor {
        return declared_type.clone();
    }

    let arg0 = &arg_types[0];
    if callee == "reshape" {
        return FeType::tensor(arg0.tensor_dtype.clone(), None, None);
    }
    if callee == "matmul" {
        return FeType::tensor(
            arg0.tensor_dtype.clone(),
            declared_type.tensor_shape_expr.clone(),
            arg0.tensor_rank.or(declared_type.tensor_rank),
        );
    }
    if callee == "cross_entropy" {
        return declared_type.clone();
    }
    if preserves_first_tensor_arg(callee) {
        return arg0.clone();
    }
    let mut result = declared_type.clone();
    result.tensor_dtype = arg0.tensor_dtype.clone();
    result.tensor_rank = arg0.tensor_rank;
    result
}
