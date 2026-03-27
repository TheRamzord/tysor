pub fn is_layer_constructor(name: &str) -> bool {
    crate::ops::library::is_callable_library_op(name)
}

pub fn is_tensor_builtin(name: &str) -> bool {
    crate::ops::library::is_primitive_tensor_op(name) || crate::ops::library::is_library_op(name)
}

pub fn preserves_first_tensor_arg(name: &str) -> bool {
    crate::ops::library::preserves_first_tensor_arg(name)
}

pub fn runtime_supports_layer_constructor(name: &str) -> bool {
    crate::ops::library::runtime_supports_library_op(name)
}
