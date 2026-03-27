use crate::ops::library::RuntimePrimitiveKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeBuiltinKind {
    #[default]
    Unsupported,
    Matmul,
    Relu,
    Scale,
}

pub fn runtime_builtin(name: &str) -> RuntimeBuiltinKind {
    match crate::ops::library::runtime_primitive(name) {
        RuntimePrimitiveKind::Matmul => RuntimeBuiltinKind::Matmul,
        RuntimePrimitiveKind::Relu => RuntimeBuiltinKind::Relu,
        RuntimePrimitiveKind::Scale => RuntimeBuiltinKind::Scale,
        RuntimePrimitiveKind::Unsupported => RuntimeBuiltinKind::Unsupported,
    }
}
