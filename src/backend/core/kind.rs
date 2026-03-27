use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    Local,
    Cuda,
    Metal,
    PyTorch,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Local => "local",
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
            BackendKind::PyTorch => "pytorch",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "local" => Ok(BackendKind::Local),
            "cuda" => Ok(BackendKind::Cuda),
            "metal" => Ok(BackendKind::Metal),
            "pytorch" => Ok(BackendKind::PyTorch),
            other => Err(format!("unsupported backend '{other}'")),
        }
    }
}
