use regex::Regex;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const EXPECTED_FAILURE_NAMES: &[&str] = &[
    "test_fn_cannot_call_layer.ty",
    "test_fn_arrow_invalid.ty",
];

const PYTORCH_SUPPORTED_TESTS: &[&str] = &[
    "test_emit_pytorch_embedding.ty",
    "test_emit_pytorch_rope_repeat_kv.ty",
    "test_emit_pytorch_transformer_train.ty",
    "test_model_matmul_relu.ty",
    "test_run_pytorch_transformer_mask.ty",
    "test_run_rms_norm.ty",
    "test_run_dropout.ty",
    "test_train_pytorch_transformer_mask.ty",
    "test_train_small_mlp.ty",
    "test_train_silu_mlp.ty",
    "test_train_dropout.ty",
    "test_train_linear_bias.ty",
];

const PYTORCH_SUPPORTED_RUNTIME_TESTS: &[&str] = &[
    "test_model_matmul_relu.ty",
    "test_run_pytorch_transformer_mask.ty",
];

const METAL_SUPPORTED_RUNTIME_TESTS: &[&str] = &[
    "test_model_matmul_relu.ty",
    "test_run_activation_layers.ty",
    "test_run_binary_ops.ty",
    "test_run_cross_entropy.ty",
    "test_run_dropout.ty",
    "test_run_linear_bias.ty",
    "test_run_pytorch_transformer_mask.ty",
    "test_run_rms_norm.ty",
    "test_run_scale_builtin.ty",
    "test_run_two_layer_mlp.ty",
];

const METAL_SUPPORTED_BACKWARD_TESTS: &[&str] = &[
    "test_backward_basic_ops.ty",
    "test_backward_cross_entropy.ty",
    "test_backward_dropout.ty",
    "test_backward_linear_bias.ty",
    "test_backward_linear_layer.ty",
    "test_backward_rms_norm.ty",
    "test_backward_silu_layer.ty",
    "test_backward_softmax_layer.ty",
];

const METAL_SUPPORTED_TRAIN_TESTS: &[&str] = &[
    "test_train_backend_single.ty",
    "test_train_dropout.ty",
    "test_train_linear_bias.ty",
    "test_train_pytorch_transformer_mask.ty",
    "test_train_small_mlp.ty",
    "test_train_silu_mlp.ty",
];

const PYTORCH_SUPPORTED_TRAIN_TESTS: &[&str] = &[
    "test_train_dropout.ty",
    "test_train_linear_bias.ty",
    "test_train_pytorch_transformer.ty",
    "test_train_pytorch_transformer_mask.ty",
    "test_train_silu_mlp.ty",
    "test_train_small_mlp.ty",
];

const TOKENS_START: &str = "--- Tokenization Step ---";
const TOKENS_END: &str = "-------------------------";
const AST_START: &str = "--- Parsing Step ---";
const AST_END: &str = "--------------------";
const SEMANTICS_START: &str = "--- Semantic Analysis Step ---";
const SEMANTICS_END: &str = "------------------------------";
const IR_START: &str = "--- Lowered Frontend IR ---";
const IR_END: &str = "---------------------------";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Local,
    Cuda,
    Metal,
    PyTorch,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::PyTorch => "pytorch",
        }
    }
}

#[derive(Debug, Default)]
struct Args {
    files: Vec<String>,
    file_flags: Vec<String>,
    binary: PathBuf,
    tests_dir: PathBuf,
    stress: bool,
    emit_pytorch: bool,
    run: bool,
    backward: bool,
    train: bool,
    backend: Option<Backend>,
    compare_cpu: bool,
    show_output: bool,
    tokens: bool,
    ast: bool,
    ir: bool,
    semantics: bool,
    fail_fast: bool,
}

#[derive(Debug)]
struct TestResult {
    path: PathBuf,
    expected_failure: bool,
    passed: bool,
    returncode: i32,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone)]
struct TensorOutput {
    shape: Vec<i64>,
    dtype: String,
    values: Vec<f64>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn default_binary() -> PathBuf {
    repo_root().join("target").join("debug").join("tysor")
}

fn default_tests_dir() -> PathBuf {
    repo_root().join("tests")
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "local" => Ok(Backend::Local),
        "cuda" => Ok(Backend::Cuda),
        "metal" => Ok(Backend::Metal),
        "pytorch" => Ok(Backend::PyTorch),
        _ => Err(format!("unknown backend '{value}'")),
    }
}

fn usage() -> &'static str {
    "run_tests [files...] [--binary <path>] [--tests-dir <path>] [--file <file>] [--stress] [--emit-pytorch] [--run] [--backend <local|cuda|metal|pytorch>] [--compare-cpu] [--backward] [--train] [--show-output] [--tokens] [--ast] [--ir] [--semantics] [--fail-fast]"
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut parsed = Args {
        binary: default_binary(),
        tests_dir: default_tests_dir(),
        ..Args::default()
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--binary" => {
                parsed.binary = PathBuf::from(args.next().ok_or_else(|| "missing value for --binary".to_string())?);
            }
            "--tests-dir" => {
                parsed.tests_dir = PathBuf::from(args.next().ok_or_else(|| "missing value for --tests-dir".to_string())?);
            }
            "--file" => {
                parsed.file_flags.push(args.next().ok_or_else(|| "missing value for --file".to_string())?);
            }
            "--stress" => parsed.stress = true,
            "--emit-pytorch" => parsed.emit_pytorch = true,
            "--run" => parsed.run = true,
            "--backend" => {
                parsed.backend = Some(parse_backend(
                    &args.next().ok_or_else(|| "missing value for --backend".to_string())?,
                )?);
            }
            "--compare-cpu" => parsed.compare_cpu = true,
            "--backward" => parsed.backward = true,
            "--train" => parsed.train = true,
            "--show-output" => parsed.show_output = true,
            "--tokens" => parsed.tokens = true,
            "--ast" => parsed.ast = true,
            "--ir" => parsed.ir = true,
            "--semantics" => parsed.semantics = true,
            "--fail-fast" => parsed.fail_fast = true,
            _ if arg.starts_with("--") => return Err(format!("unknown option: {arg}")),
            _ => parsed.files.push(arg),
        }
    }

    Ok(parsed)
}

fn runtime_test_args() -> BTreeMap<&'static str, Vec<&'static str>> {
    BTreeMap::from([
        ("test_run_activation_layers.ty", vec!["--shape", "x=2x3"]),
        ("test_run_cross_entropy.ty", vec!["--shape", "logits=2x3", "--shape", "target=2x3"]),
        ("test_run_dropout.ty", vec!["--shape", "x=2x3"]),
        ("test_run_linear_bias.ty", vec!["--shape", "x=2x3"]),
        ("test_model_matmul_relu.ty", vec!["--shape", "x=2x3", "--shape", "w=3x4"]),
        ("test_run_pytorch_transformer_mask.ty", vec!["--shape", "idx=1"]),
        ("test_run_binary_ops.ty", vec!["--shape", "x=2x3", "--shape", "y=2x3"]),
        ("test_run_rms_norm.ty", vec!["--shape", "x=2x3"]),
        ("test_run_scale_builtin.ty", vec!["--shape", "x=2x3"]),
        ("test_run_two_layer_mlp.ty", vec!["--shape", "x=2x3"]),
    ])
}

fn backward_test_args() -> BTreeMap<&'static str, Vec<&'static str>> {
    BTreeMap::from([
        ("test_backward_basic_ops.ty", vec!["--shape", "x=2x3", "--shape", "y=2x3", "--shape", "w=3x2"]),
        ("test_backward_cross_entropy.ty", vec!["--shape", "logits=2x3", "--shape", "target=2x3"]),
        ("test_backward_dropout.ty", vec!["--shape", "x=2x3"]),
        ("test_backward_linear_bias.ty", vec!["--shape", "x=2x3"]),
        ("test_backward_linear_layer.ty", vec!["--shape", "x=2x3"]),
        ("test_backward_rms_norm.ty", vec!["--shape", "x=2x3"]),
        ("test_backward_silu_layer.ty", vec!["--shape", "x=2x3"]),
        ("test_backward_softmax_layer.ty", vec!["--shape", "x=2x3"]),
    ])
}

fn train_test_args() -> BTreeMap<&'static str, Vec<&'static str>> {
    BTreeMap::from([
        ("test_train_backend_single.ty", vec!["--shape", "x=2x3", "--shape", "target=2x3"]),
        ("test_train_dropout.ty", vec!["--shape", "x=2x3", "--shape", "target=2x3"]),
        ("test_train_linear_bias.ty", vec!["--shape", "x=2x3", "--shape", "target=2x3"]),
        ("test_train_pytorch_transformer.ty", vec!["--shape", "idx=1", "--shape", "target=1x3"]),
        ("test_train_pytorch_transformer_mask.ty", vec!["--shape", "idx=1", "--shape", "target=1x3"]),
        ("test_train_small_mlp.ty", vec!["--shape", "x=2x3", "--shape", "target=2x3"]),
        ("test_train_silu_mlp.ty", vec!["--shape", "x=2x3", "--shape", "target=2x3"]),
    ])
}

fn expected_failure_for(path: &Path) -> bool {
    let name = path.file_name().and_then(|x| x.to_str()).unwrap_or_default();
    name.contains("invalid") || EXPECTED_FAILURE_NAMES.contains(&name)
}

fn discover_tests(tests_dir: &Path, include_stress: bool) -> Result<Vec<PathBuf>, String> {
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(tests_dir).map_err(|err| format!("could not read {}: {err}", tests_dir.display()))? {
        let path = entry.map_err(|err| err.to_string())?.path();
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        let is_test = name.starts_with("test_") && name.ends_with(".ty");
        let is_stress = include_stress && name.starts_with("stress_") && name.ends_with(".ty");
        if is_test || is_stress {
            files.insert(path.canonicalize().map_err(|err| format!("could not resolve {}: {err}", path.display()))?);
        }
    }
    Ok(files.into_iter().collect())
}

fn resolve_test_files(requested: &[String], tests_dir: &Path, include_stress: bool) -> Result<Vec<PathBuf>, String> {
    if requested.is_empty() {
        return discover_tests(tests_dir, include_stress);
    }

    let all_candidates = discover_tests(tests_dir, true)?
        .into_iter()
        .filter_map(|path| Some((path.file_name()?.to_str()?.to_string(), path)))
        .collect::<BTreeMap<_, _>>();

    let mut resolved = Vec::new();
    for item in requested {
        let candidate = PathBuf::from(item);
        if candidate.is_file() {
            resolved.push(candidate.canonicalize().map_err(|err| format!("could not resolve {}: {err}", candidate.display()))?);
            continue;
        }

        let direct = tests_dir.join(item);
        if direct.is_file() {
            resolved.push(direct.canonicalize().map_err(|err| format!("could not resolve {}: {err}", direct.display()))?);
            continue;
        }

        if let Some(path) = all_candidates.get(candidate.file_name().and_then(|x| x.to_str()).unwrap_or_default()) {
            resolved.push(path.clone());
            continue;
        }

        return Err(format!("Could not resolve test file '{item}'"));
    }
    Ok(resolved)
}

fn build_command(
    binary: &Path,
    test_file: &Path,
    args: &Args,
    backend: Backend,
) -> Vec<String> {
    let runtime_map = runtime_test_args();
    let backward_map = backward_test_args();
    let train_map = train_test_args();
    let mut cmd = vec![
        binary.display().to_string(),
        test_file.display().to_string(),
    ];

    if args.tokens {
        cmd.push("--tokens".to_string());
    }
    if args.ast {
        cmd.push("--ast".to_string());
    }
    if args.semantics {
        cmd.push("--semantics".to_string());
    }
    if args.ir {
        cmd.push("--ir".to_string());
    }

    let name = test_file.file_name().and_then(|x| x.to_str()).unwrap_or_default();
    if args.run {
        cmd.push("--run".to_string());
        if backend != Backend::Local {
            cmd.push("--backend".to_string());
            cmd.push(backend.as_str().to_string());
        }
        if let Some(extra) = runtime_map.get(name) {
            cmd.extend(extra.iter().map(|x| (*x).to_string()));
        }
    } else if args.backward {
        cmd.push("--backward".to_string());
        if let Some(extra) = backward_map.get(name) {
            cmd.extend(extra.iter().map(|x| (*x).to_string()));
        }
    } else if args.train {
        cmd.push("--train".to_string());
        cmd.push("--backend".to_string());
        cmd.push(backend.as_str().to_string());
        if let Some(extra) = train_map.get(name) {
            cmd.extend(extra.iter().map(|x| (*x).to_string()));
        }
    } else if args.emit_pytorch {
        cmd.push("--emit-pytorch".to_string());
    }
    cmd
}

fn run_process(cmd: &[String]) -> Result<(i32, String, String), String> {
    let output = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run '{}': {err}", cmd[0]))?;
    Ok((
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn parse_training_losses(output: &str) -> Vec<f64> {
    let re = Regex::new(r"step=(\d+)\s+loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)").unwrap();
    re.captures_iter(output)
        .filter_map(|caps| caps.get(2).and_then(|m| m.as_str().parse::<f64>().ok()))
        .collect()
}

fn parse_execution_output(output: &str) -> Option<TensorOutput> {
    let re = Regex::new(
        r"(?s)--- Execution Output ---\s*shape=\[(?P<shape>[^\]]*)\]\s+dtype=(?P<dtype>\S+)\s*values=\[(?P<values>[^\]]*)\]"
    ).unwrap();
    let caps = re.captures(output)?;
    Some(TensorOutput {
        shape: parse_shape_list(caps.name("shape")?.as_str()),
        dtype: caps.name("dtype")?.as_str().to_string(),
        values: parse_float_list(caps.name("values")?.as_str()),
    })
}

fn parse_named_tensors(output: &str) -> BTreeMap<String, TensorOutput> {
    let re = Regex::new(
        r"(?s)(?P<name>[A-Za-z0-9_\.]+):\s*shape=\[(?P<shape>[^\]]*)\]\s+dtype=(?P<dtype>\S+)\s*values=\[(?P<values>[^\]]*)\]"
    ).unwrap();
    let mut parsed = BTreeMap::new();
    for caps in re.captures_iter(output) {
        let Some(name) = caps.name("name") else { continue };
        let Some(shape) = caps.name("shape") else { continue };
        let Some(dtype) = caps.name("dtype") else { continue };
        let Some(values) = caps.name("values") else { continue };
        parsed.insert(
            name.as_str().to_string(),
            TensorOutput {
                shape: parse_shape_list(shape.as_str()),
                dtype: dtype.as_str().to_string(),
                values: parse_float_list(values.as_str()),
            },
        );
    }
    parsed
}

fn parse_shape_list(text: &str) -> Vec<i64> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        text.split(',')
            .filter_map(|item| item.trim().parse::<i64>().ok())
            .collect()
    }
}

fn parse_float_list(text: &str) -> Vec<f64> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        text.split(',')
            .filter_map(|item| item.trim().parse::<f64>().ok())
            .collect()
    }
}

fn is_close(left: f64, right: f64, atol: f64, rtol: f64) -> bool {
    (left - right).abs() <= atol.max(rtol * right.abs())
}

fn outputs_close(lhs: &TensorOutput, rhs: &TensorOutput, atol: f64, rtol: f64) -> bool {
    lhs.shape == rhs.shape
        && lhs.dtype == rhs.dtype
        && lhs.values.len() == rhs.values.len()
        && lhs
            .values
            .iter()
            .zip(&rhs.values)
            .all(|(left, right)| is_close(*left, *right, atol, rtol))
}

fn named_outputs_close(
    lhs: &BTreeMap<String, TensorOutput>,
    rhs: &BTreeMap<String, TensorOutput>,
    atol: f64,
    rtol: f64,
) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().all(|(name, left)| {
            rhs.get(name)
                .map(|right| outputs_close(left, right, atol, rtol))
                .unwrap_or(false)
        })
}

fn append_cpu_output(stdout: &mut String, stderr: &mut String, cpu_stdout: &str, cpu_stderr: &str) {
    if !cpu_stdout.trim().is_empty() {
        stdout.push_str("\n--- cpu stdout ---\n");
        stdout.push_str(cpu_stdout);
    }
    if !cpu_stderr.trim().is_empty() {
        if !stderr.trim().is_empty() {
            stderr.push('\n');
        }
        stderr.push_str("--- cpu stderr ---\n");
        stderr.push_str(cpu_stderr);
    }
}

fn run_test(binary: &Path, test_file: &Path, args: &Args, backend: Backend) -> Result<TestResult, String> {
    let cmd = build_command(binary, test_file, args, backend);
    let (returncode, mut stdout, mut stderr) = run_process(&cmd)?;
    let expected_failure = expected_failure_for(test_file);
    let mut passed = if expected_failure { returncode != 0 } else { returncode == 0 };
    let test_name = test_file.file_name().and_then(|x| x.to_str()).unwrap_or_default();

    if passed && args.emit_pytorch && !expected_failure {
        passed = stdout.contains("--- PyTorch Source ---")
            && stdout.contains("import torch")
            && stdout.contains("import torch.nn as nn")
            && stdout.contains("class ")
            && stdout.contains("(nn.Module):")
            && stdout.contains("def build_model(")
            && stdout.contains("def forward(");
        if passed && test_name.starts_with("test_train_") {
            passed = stdout.contains("TRAIN_CONFIG = {");
        }
    }
    if passed && args.run && !expected_failure {
        passed = stdout.contains("--- Execution Output ---");
    }
    if passed && args.backward && !expected_failure {
        passed = stdout.contains("--- Gradient Output ---");
    }
    if passed && args.train && !expected_failure {
        passed = stdout.contains("--- Training Output ---");
    }
    if passed && args.train && !expected_failure {
        let losses = parse_training_losses(&stdout);
        passed = losses.len() >= 2
            && losses.iter().all(|loss| loss.is_finite())
            && losses.last().copied().unwrap_or(f64::INFINITY) < losses.first().copied().unwrap_or(f64::NEG_INFINITY);
    }
    if passed && args.compare_cpu && backend == Backend::Metal && !expected_failure {
        let cpu_cmd = build_command(binary, test_file, args, Backend::Local);
        let (cpu_returncode, cpu_stdout, cpu_stderr) = run_process(&cpu_cmd)?;
        passed = cpu_returncode == 0;
        if passed && args.run {
            let metal_output = parse_execution_output(&stdout);
            let cpu_output = parse_execution_output(&cpu_stdout);
            passed = metal_output
                .as_ref()
                .zip(cpu_output.as_ref())
                .map(|(left, right)| outputs_close(left, right, 5e-3, 5e-3))
                .unwrap_or(false);
        } else if passed && args.backward {
            let metal_grads = parse_named_tensors(&stdout);
            let cpu_grads = parse_named_tensors(&cpu_stdout);
            passed = !metal_grads.is_empty()
                && !cpu_grads.is_empty()
                && named_outputs_close(&metal_grads, &cpu_grads, 5e-3, 5e-3);
        } else if passed && args.train {
            let metal_losses = parse_training_losses(&stdout);
            let cpu_losses = parse_training_losses(&cpu_stdout);
            passed = metal_losses.len() == cpu_losses.len() && metal_losses.len() >= 2;
            if passed {
                for (metal_loss, cpu_loss) in metal_losses.iter().zip(&cpu_losses) {
                    if !is_close(*metal_loss, *cpu_loss, 5e-3, 5e-3) {
                        passed = false;
                        break;
                    }
                }
            }
        }
        if !passed {
            append_cpu_output(&mut stdout, &mut stderr, &cpu_stdout, &cpu_stderr);
        } else if !cpu_stderr.trim().is_empty() {
            append_cpu_output(&mut String::new(), &mut stderr, "", &cpu_stderr);
        }
    }

    Ok(TestResult {
        path: test_file.to_path_buf(),
        expected_failure,
        passed,
        returncode,
        stdout,
        stderr,
    })
}

fn extract_section<'a>(output: &'a str, start_marker: &str, end_marker: &str) -> Option<&'a str> {
    let start = output.find(start_marker)?;
    let end = output[start + start_marker.len()..]
        .find(end_marker)
        .map(|offset| start + start_marker.len() + offset + end_marker.len())
        .unwrap_or(output.len());
    Some(output[start..end].trim())
}

fn print_selected_sections(result: &TestResult, args: &Args) {
    let mut requested = Vec::new();
    if args.tokens {
        requested.push(("tokens", TOKENS_START, TOKENS_END));
    }
    if args.ast {
        requested.push(("ast", AST_START, AST_END));
    }
    if args.semantics {
        requested.push(("semantics", SEMANTICS_START, SEMANTICS_END));
    }
    if args.ir {
        requested.push(("ir", IR_START, IR_END));
    }
    if requested.is_empty() {
        return;
    }

    println!("\n=== {} ===", result.path.file_name().and_then(|x| x.to_str()).unwrap_or_default());
    for (key, start, end) in requested {
        if let Some(section) = extract_section(&result.stdout, start, end) {
            println!("{section}\n");
        } else {
            println!("[missing section: {key}]");
        }
    }
    if !result.stderr.trim().is_empty() {
        println!("--- stderr ---");
        println!("{}", result.stderr.trim());
    }
}

fn print_failure(result: &TestResult) {
    let expectation = if result.expected_failure {
        "expected failure"
    } else {
        "expected success"
    };
    println!(
        "\n[FAIL] {} ({expectation}, exit={})",
        result.path.file_name().and_then(|x| x.to_str()).unwrap_or_default(),
        result.returncode
    );
    if !result.stdout.trim().is_empty() {
        println!("--- stdout ---");
        println!("{}", result.stdout.trim_end());
    }
    if !result.stderr.trim().is_empty() {
        println!("--- stderr ---");
        println!("{}", result.stderr.trim_end());
    }
}

fn sorted_list(values: &[&str]) -> String {
    let mut items = values.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    items.sort();
    items.join(", ")
}

fn ensure_supported(tests: &[PathBuf], allowed: &[&str], label: &str) -> Result<(), String> {
    let unsupported = tests
        .iter()
        .filter_map(|test| test.file_name().and_then(|x| x.to_str()))
        .filter(|name| !allowed.contains(name))
        .map(|x| x.to_string())
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!("{label}: {}", sorted_list(allowed)))
    }
}

fn real_main() -> i32 {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("usage: {}", usage());
            return 2;
        }
    };

    let binary = match args.binary.canonicalize() {
        Ok(path) => path,
        Err(_) => args.binary.clone(),
    };
    let tests_dir = match args.tests_dir.canonicalize() {
        Ok(path) => path,
        Err(_) => args.tests_dir.clone(),
    };
    let backend = args.backend.unwrap_or(Backend::Local);

    if !binary.is_file() {
        eprintln!("error: compiler binary not found at {}", binary.display());
        return 2;
    }
    if !tests_dir.is_dir() {
        eprintln!("error: tests directory not found at {}", tests_dir.display());
        return 2;
    }

    let mut requested = args.file_flags.clone();
    requested.extend(args.files.clone());
    let tests = match resolve_test_files(&requested, &tests_dir, args.stress) {
        Ok(tests) => tests,
        Err(err) => {
            eprintln!("error: {err}");
            return 2;
        }
    };

    if tests.is_empty() {
        eprintln!("error: no tests selected");
        return 2;
    }
    if backend == Backend::Metal && !(args.run || args.backward || args.train) {
        eprintln!("error: --backend metal currently applies only to --run, --backward, or --train");
        return 2;
    }
    if args.emit_pytorch {
        if let Err(err) = ensure_supported(&tests, PYTORCH_SUPPORTED_TESTS, "PyTorch export tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }
    if backend == Backend::Metal && args.run {
        if let Err(err) = ensure_supported(&tests, METAL_SUPPORTED_RUNTIME_TESTS, "metal runtime tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }
    if backend == Backend::Metal && args.backward {
        if let Err(err) = ensure_supported(&tests, METAL_SUPPORTED_BACKWARD_TESTS, "metal backward tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }
    if backend == Backend::Metal && args.train {
        if let Err(err) = ensure_supported(&tests, METAL_SUPPORTED_TRAIN_TESTS, "metal train tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }
    if backend == Backend::PyTorch && args.train {
        if let Err(err) = ensure_supported(&tests, PYTORCH_SUPPORTED_TRAIN_TESTS, "pytorch train tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }
    if backend == Backend::PyTorch && args.run {
        if let Err(err) = ensure_supported(&tests, PYTORCH_SUPPORTED_RUNTIME_TESTS, "pytorch runtime tests are currently supported only for") {
            eprintln!("error: {err}");
            return 2;
        }
    }

    let mut passed = 0usize;
    let mut failed = 0usize;

    for test_file in &tests {
        let result = match run_test(&binary, test_file, &args, backend) {
            Ok(result) => result,
            Err(err) => TestResult {
                path: test_file.clone(),
                expected_failure: expected_failure_for(test_file),
                passed: false,
                returncode: 1,
                stdout: String::new(),
                stderr: err,
            },
        };
        let status = if result.passed { "PASS" } else { "FAIL" };
        let expectation = if result.expected_failure { "xfail" } else { "ok" };
        println!(
            "[{status}] {} [{expectation}]",
            test_file.file_name().and_then(|x| x.to_str()).unwrap_or_default()
        );

        if args.show_output {
            if !result.stdout.trim().is_empty() {
                println!("--- stdout ---");
                println!("{}", result.stdout.trim_end());
            }
            if !result.stderr.trim().is_empty() {
                println!("--- stderr ---");
                println!("{}", result.stderr.trim_end());
            }
        } else {
            print_selected_sections(&result, &args);
        }

        if result.passed {
            passed += 1;
        } else {
            failed += 1;
            print_failure(&result);
            if args.fail_fast {
                break;
            }
        }
    }

    println!("\nSummary: {passed} passed, {failed} failed, {} total", tests.len());
    if failed == 0 { 0 } else { 1 }
}

fn main() {
    std::process::exit(real_main());
}
