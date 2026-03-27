use std::env;
use std::fs;
use std::path::PathBuf;
use std::collections::BTreeMap;

use tysor::backend::local::{run_local_backward_module, run_local_forward_module, run_local_train_module};
use tysor::backend::metal::runtime::run_metal_module;
use tysor::backend::pytorch::runtime::{run_pytorch_forward_module, run_pytorch_train_module};
use tysor::compiler::frontend_ir::FrontendLowerer;
use tysor::compiler::lexer::tokenize;
use tysor::compiler::parser::Parser;
use tysor::compiler::semantic_analyzer::SemanticAnalyzer;
use tysor::backend::core::execution_plan::compile_function_execution_plan;
use tysor::backend::core::kind::BackendKind;
use tysor::backend::metal::codegen::generate_metal_code;
use tysor::backend::pytorch::codegen::generate_standalone_pytorch_module;
use tysor::runtime::interpreter::RuntimeRunOptions;
use tysor::training::backward::BackwardRunOptions;
use tysor::training::executor::TrainRunOptions;

/// Thin CLI wrapper around the Rust pipeline.
///
/// The heavy lifting stays in the library modules; this file mostly parses flags,
/// runs the frontend once, and dispatches to the requested backend/export path.
#[derive(Debug, Default)]
struct CliOptions {
    input_path: Option<PathBuf>,
    emit_metal: bool,
    emit_pytorch: bool,
    run: bool,
    backward: bool,
    train: bool,
    tokens: bool,
    ast: bool,
    semantics: bool,
    ir: bool,
    print_pipeline: bool,
    backend_overridden: bool,
    entry: String,
    backend: tysor::backend::core::kind::BackendKind,
    tensor_shapes: BTreeMap<String, Vec<i64>>,
}

fn usage() -> &'static str {
    "tysor <input.ty> [--emit-metal] [--emit-pytorch] [--run] [--backward] [--train] [--tokens] [--ast] [--semantics] [--ir] [--print-pipeline] [--entry <name>] [--backend <local|metal|pytorch>]"
}

fn parse_cli() -> Result<CliOptions, String> {
    let mut args = env::args().skip(1);
    let mut options = CliOptions {
        entry: "model".to_string(),
        backend: tysor::backend::core::kind::BackendKind::Local,
        ..CliOptions::default()
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--emit-metal" => options.emit_metal = true,
            "--emit-pytorch" => options.emit_pytorch = true,
            "--run" => options.run = true,
            "--backward" => options.backward = true,
            "--train" => options.train = true,
            "--tokens" => options.tokens = true,
            "--ast" => options.ast = true,
            "--semantics" => options.semantics = true,
            "--ir" => options.ir = true,
            "--print-pipeline" => options.print_pipeline = true,
            "--entry" => {
                options.entry = args.next().ok_or_else(|| "missing value for --entry".to_string())?;
            }
            "--backend" => {
                let backend = args.next().ok_or_else(|| "missing value for --backend".to_string())?;
                options.backend = backend.parse()?;
                options.backend_overridden = true;
            }
            "--shape" => {
                let spec = args.next().ok_or_else(|| "missing value for --shape".to_string())?;
                let (name, dims) = parse_shape_spec(&spec)?;
                options.tensor_shapes.insert(name, dims);
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option: {arg}")),
            _ => {
                if options.input_path.is_some() {
                    return Err("multiple input paths provided".to_string());
                }
                options.input_path = Some(PathBuf::from(arg));
            }
        }
    }

    if options.input_path.is_none() {
        return Err("missing input path".to_string());
    }

    Ok(options)
}

fn parse_shape_spec(spec: &str) -> Result<(String, Vec<i64>), String> {
    let equals = spec
        .find('=')
        .ok_or_else(|| format!("expected --shape name=dimxdim, got '{spec}'"))?;
    if equals == 0 || equals + 1 >= spec.len() {
        return Err(format!("expected --shape name=dimxdim, got '{spec}'"));
    }
    let name = spec[..equals].to_string();
    let dims = spec[equals + 1..]
        .split('x')
        .map(|part| {
            let value = part.trim().parse::<i64>().map_err(|_| format!("invalid dimension '{part}' in shape '{spec}'"))?;
            if value <= 0 {
                return Err(format!("invalid dimension '{part}' in shape '{spec}'"));
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.is_empty() {
        return Err(format!("shape '{spec}' must contain at least one dimension"));
    }
    Ok((name, dims))
}

fn print_section(title: &str, body: impl AsRef<str>, end_marker: &str) {
    println!("{title}");
    println!("{}", body.as_ref());
    println!("{end_marker}");
}

fn main() {
    let options = match parse_cli() {
        Ok(options) => options,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("usage: {}", usage());
            std::process::exit(2);
        }
    };

    let input_path = options.input_path.as_ref().expect("validated input path");
    let source = match fs::read_to_string(input_path) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("error: could not read {}: {err}", input_path.display());
            std::process::exit(1);
        }
    };

    println!("tysor rust port bootstrap");
    println!("input={}", input_path.display());
    println!("bytes={}", source.len());
    println!("entry={}", options.entry);
    println!("backend={}", options.backend.as_str());
    // Keep the top-level compilation flow visible here:
    // source -> tokenize -> parse -> semantic analysis -> frontend lowering.
    match tokenize(&source).and_then(|tokens| {
        let token_count = tokens.len();
        let token_dump = if options.tokens {
            Some(
                tokens
                    .iter()
                    .map(|token| token.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            None
        };
        let mut parser = Parser::new(tokens);
        parser.parse_program().and_then(|program| {
            let ast_dump = if options.ast {
                Some(format!("{program:#?}"))
            } else {
                None
            };
            let mut analyzer = SemanticAnalyzer::new();
            analyzer.analyze(&program)?;
            let semantics_dump = if options.semantics {
                Some(format!(
                    "ok\nconfigs={}\ntrains={}\nlayers={}\nfunctions={}\nglobals={}",
                    program.configs.len(),
                    program.trains.len(),
                    program.layers.len(),
                    program.functions.len(),
                    program.globals.len()
                ))
            } else {
                None
            };
            let mut lowerer = FrontendLowerer::new(&program);
            let lowered = lowerer.lower()?;
            let ir_dump = if options.ir {
                Some(format!("{lowered:#?}"))
            } else {
                None
            };
            Ok((token_count, program, lowered, token_dump, ast_dump, semantics_dump, ir_dump))
        })
    }) {
        Ok((token_count, program, lowered, token_dump, ast_dump, semantics_dump, ir_dump)) => {
            println!("tokens={token_count}");
            println!(
                "program=configs:{} trains:{} layers:{} functions:{} globals:{}",
                program.configs.len(),
                program.trains.len(),
                program.layers.len(),
                program.functions.len(),
                program.globals.len()
            );
            println!(
                "lowered=configs:{} trains:{} functions:{} globals:{} execution_plan:{}",
                lowered.configs.len(),
                lowered.trains.len(),
                lowered.functions.len(),
                lowered.globals.len(),
                if lowered.execution_plan.is_some() { "yes" } else { "no" }
            );
            if let Some(token_dump) = token_dump {
                print_section("--- Tokenization Step ---", token_dump, "-------------------------");
            }
            if let Some(ast_dump) = ast_dump {
                print_section("--- Parsing Step ---", ast_dump, "--------------------");
            }
            if let Some(semantics_dump) = semantics_dump {
                print_section(
                    "--- Semantic Analysis Step ---",
                    semantics_dump,
                    "------------------------------",
                );
            }
            if let Some(ir_dump) = ir_dump {
                print_section("--- Lowered Frontend IR ---", ir_dump, "---------------------------");
            }
            if options.emit_metal {
                // Metal emission starts from one lowered entry function and a Metal plan for it.
                let entry = match lowered
                    .functions
                    .iter()
                    .find(|function| function.name == options.entry)
                    .ok_or_else(|| format!("Entry function '{}' not found in lowered module", options.entry))
                    .and_then(|function| compile_function_execution_plan(function, BackendKind::Metal))
                {
                    Ok(entry) => entry,
                    Err(err) => {
                        eprintln!("{err}");
                        std::process::exit(1);
                    }
                };
                let metal = match generate_metal_code(&entry) {
                    Ok(metal) => metal,
                    Err(err) => {
                        eprintln!("{err}");
                        std::process::exit(1);
                    }
                };
                println!("\n--- Metal Source ---");
                println!("{}", metal.source);
            }
            if options.emit_pytorch {
                let pytorch = match generate_standalone_pytorch_module(&lowered, &options.entry) {
                    Ok(pytorch) => pytorch,
                    Err(err) => {
                        eprintln!("{err}");
                        std::process::exit(1);
                    }
                };
                println!("\n--- PyTorch Source ---");
                println!("{}", pytorch.source);
            }
            if options.run {
                // Runtime dispatch is backend-specific, but each backend consumes the same lowered module.
                let runtime_options = RuntimeRunOptions {
                    entry: options.entry.clone(),
                    tensor_shapes: options.tensor_shapes.clone(),
                };
                let result = match options.backend {
                    BackendKind::Local => run_local_forward_module(&lowered, &runtime_options),
                    BackendKind::Metal => run_metal_module(&lowered, &runtime_options),
                    BackendKind::PyTorch => run_pytorch_forward_module(&lowered, &runtime_options),
                };
                if let Err(err) = result {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            }
            if options.backward {
                // Backward uses one Rust driver and selects the execution backend internally.
                let backward_options = BackwardRunOptions {
                    backend: options.backend,
                    entry: options.entry.clone(),
                    tensor_shapes: options.tensor_shapes.clone(),
                };
                if let Err(err) = run_local_backward_module(&lowered, &backward_options) {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            }
            if options.train {
                // Training shares a Rust SGD loop and swaps only the forward execution backend.
                let train_options = TrainRunOptions {
                    backend: options.backend,
                    backend_overridden: options.backend_overridden,
                    entry: options.entry.clone(),
                    tensor_shapes: options.tensor_shapes.clone(),
                };
                let result = match options.backend {
                    BackendKind::Local | BackendKind::Metal => run_local_train_module(&lowered, &train_options),
                    BackendKind::PyTorch => run_pytorch_train_module(&lowered, &train_options),
                };
                if let Err(err) = result {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            }
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
    if options.print_pipeline {
        println!("pipeline=rust lexer+parser active, semantic/runtime port in progress");
    }
}
