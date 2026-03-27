use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/metal_bridge.mm");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let source = PathBuf::from("native/metal_bridge.mm");
    let object = out_dir.join("metal_bridge.o");
    let library = out_dir.join("libtysor_metal_bridge.a");

    let status = Command::new("xcrun")
        .args([
            "clang++",
            "-c",
            source.to_str().expect("utf8 path"),
            "-o",
            object.to_str().expect("utf8 path"),
            "-std=c++17",
            "-fobjc-arc",
        ])
        .status()
        .expect("failed to invoke xcrun clang++");
    if !status.success() {
        panic!("failed to compile native/metal_bridge.mm");
    }

    let status = Command::new("ar")
        .args([
            "rcs",
            library.to_str().expect("utf8 path"),
            object.to_str().expect("utf8 path"),
        ])
        .status()
        .expect("failed to invoke ar");
    if !status.success() {
        panic!("failed to archive Metal bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tysor_metal_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=c++");
}
