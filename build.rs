fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" && std::env::var("CARGO_FEATURE_NEURAL").is_ok() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift-5.5/macosx");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx");
    }
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_w8_gemm();
    }
}

/// csrc/w8_gemm.cu, the KEV_W8=int8 CUTLASS kernel: nvcc against a CUTLASS checkout (CUTLASS_DIR, v3.9) into a
/// static library, linked with the static CUDA runtime it launches through.
fn build_w8_gemm() {
    use std::process::Command;
    println!("cargo:rerun-if-changed=csrc/w8_gemm.cu");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    let cutlass = std::env::var("CUTLASS_DIR").expect("the cuda feature needs CUTLASS_DIR (a CUTLASS v3.9 checkout) for csrc/w8_gemm.cu");
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("w8_gemm.o");
    let ok = Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC"])
        .args(["-gencode", "arch=compute_80,code=sm_80", "-gencode", "arch=compute_86,code=sm_86", "-gencode", "arch=compute_89,code=sm_89"])
        .arg(format!("-I{cutlass}/include"))
        .args(["-c", "csrc/w8_gemm.cu", "-o"])
        .arg(&obj)
        .status()
        .expect("nvcc");
    assert!(ok.success(), "nvcc failed on csrc/w8_gemm.cu");
    let ok = Command::new("ar").arg("crs").arg(out.join("libkevw8.a")).arg(&obj).status().expect("ar");
    assert!(ok.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=kevw8");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=cudart_static");
    for lib in ["stdc++", "dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
