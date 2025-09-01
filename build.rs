// build.rs
use std::process::Command;
use std::env;

fn main() {
    let trt_root_dir = env::var("TENSORRT_ROOT_DIR").unwrap_or("/home/lemon_cn/tensorrt10".to_string());
    let cuda_root_dir = env::var("CUDA_HOME").unwrap_or("/home/lemon_cn/cuda-12.8".to_string());
    let trt_include_dir = format!("{}/include", trt_root_dir);
    let trt_lib_dir = format!("{}/lib", trt_root_dir);
    let cuda_lib_dir = format!("{}/lib64", cuda_root_dir);
    let cuda_include_dir = format!("{}/include", cuda_root_dir);

    // 直接调用 g++ 编译
    let output = Command::new("g++")
        .args(&[
            "-std=c++14", "-fPIC", "-shared",
            &format!("-I{}", trt_include_dir),
            &format!("-I{}", cuda_include_dir),
            &format!("-L{}", trt_lib_dir),
            &format!("-L{}", cuda_lib_dir),
            "tensorrt_wrapper_v3.cpp",
            "-lnvinfer",
            "-lnvonnxparser",
            "-lcudart",
            "-o", "libtensorrt_wrapper_v3.so"
        ])
        .output()
        .expect("Failed to execute g++");

    if !output.status.success() {
        panic!("G++ compilation failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("cargo:rustc-link-search=native={}", trt_lib_dir);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-search=native=.");

    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=tensorrt_wrapper_v3");

    println!("cargo:rerun-if-changed=tensorrt_wrapper_v3.cpp");
}
