fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    // println!("cargo:rustc-link-search=/opt/intel/oneapi/mkl/latest/lib/intel64");
    // println!("cargo:rustc-link-lib=mkl_intel_lp64");
    // println!("cargo:rustc-link-lib=mkl_sequential");
    // println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    // lapack library for matrix operations
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=cblas");
}
