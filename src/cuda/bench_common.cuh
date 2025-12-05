#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <vector>
#include <iostream>
#include <string_view>
#include <cmath>
#include <type_traits>
#include <string_view>

#include "macros.cuh"

template<auto Kernel>
struct Launcher {
    dim3 grid;
    dim3 block;
    float alpha = 1.0f;
    float beta  = 0.0f;

    // 8-arg form (uses passed alpha/beta)
    template<typename Tin>
    __host__ void operator()(const Tin* A, const Tin* B, float* C,
                             int M, int N, int K,
                             float a, float b) const
    {
        Kernel<<<grid, block>>>(A, B, C, M, N, K, a, b);
    }
};

template<auto Kernel>
inline Launcher<Kernel> make_launcher(dim3 grid, dim3 block,
                                      float alpha = 1.0f, float beta = 0.0f)
{
    return {grid, block, alpha, beta};
}

// -------------------------------
// Tolerance traits
// -------------------------------
template <typename T>
struct GemmTol {
    static constexpr double abs = 1e-3;
    static constexpr double rel = 1e-3;
};

template <>
struct GemmTol<__nv_bfloat16> {
    static constexpr double abs = 0.1;
    static constexpr double rel = 0.1;
};

// -------------------------------
// cuBLAS reference GEMM
// Inputs: row-major A[M,K], B[K,N]
// Output: row-major C[M,N] in FP32
// -------------------------------
template <typename FloatType>
inline void cublas_ref_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const FloatType* dA,
    const FloatType* dB,
    float* dC_ref)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Your exisFloatTypeg row-major trick:
    // C_row(M,N) = A_row(M,K) * B_row(K,N)
    // Map to col-major by swapping operands:
    // C_col(N,M) = B_col(N,K) * A_col(K,M)
    int m = N;
    int n = M;
    int k = K;

    int lda = N;  // leading dim of B in this mapping
    int ldb = K;  // leading dim of A
    int ldc = N;

    if constexpr (std::is_same_v<FloatType, float>) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dB, lda,
            dA, ldb,
            &beta,
            dC_ref, ldc));
    }
    else if constexpr (std::is_same_v<FloatType, __nv_bfloat16>) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dB, CUDA_R_16BF, lda,
            dA, CUDA_R_16BF, ldb,
            &beta,
            dC_ref, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else {
        static_assert(!sizeof(FloatType), "Unsupported FloatType for cublas_ref_gemm");
    }
}

// -------------------------------
// Generic runner:
// - FloatType is the input type (float or bf16)
// - LaunchFn is a callable that launches your kernel
//   (captures grid/block/alpha/beta/whatever it needs)
// -------------------------------
template <typename FloatType, typename LaunchFn>
double run_gemm_bench(
    cublasHandle_t handle,
    int M, int N, int K,
    const FloatType* dA,
    const FloatType* dB,
    float* dC,
    int iters,
    LaunchFn&& launch,
    std::string_view name,
    float alpha,
    float beta,
    bool verify = true,
    bool verbose = true)
{
    const int64_t sizeC = int64_t(M) * N;

    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    auto do_launch = [&]() {
        // pass pointers + sizes to the launch callable
        launch(dA, dB, dC, M, N, K, alpha, beta);
    };

    // Warm-up
    do_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify
    if (verify) {
        float* dC_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
        CHECK_CUDA(cudaMemset(dC_ref, 0, sizeC * sizeof(float)));

        cublas_ref_gemm<FloatType>(handle, M, N, K, dA, dB, dC_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> hC(sizeC);
        std::vector<float> hRef(sizeC);

        CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hRef.data(), dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(dC_ref));

        double max_abs = 0.0;
        double max_rel = 0.0;

        for (int64_t i = 0; i < sizeC; ++i) {
            double ref  = hRef[i];
            double val  = hC[i];
            double diff = std::abs(val - ref);
            double rel  = (std::abs(ref) > 0.0) ? diff / std::abs(ref) : diff;
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, rel);
        }

        if (verbose) {
            std::cout << "check: max_abs = " << max_abs
                      << ", max_rel = " << max_rel << "   ";
        }

        const double tol_abs = GemmTol<FloatType>::abs;
        const double tol_rel = GemmTol<FloatType>::rel;

        bool ok = (max_abs <= tol_abs)/* && (max_rel <= tol_rel) */;

        if (verbose) {
            std::cout << (ok ? "(OK)" : "(WARN)") << "\n";
        }

        if (!ok) return 0.0;
    }

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        do_launch();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = total_ms / iters;
    double flops  = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    if (verbose) {
        std::cout << "[" << name << "]   avg time: "
                  << avg_ms << " ms, " << tflops << " TFLOP/s\n\n";
    }

    return tflops;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
extern double BASELINE_TFLOPS;

// Unified cuBLAS GEMM for row-major A/B/C buffers
// Tin = float or __nv_bfloat16
template <typename Tin>
double run_cublas_rowmajor_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const Tin* dA,   // row-major A[M,K]
    const Tin* dB,   // row-major B[K,N]
    float* dC,       // row-major C[M,N] (FP32 output)
    int iters,
    std::string_view tag = {},
    bool update_baseline = false)
{
    static_assert(
        std::is_same_v<Tin, float> || std::is_same_v<Tin, __nv_bfloat16>,
        "run_cublas_rowmajor_gemm supports only float or __nv_bfloat16 inputs"
    );

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Row-major trick:
    // C_row(M,N) = A_row(M,K) * B_row(K,N)
    // -> treat buffers as column-major:
    // C_col(N,M) = B_col(N,K) * A_col(K,M)
    const int m = N;
    const int n = M;
    const int k = K;

    const int lda = N; // leading dim of B in this mapping
    const int ldb = K; // leading dim of A
    const int ldc = N; // leading dim of C

    auto do_gemm = [&]() {
        if constexpr (std::is_same_v<Tin, float>) {
            CHECK_CUBLAS(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                dB, lda,
                dA, ldb,
                &beta,
                dC, ldc));
        } else { // __nv_bfloat16
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                dB, CUDA_R_16BF, lda,
                dA, CUDA_R_16BF, ldb,
                &beta,
                dC, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    };

    // Warm-up
    do_gemm();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        do_gemm();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const double avg_ms = total_ms / iters;
    const double flops  = 2.0 * double(M) * double(N) * double(K);
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;

    if (update_baseline) {
        BASELINE_TFLOPS = tflops;
    }

    if (!tag.empty()) {
        std::cout << "[" << tag << "] ";
    } else {
        if constexpr (std::is_same_v<Tin, float>)
            std::cout << "[cuBLAS SGEMM] ";
        else
            std::cout << "[cuBLAS BF16 TC] ";
    }

    std::cout << "avg time: " << avg_ms
              << " ms, " << tflops << " TFLOP/s\n\n";

    return tflops;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
