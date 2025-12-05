// gemm_bench.cu
//
// Benchmark three FP32 GEMMs (row-major):
//   1) cuBLAS sgemm
//   2) Naive CUDA kernel
//   3) CUTLASS SIMT FP32 (device::Gemm)
//
// Problem: C = A * B
// A: M x K, B: K x N, C: M x N, all row-major.
//
// Command line:
//   ./gemm_bench [M] [N] [K] [iters]
// Defaults:
//   M = N = K = 2048, iters = 10
//
// Build example (adjust paths & arch as needed):
//   nvcc -O3 -std=c++17 fp_16_gemm.cu -o fp16_gemm_bench -I ~/cutlass/include -lcublas -arch=sm_86
// How to profile:
// sudo /usr/local/cuda-12.3/bin/ncu  --set full --kernel-name "gemm_warp_tensorcore" ./fp16_gemm_bench


#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include "macros.cuh"

#include <mma.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convert_f32_to_nvbf16_kernel(const float* __restrict__ in,
                                             __nv_bfloat16* __restrict__ out,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

void run_cublas_bf16_tc_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16* dA_bf16, const __nv_bfloat16* dB_bf16, float* dC_f32,
    int iters)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Row-major trick, same as your FP32 version:
    // C_row = A_row * B_row  (M x K, K x N)
    // Use column-major with:
    //   C_col (N x M) = B_col (N x K) * A_col (K x M)
    // so m=N, n=M, k=K, A=B_row, B=A_row, C=C_row.
    int m = N;
    int n = M;
    int k = K;
    int lda = N;  // B_row leading dim
    int ldb = K;  // A_row leading dim
    int ldc = N;  // C_row leading dim

    // Warm-up
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        dB_bf16, CUDA_R_16BF, lda,
        dA_bf16, CUDA_R_16BF, ldb,
        &beta,
        dC_f32, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dB_bf16, CUDA_R_16BF, lda,
            dA_bf16, CUDA_R_16BF, ldb,
            &beta,
            dC_f32, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

    std::cout << "[cuBLAS BF16 TC] avg time: " << avg_ms
              << " ms,  TFLOP/s: " << tflops << std::endl << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace nvcuda;

// WMMA tile: 16x16x16 (m,n,k)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Host wrapper for WMMA BF16 GEMM
using KernelFunc = void(*)(const __nv_bfloat16*,
                           const __nv_bfloat16*,
                           float*,
                           int,int,int);

void cublas_fp16_ref_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16* dA,  // row-major A[M,K]
    const __nv_bfloat16* dB,  // row-major B[K,N]
    float* dC_ref)    // row-major C[M,N]
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int m = N;
    int n = M;
    int k = K;
    int lda = N;
    int ldb = K;
    int ldc = N;

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

/////////////////////////////////////////////////////////////////////////////////////////////////

template <KernelFunc Kernel>
void run_wmma_bf16_gemm(int M, int N, int K,
                        const __nv_bfloat16* dA_bf16,
                        const __nv_bfloat16* dB_bf16,
                        float* dC_f32,
                        int iters,
                        dim3 blockDim,
                        dim3 gridDim,
                        cublasHandle_t handle,
                        std::string name)
{
    // Require multiples of 16 for this simple demo
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        std::cerr << "M, N, K must be multiples of 16 for this WMMA example.\n";
        return;
    }

    // ---------------- Warm-up ----------------
    Kernel<<<gridDim, blockDim>>>(dA_bf16, dB_bf16, dC_f32, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------- Verification ----------------
    {
        size_t sizeC = M * N;
        // Compute reference FP32 GEMM with cuBLAS
        float* dC_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
        CHECK_CUDA(cudaMemset(dC_ref, 0, sizeC * sizeof(float)));

        cublas_fp16_ref_gemm(handle, M, N, K, dA_bf16, dB_bf16, dC_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy both results back to host
        std::vector<float> hC(sizeC);
        std::vector<float> hC_ref(sizeC);

        CHECK_CUDA(cudaMemcpy(hC.data(),     dC_f32, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(dC_ref));

        double max_abs = 0.0;
        double max_rel = 0.0;
        const int max_print_count = 0;
        for (int64_t i = 0; i < sizeC; ++i) {
            double ref = hC_ref[i];
            double val = hC[i];
            double diff = std::abs(val - ref);
            double rel  = (std::abs(ref) > 0.0) ? diff / std::abs(ref) : diff;

            if(i < max_print_count) {
                std::cout << ref << " vs " << val << std::endl;
            }

            max_abs = std::max(diff, max_abs);
            max_rel = std::max(rel, max_rel);
        }

        std::cout << "check: max_abs = " << max_abs;

        double tol = 0.1;  // BF16-level accuracy, adjust if you like
        if (max_rel < tol) {
            std::cout << "  (OK)" << std::endl;
        } else {
            std::cout << "  (WARN)" << std::endl;
        }
    }

    // ---------------- Timing ----------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        Kernel<<<gridDim, blockDim>>>(dA_bf16, dB_bf16, dC_f32, M, N, K);
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

    std::cout << "[" << name << "]   avg time: " << avg_ms
              << " ms,  TFLOP/s: " << tflops << std::endl << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Each block is a single warp (32 threads).
// Each warp computes one 16x16 tile of C.
__global__ void wmma_bf16_naive_gemm_kernel(const __nv_bfloat16* __restrict__ A,
                                            const __nv_bfloat16* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K)
{
    // Tile indices (in units of 16x16)
    int tile_n = blockIdx.y;

    int tile_m = blockIdx.x;

    int row = tile_m * WMMA_M;
    int col = tile_n * WMMA_N;

    if (row >= M || col >= N) return;

    // Fragments:
    // A, B as BF16, C as FP32 accumulator
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in tiles of 16
    for (int k = 0; k < K; k += WMMA_K) {
        const __nv_bfloat16* tileA = A + row * K + k;  // A[row:row+16, k:k+16]
        const __nv_bfloat16* tileB = B + k * N + col;  // B[k:k+16, col:col+16]

        // Load 16x16 tiles from row-major storage
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);

        // C_tile += A_tile * B_tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result tile back to C (row-major)
    float* tileC = C + row * N + col;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void wmma_bf16_cta(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K)
{
    // 32 threads among x sit inside each warp, and we use Y/Z dims for tiling 
    int warp_tile_m = threadIdx.y;
    int warp_tile_n = threadIdx.z;

    // global coord of tile among m and n matrix dims
    int tile_m = blockIdx.y * blockDim.y + warp_tile_m;
    int tile_n = blockIdx.z * blockDim.z + warp_tile_n;

    // starting row and col of tile
    int row = tile_m * WMMA_M;
    int col = tile_n * WMMA_N;

    if (row >= M || col >= N)
        return;

    // Fragments:
    // A, B as BF16, C as FP32 accumulator
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in tiles of 16
    for (int k = 0; k < K; k += WMMA_K) {
        const __nv_bfloat16* tileA = A + row * K + k;  // A[row:row+16, k:k+16]
        const __nv_bfloat16* tileB = B + k * N + col;  // B[k:k+16, col:col+16]

        // Load 16x16 tiles from row-major storage
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);

        // C_tile += A_tile * B_tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result tile back to C (row-major)
    float* tileC = C + row * N + col;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int BM, int BN, int BK, int WM, int WN>
__global__ void
gemm_warp_tiling(const __nv_bfloat16* __restrict__ A,
                 const __nv_bfloat16* __restrict__ B,
                 float      * __restrict__ C,
                 int M, int N, int K)
{
    // Tensor Core tile shape (Ampere)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int WARP_SIZE = 32;

    static_assert(WM % MMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % MMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % MMA_K == 0, "BK must be multiple of 16");

    // block tile origin in C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // leading dims
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // thread / warp ids
    const int tid   = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = tid / WARP_SIZE;

    // warp tiling in the block
    constexpr int WARPS_PER_BLOCK_N = BN / WN;

    const int warp_row = warp_id / WARPS_PER_BLOCK_N;
    const int warp_col = warp_id % WARPS_PER_BLOCK_N;

    const int warp_c_row = block_row + warp_row * WM;
    const int warp_c_col = block_col + warp_col * WN;

    // Warp tile decomposed into MMA tiles
    constexpr int WARP_M_TILES = WM / MMA_M;
    constexpr int WARP_N_TILES = WN / MMA_N;

    // Shared memory: block tile of A and B
    __shared__ __nv_bfloat16 As[BM][BK];   // M x K
    __shared__ __nv_bfloat16 Bs[BK][BN];   // K x N

    const int num_k_tiles = (K + BK - 1) / BK;

    // Accumulator fragments per warp
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float>
        c_frags[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            wmma::fill_fragment(c_frags[mi][nj], 0.0f);
        }
    }

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        const int k_base = tile_k * BK;

        // ----------------------------
        // 1) load block tile of A and B into shared (cooperatively)
        // ----------------------------
        const int block_threads = blockDim.x * blockDim.y;

        for (int i = tid; i < BM * BK; i += block_threads) {
            int row = i / BK;
            int col = i % BK;

            int g_row = block_row + row;
            int g_col = k_base + col;

            As[row][col] = A[g_row * lda + g_col];

        }

        for (int i = tid; i < BK * BN; i += block_threads) {
            int row = i / BN;
            int col = i % BN;

            int g_row = k_base + row;
            int g_col = block_col + col;

            Bs[row][col] = B[g_row * ldb + g_col];
        }

        __syncthreads();

        // ----------------------------
        // 2) warp-level MMA over this K-tile
        // ----------------------------
        for (int kk = 0; kk < BK; kk += MMA_K) {

            // A frags for each "row" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_a,
                           MMA_M, MMA_N, MMA_K,
                           __nv_bfloat16, wmma::row_major> a_frags[WARP_M_TILES];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                int a_row = (warp_c_row - block_row) + mi * MMA_M; // within As
                int a_col = kk;                                    // within As

                const __nv_bfloat16* a_ptr = &As[a_row][a_col];
                wmma::load_matrix_sync(a_frags[mi], a_ptr, BK);
            }

            // B frags for each "column" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_b,
                           MMA_M, MMA_N, MMA_K,
                           __nv_bfloat16, wmma::row_major> b_frags[WARP_N_TILES];

            #pragma unroll
            for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                int b_row = kk;                                    // within Bs
                int b_col = (warp_c_col - block_col) + nj * MMA_N;

                const __nv_bfloat16* b_ptr = &Bs[b_row][b_col];
                wmma::load_matrix_sync(b_frags[nj], b_ptr, BN);
            }

            // MMA: for each MMA tile in warp’s (WM x WN) region
            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                    wmma::mma_sync(c_frags[mi][nj],
                                   a_frags[mi],
                                   b_frags[nj],
                                   c_frags[mi][nj]);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------
    // 3) Store accumulators to C (+ alpha/beta epilogue)
    // ----------------------------
    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            int row = warp_c_row + mi * MMA_M;
            int col = warp_c_col + nj * MMA_N;

            if (row < M && col < N) {
                float* c_ptr = &C[row * ldc + col];

                // Write MMA tile into global C (row-major)
                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj], ldc, wmma::mem_row_major);

                // Apply alpha/beta if you want it fused here:
                // (simplest: post-process in-place)
                for (int i = 0; i < MMA_M; ++i) {
                    int gr = row + i;
                    if (gr >= M) break;
                    for (int j = 0; j < MMA_N; ++j) {
                        int gc = col + j;
                        if (gc >= N) break;
                        int idx = gr * ldc + gc;
                        C[idx] = 1.0 * C[idx] + 0.0 * C[idx];
                    }
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int BM, int BN, int BK, int WM, int WN>
__global__ void
gemm_vector_loads(const __nv_bfloat16* __restrict__ A,
                  const __nv_bfloat16* __restrict__ B,
                  float      * __restrict__ C,
                  int M, int N, int K)
{
    // Tensor Core tile shape (Ampere)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int WARP_SIZE = 32;

    static_assert(WM % MMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % MMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % MMA_K == 0, "BK must be multiple of 16");

    // --- vectorization config for bf16 ---
    constexpr int VEC_ELEMS = 8;          // 8 bf16 per 16-byte vector
    using Vec = uint4;                    // 16-byte raw vector
    static_assert(BK % VEC_ELEMS == 0, "BK must be multiple of vector width");
    static_assert(BN % VEC_ELEMS == 0, "BN must be multiple of vector width");

    // block tile origin in C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // leading dims
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // thread / warp ids
    const int tid     = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = tid / WARP_SIZE;

    // warp tiling in the block
    constexpr int WARPS_PER_BLOCK_N = BN / WN;

    const int warp_row = warp_id / WARPS_PER_BLOCK_N;
    const int warp_col = warp_id % WARPS_PER_BLOCK_N;

    const int warp_c_row = block_row + warp_row * WM;
    const int warp_c_col = block_col + warp_col * WN;

    // Warp tile decomposed into MMA tiles
    constexpr int WARP_M_TILES = WM / MMA_M;
    constexpr int WARP_N_TILES = WN / MMA_N;

    // Shared memory: block tile of A and B
    __shared__ __align__(16) __nv_bfloat16 As[BM][BK];   // M x K
    __shared__ __align__(16) __nv_bfloat16 Bs[BK][BN];   // K x N

    const int num_k_tiles = (K + BK - 1) / BK;

    // Accumulator fragments per warp
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float>
        c_frags[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            wmma::fill_fragment(c_frags[mi][nj], 0.0f);
        }
    }

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        const int k_base = tile_k * BK;

        // ----------------------------
        // 1) load block tile of A and B into shared (vectorized)
        // ----------------------------
        const int block_threads = blockDim.x * blockDim.y;

        // A: BM x BK, laid out row-major in global and in shared
        {
            const int num_vec = (BM * BK) / VEC_ELEMS;

            #pragma unroll
            for (int vi = tid; vi < num_vec; vi += block_threads) {
                int linear_elem = vi * VEC_ELEMS;   // element index in [0, BM*BK)
                int row         = linear_elem / BK; // [0, BM)
                int col         = linear_elem % BK; // [0, BK), step VEC_ELEMS

                int g_row = block_row + row;
                int g_col = k_base    + col;

                // global address: A[g_row * lda + g_col]
                const Vec* src = reinterpret_cast<const Vec*>(
                    &A[g_row * lda + g_col]
                );
                Vec v = *src;

                // shared address: As[row][col] (same row-major layout)
                Vec* dst = reinterpret_cast<Vec*>(
                    &As[row][col]
                );
                *dst = v;
            }
        }

        // B: BK x BN, laid out row-major in global and in shared
        {
            const int num_vec = (BK * BN) / VEC_ELEMS;

            #pragma unroll
            for (int vi = tid; vi < num_vec; vi += block_threads) {
                int linear_elem = vi * VEC_ELEMS;    // element index in [0, BK*BN)
                int row         = linear_elem / BN;  // [0, BK)
                int col         = linear_elem % BN;  // [0, BN), step VEC_ELEMS

                int g_row = k_base    + row;
                int g_col = block_col + col;

                const Vec* src = reinterpret_cast<const Vec*>(
                    &B[g_row * ldb + g_col]
                );
                Vec v = *src;

                Vec* dst = reinterpret_cast<Vec*>(
                    &Bs[row][col]
                );
                *dst = v;
            }
        }

        __syncthreads();

        // ----------------------------
        // 2) warp-level MMA over this K-tile (unchanged)
        // ----------------------------
        for (int kk = 0; kk < BK; kk += MMA_K) {

            // A frags for each "row" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_a,
                           MMA_M, MMA_N, MMA_K,
                           __nv_bfloat16, wmma::row_major> a_frags[WARP_M_TILES];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                int a_row = (warp_c_row - block_row) + mi * MMA_M; // within As
                int a_col = kk;                                    // within As

                const __nv_bfloat16* a_ptr = &As[a_row][a_col];
                wmma::load_matrix_sync(a_frags[mi], a_ptr, BK);
            }

            // B frags for each "column" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_b,
                           MMA_M, MMA_N, MMA_K,
                           __nv_bfloat16, wmma::row_major> b_frags[WARP_N_TILES];

            #pragma unroll
            for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                int b_row = kk;                                    // within Bs
                int b_col = (warp_c_col - block_col) + nj * MMA_N;

                const __nv_bfloat16* b_ptr = &Bs[b_row][b_col];
                wmma::load_matrix_sync(b_frags[nj], b_ptr, BN);
            }

            // MMA: for each MMA tile in warp’s (WM x WN) region
            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                    wmma::mma_sync(c_frags[mi][nj],
                                   a_frags[mi],
                                   b_frags[nj],
                                   c_frags[mi][nj]);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------
    // 3) Store accumulators to C (epilogue – same as you had, or tune later)
    // ----------------------------
    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            int row = warp_c_row + mi * MMA_M;
            int col = warp_c_col + nj * MMA_N;

            if (row < M && col < N) {
                float* c_ptr = &C[row * ldc + col];

                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj],
                                        ldc, wmma::mem_row_major);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int size_sq = 4096;
    if (argc > 1)
        size_sq = std::atoi(argv[1]);
    int M = size_sq;
    int N = size_sq;
    int K = size_sq;
    int iters = 2;

    std::cout << "GEMM benchmark: C = A * B (row-major FP32)\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K
        << ", iters = " << iters << "\n";

    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    size_t bytesA = sizeA * sizeof(__nv_bfloat16);
    size_t bytesB = sizeB * sizeof(__nv_bfloat16);
    size_t bytesC = sizeC * sizeof(float);

    // Host buffers
    std::vector<__nv_bfloat16> hA(sizeA), hB(sizeB);
    std::vector<float> hC(sizeC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i)
        hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i)
        hB[i] = dist(rng);
    std::fill(hC.begin(), hC.end(), 0.0f);

    // Device buffers
    __nv_bfloat16* dA, * dB;
    float * dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    constexpr int WARP_SIZE = 32;

    {
        std::cout << "\n -------------- BF16 tests -------------- \n";

        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        run_cublas_bf16_tc_gemm(handle, M, N, K, dA, dB, dC, iters);
    }

    {
        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 naive_block(32, 1, 1);
        dim3 naive_grid((M - 1)/WMMA_M + 1, (N - 1)/WMMA_N + 1);
        run_wmma_bf16_gemm<wmma_bf16_naive_gemm_kernel>(M, N, K, dA, dB, dC, iters, naive_block, naive_grid, handle, "WMMA naive");
    }

    {
        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 cta_block(32, 4, 4);
        dim3 cta_grid(1, (M - 1)/(WMMA_M*4) + 1, (N - 1)/(WMMA_N*4) + 1);
        run_wmma_bf16_gemm<wmma_bf16_cta>(M, N, K, dA, dB, dC, iters, cta_block, cta_grid, handle, "WMMA CTA");
    }

    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;

        const int WM = 64;
        const int WN = 64;

        const int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);

        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 opt_block(WARP_SIZE, WARPS_PER_BLOCK);
        dim3 opt_grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
        run_wmma_bf16_gemm<gemm_warp_tiling<BM, BN, BK, WM, WN>>(M, N, K, dA, dB, dC, iters, opt_block, opt_grid, handle, "WMMA OPT");
    }

    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;

        const int WM = 64;
        const int WN = 64;

        const int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);

        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 opt_block(WARP_SIZE, WARPS_PER_BLOCK);
        dim3 opt_grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
        run_wmma_bf16_gemm<gemm_vector_loads<BM, BN, BK, WM, WN>>(M, N, K, dA, dB, dC, iters, opt_block, opt_grid, handle, "WMMA OPT");
    }

    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}