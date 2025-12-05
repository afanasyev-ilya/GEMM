// Build example (adjust paths & arch as needed):
//   nvcc -O3 -std=c++20 sgemm.cu -o gemm_bench -I ~/cutlass/include -lcublas -arch=sm_86

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include "macros.cuh"

double BASELINE_TFLOPS = 1;

// -------------------------- cublas performance reference --------------------------

void run_cublas_gemm(cublasHandle_t handle, int M, int N, int K, const float* dA, const float* dB, float* dC, int iters)
{
    // We store row-major A, B, C in C/C++.
    // cuBLAS assumes column-major. To avoid data copies, we use the well-known trick:
    //
    // Row-major C = A * B  (A[M,K], B[K,N], C[M,N])
    //
    // If we treat these same buffers as column-major, they represent the transposed
    // matrices:
    //   A_row (M x K, RM)  <->  A_col^T (K x M, CM)
    //   B_row (K x N, RM)  <->  B_col^T (N x K, CM)
    //   C_row (M x N, RM)  <->  C_col^T (N x M, CM)
    //
    // We want C_row = A_row * B_row.
    // In column-major:
    //   C_col = B_col * A_col   (no transposes), where:
    //     B_col = B_row^T (N x K)
    //     A_col = A_row^T (K x M)
    // Then:
    //   C_col = B_row^T * A_row^T  =>  C_col^T = A_row * B_row
    // And C_col^T is exactly C_row.
    //
    // So we call cublasSgemm with:
    //   m = N, n = M, k = K
    //   A = dB (B_row), lda = N
    //   B = dA (A_row), ldb = K
    //   C = dC, ldc = N

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int m = N;
    int n = M;
    int k = K;
    int lda = N;
    int ldb = K;
    int ldc = N;

    // Warm-up
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        dB, lda,
        dA, ldb,
        &beta,
        dC, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dB, lda,
            dA, ldb,
            &beta,
            dC, ldc));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = total_ms / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    BASELINE_TFLOPS = tflops;

    std::cout << "[cuBLAS SGEMM]\navg time: " << avg_ms << " ms, \n   TFLOP/s: " << tflops << std::endl << std::endl;
}

// -------------------------- wrappers and verification --------------------------

void cublas_ref_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const float* dA,  // row-major A[M,K]
    const float* dB,  // row-major B[K,N]
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

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dB, lda, dA, ldb, &beta, dC_ref, ldc));
}

using KernelFunc = void(*)(const float* __restrict__ ,
                           const float* __restrict__ ,
                           float* __restrict__,
                           int, int, int, float, float);

template <KernelFunc GemmKernel>
double run_custom_sgemm(cublasHandle_t handle,
                        const float* dA, 
                        const float* dB, 
                        float* dC,
                        int M,
                        int N,
                        int K,
                        int iters,
                        dim3 block_size,
                        dim3 grid_size,
                        std::string name,
                        bool verify = true,
                        bool verbose = true)
    {
    float alpha = 1.0;
    float betta = 1.0;

    CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(float)));

    // Warm-up
    SAFE_KERNEL_CALL((GemmKernel<<<grid_size, block_size>>> (dA, dB, dC, M, N, K, alpha, betta)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify
    if(verify) {
        int64_t sizeC = int64_t(M) * N;

        // Compute reference FP32 GEMM with cuBLAS
        float* dC_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
        CHECK_CUDA(cudaMemset(dC_ref, 0, sizeC * sizeof(float)));

        cublas_ref_gemm(handle, M, N, K, dA, dB, dC_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy both results back to host
        std::vector<float> hC(sizeC);
        std::vector<float> hC_ref(sizeC);

        CHECK_CUDA(cudaMemcpy(hC.data(),     dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(dC_ref));

        double max_abs = 0.0;
        double max_rel = 0.0;
        for (int64_t i = 0; i < sizeC; ++i) {
            double ref = hC_ref[i];
            double val = hC[i];
            double diff = std::abs(val - ref);
            double rel  = (std::abs(ref) > 0.0) ? diff / std::abs(ref) : diff;

            max_abs = std::max(diff, max_abs);
            max_rel = std::max(rel, max_rel);
        }

        if(verbose)
            std::cout << "check: max_abs = " << max_abs << ", max_rel = " << max_rel << "   ";

        double tol = 1e-3;  // BF16-level accuracy, adjust if you like
        bool correct = true;
        if (max_abs < tol) {
            if(verbose)
                std::cout << "  (OK)" << std::endl;
        } else {
            if(verbose)
                std::cout << "  (ERROR)" << std::endl;
            correct = false;
        }
        if(!correct) {
            return 0.0;
        }
    }

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start)); 
    for (int i = 0; i < iters; ++i)
    {
        GemmKernel<<<grid_size, block_size>>> (dA, dB, dC, M, N, K, alpha, betta);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = total_ms / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    if(verbose) {
        std::cout << "[" << name << "]\n";
        std::cout << "avg time: " << avg_ms << " ms\n";
        std::cout << tflops << " TFLOP/s (" << 100.0*(tflops/BASELINE_TFLOPS) << "%)" << std::endl << std::endl;
    }
    return tflops;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// main kernels optimized one by one
/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void sgemm_naive(const float * __restrict__ A,
                            const float * __restrict__ B, 
                            float * __restrict__ C, 
                            int M, int N, int K,
                            float alpha, float beta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= N)
        return;
    
    float tmp = 0.0;

    for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
    }

    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void sgemm_coalcesed(const float * __restrict__ A,
                                const float * __restrict__ B, 
                                float * __restrict__ C, 
                                int M, int N, int K,
                                float alpha, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= M)
        return;
    
    float sum = 0.0;

    for (int i = 0; i < K; ++i) {
        // we improved memory access patter here, 
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int TILE_SIZE>
__global__ void sgemm_shared(const float * __restrict__ A,
                             const float * __restrict__ B, 
                             float * __restrict__ C, 
                             int M, int N, int K,
                             float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Global output position
    const int row = blockIdx.y * blockDim.y + thread_row;
    const int col = blockIdx.x * blockDim.x + thread_col;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int num_tiles = (K - 1) / TILE_SIZE + 1;

    float sum = 0.0f;
    
    for (int t = 0; t < num_tiles; t++) {
        if(row < M && (thread_col + t*TILE_SIZE) < K)
            As[thread_row][thread_col] = A[row * K + (thread_col + t*TILE_SIZE)];
        else
            As[thread_row][thread_col] = 0;
        if((thread_row + t*TILE_SIZE) < K && col < N)
            Bs[thread_row][thread_col] = B[(thread_row + t*TILE_SIZE) * N + col];
        else
            Bs[thread_row][thread_col] = 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// idea is https://siboehm.com/assets/img/CUDA-MMM/kernel_4_1D_blocktiling.png
// each thread calculates small TM size column of matrix C elements
template<int BM, int BN, int BK, int ELEM_PER_THREAD>
__global__ void sgemm_1D_blocking(const float * __restrict__ A,
                                  const float * __restrict__ B, 
                                  float * __restrict__ C, 
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y; // should be 0, 8
    const int thread_col = threadIdx.x; // should be 0, 64
    
    // Starting coords of 64x64 output tile for matrix C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // we multiply 64x8 * 8x64 to get 64x64 block. 
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // This results into more K steps then previosly (we used 32), but does not matter much
    const int num_tiles = (K - 1) / BK + 1;

    // ELEM_PER_THREAD = 64 / 8 = BM / block_size.y
    float sums[ELEM_PER_THREAD] = {0};
    
    for (int t = 0; t < num_tiles; t++) {
        int tile_offset = t * BK;
        // since we have 64*8 = 512 threads and shared memory size is 512, we can copy in one pass without loop
        if((block_row + thread_col) < M && (thread_row + tile_offset) < K)
            As[thread_col][thread_row] = A[(block_row + thread_col) * K + (thread_row + tile_offset)];
        else 
            As[thread_col][thread_row] = 0;
        // B access is coalcesed
        if((thread_row + tile_offset) < K && (block_col + thread_col) < N)
            Bs[thread_row][thread_col] = B[(thread_row + tile_offset) * N + (block_col + thread_col)];
        else
            Bs[thread_row][thread_col] = 0;
        __syncthreads();

        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float B_val = Bs[dot_idx][thread_col];
            #pragma unroll
            for (int elt_idx = 0; elt_idx < ELEM_PER_THREAD; ++elt_idx) {
                sums[elt_idx] += As[ELEM_PER_THREAD * thread_row + elt_idx][dot_idx] * B_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int elt_idx = 0; elt_idx < ELEM_PER_THREAD; elt_idx++) {
        // each threads writes back TM elements of matrix C of the same col (adj rows)
        int row = block_row + thread_row * ELEM_PER_THREAD + elt_idx;
        int col = block_col + thread_col;
        if (row < M && col < N) {
            C[row * N + col] = alpha * sums[elt_idx] + beta * C[row * N + col];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// MICRO_M * MICRO_K = elements processed by each thread
template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_2D_blocking(const float * __restrict__ A,
                                  const float * __restrict__ B, 
                                  float * __restrict__ C, 
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Starting coords of output tile for matrix C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // leading dims for simplicity
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // indexes for loading
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0};
    float reg_a[MICRO_M] = {0};
    float reg_b[MICRO_N] = {0};
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        int tile_offset = tile_id * TILE_K;
        // copy A
        #pragma unroll
        for(int i = tid; i < TILE_M*TILE_K; i += block_size) {
            int shared_col = i % TILE_K; // changes in range of [0, TILE_K]
            int shared_row = i / TILE_K; // changes in range of [0, TILE_M]

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;
            float val = 0.0f;
            if(global_col < K && global_row < M)
                val = A[global_row * lda + global_col];
            As[shared_row][shared_col] = val;
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < TILE_K*TILE_N; i += block_size) {
            int shared_col = i % TILE_N; // changes in range of [0, TILE_N]
            int shared_row = i / TILE_N; // changes in range of [0, TILE_K]

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;
            float val = 0.0f;
            if(global_col < N && global_row < K)
                val = B[global_row * ldb + global_col];
            Bs[shared_row][shared_col] = val;
        }

        __syncthreads();

        for(int dot_idx = 0; dot_idx < TILE_K; dot_idx++) {
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_M; elt_idx++) {
                // this is actually most complext thing here
                // similar to 1d, dot_idx runs among cols A
                // and for rows, we just copy MICRO_M elements (row elements per thread)
                reg_a[elt_idx] = As[thread_row * MICRO_M + elt_idx][dot_idx];
            }
            
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_N; elt_idx++) {
                // vise versa but for B matrix
                reg_b[elt_idx] = Bs[dot_idx][thread_col * MICRO_N + elt_idx];
            }

            // actual matmul on registers here
            #pragma unroll
            for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
                #pragma unroll
                for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
                    reg_sums[a_idx][b_idx] += reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
        #pragma unroll
        for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
            int row = block_row + thread_row * MICRO_M + a_idx;
            int col = block_col + thread_col * MICRO_N + b_idx;
            if (row < M && col < N) {
                C[row * ldc + col] = alpha * reg_sums[a_idx][b_idx] + beta * C[row * ldc + col];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_vectorize_smem(const float * __restrict__ A,
                                     const float * __restrict__ B, 
                                     float * __restrict__ C, 
                                     int M, int N, int K,
                                     float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Starting coords of output tile for matrix C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // leading dims for simplicity
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // indexes for loading
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    // change: we transposed As here
    __shared__ float As[TILE_K * TILE_M];
    __shared__ float Bs[TILE_K * TILE_N];

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0};
    float reg_a[MICRO_M] = {0};
    float reg_b[MICRO_N] = {0};
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        int tile_offset = tile_id * TILE_K;
        // copy A
        constexpr int VECTOR_LENGTH = 4;
        
        // we do less loop step now, because of copy is done using vectors
        #pragma unroll
        for(int i = tid; i < (TILE_M*TILE_K) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            // shared cols go with stride 4 (VECTOR_LENGTH) now
            int shared_col = linear_idx % TILE_K; // changes in range of [0, TILE_K], with strides
            int shared_row = linear_idx / TILE_K; // changes in range of [0, TILE_M]

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;
            
            float4 tmp = reinterpret_cast<const float4 *>(&A[global_row * lda + global_col])[0];
            As[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
            As[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
            As[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
            As[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < (TILE_K*TILE_N) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_N; // changes in range of [0, TILE_N], with strides
            int shared_row = linear_idx / TILE_N; // changes in range of [0, TILE_K]

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(&B[global_row * ldb + global_col])[0];
            Bs[(shared_row) * TILE_N + shared_col + 0] = tmp.x;
            Bs[(shared_row) * TILE_N + shared_col + 1] = tmp.y;
            Bs[(shared_row) * TILE_N + shared_col + 2] = tmp.z;
            Bs[(shared_row) * TILE_N + shared_col + 3] = tmp.w;
        }

        __syncthreads();

        for(int dot_idx = 0; dot_idx < TILE_K; dot_idx++) {
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_M; elt_idx++) {
                // this is actually most complext thing here
                // similar to 1d, dot_idx runs among cols A
                // and for rows, we just copy MICRO_M elements (row elements per thread)
                reg_a[elt_idx] = As[dot_idx * TILE_M + thread_row * MICRO_M + elt_idx];
            }
            
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_N; elt_idx++) {
                // vise versa but for B matrix
                reg_b[elt_idx] = Bs[dot_idx * TILE_N + thread_col * MICRO_N + elt_idx];
            }

            // actual matmul on registers here
            #pragma unroll
            for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
                #pragma unroll
                for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
                    reg_sums[a_idx][b_idx] += reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
        #pragma unroll
        for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
            int row = block_row + thread_row * MICRO_M + a_idx;
            int col = block_col + thread_col * MICRO_N + b_idx;
            if (row < M && col < N) {
                C[row * ldc + col] = alpha * reg_sums[a_idx][b_idx] + beta * C[row * ldc + col];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// we swap notation here, since now we have nested tiling
// BM = Block M (number of matrix element processed by block), e.g. 128
// WM = Warp M  (number of matrix element processed by warp), e.g. 32
// TM = Thread M (number of thread elements processed by thread), e.g. 8
// since 32 (warp size) is not SQRT, we need to have rectangular WM tile size, usually 64x32 or 32x64
template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void 
__launch_bounds__(256)
sgemm_warp_tiling(const float * __restrict__ A,
                                  const float * __restrict__ B,
                                  float * __restrict__  C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Starting coords of output tile for matrix C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // leading dims for simplicity
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // indexes for loading
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;
    constexpr int WARP_SIZE = 32;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // seems like only one is needed
    //const int W_ITER_M = BM / WM;
    const int W_ITER_N = BN / WN;

    const int warp_row = warp_id / W_ITER_N;
    const int warp_col = warp_id % W_ITER_N;
    // each thread processes TN by single DIM, simple division
    const int threads_per_warp_row = WN / TN;
    const int thread_row_in_warp = lane_id / threads_per_warp_row;
    const int thread_col_in_warp = lane_id % threads_per_warp_row;

    // change: we transposed As here
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int num_tiles = (K - 1) / BK + 1;

    float reg_sums[TM][TN] = {0};
    float reg_a[TM] = {0};
    float reg_b[TN] = {0};
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        int tile_offset = tile_id * BK;
        // copy A
        constexpr int VECTOR_LENGTH = 4;
        
        // we do less loop step now, because of copy is done using vectors
        #pragma unroll
        for(int i = tid; i < (BM*BK) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            // shared cols go with stride 4 (VECTOR_LENGTH) now
            int shared_col = linear_idx % BK; // changes in range of [0, BK], with strides
            int shared_row = linear_idx / BK; // changes in range of [0, BM]

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;
            
            float4 tmp = reinterpret_cast<const float4 *>(&A[global_row * lda + global_col])[0];
            As[(shared_col + 0)][shared_row] = tmp.x;
            As[(shared_col + 1)][shared_row] = tmp.y;
            As[(shared_col + 2)][shared_row] = tmp.z;
            As[(shared_col + 3)][shared_row] = tmp.w;
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < (BK*BN) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % BN; // changes in range of [0, TN], with strides
            int shared_row = linear_idx / BN; // changes in range of [0, TK]

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(&B[global_row * ldb + global_col])[0];
            Bs[(shared_row)][shared_col + 0] = tmp.x;
            Bs[(shared_row)][shared_col + 1] = tmp.y;
            Bs[(shared_row)][shared_col + 2] = tmp.z;
            Bs[(shared_row)][shared_col + 3] = tmp.w;
        }

        __syncthreads();

        int my_row_in_shared = (warp_row * WM) + (thread_row_in_warp * TM);
        int my_col_in_shared = (warp_col * WN) + (thread_col_in_warp * TN);

        for(int dot_idx = 0; dot_idx < BK; dot_idx++) {
            #pragma unroll
            for(int elt_idx = 0; elt_idx < TM; elt_idx++) {
                // this is actually most complext thing here
                // similar to 1d, dot_idx runs among cols A
                // and for rows, we just copy MICRO_M elements (row elements per thread)
                reg_a[elt_idx] = As[dot_idx][my_row_in_shared + elt_idx];
            }
            
            #pragma unroll
            for(int elt_idx = 0; elt_idx < TN; elt_idx++) {
                // vise versa but for B matrix
                reg_b[elt_idx] = Bs[dot_idx][my_col_in_shared + elt_idx];
            }

            // actual matmul on registers here
            #pragma unroll
            for(int a_idx = 0; a_idx < TM; a_idx++) {
                #pragma unroll
                for(int b_idx = 0; b_idx < TN; b_idx++) {
                    reg_sums[a_idx][b_idx] += reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int m = 0; m < TM; m++) {
        #pragma unroll
        for(int n = 0; n < TN; n++) {
            // just simple nested indexing, like ib grid/block
            int row = block_row + (warp_row * WM) + (thread_row_in_warp * TM) + m;
            int col = block_col + (warp_col * WN) + (thread_col_in_warp * TN) + n;
            if (row < M && col < N) {
                C[row * ldc + col] = alpha * reg_sums[m][n] + beta * C[row * ldc + col];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_double_buffering(const float * __restrict__ A,
                                       const float * __restrict__ B,
                                       float * __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Starting coords of output tile for matrix C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // leading dims for simplicity
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // indexes for loading
    const int tid        = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    // double-buffered shared memory: we transpose As
    __shared__ float As[2][TILE_K * TILE_M];  // [buffer][k * TILE_M + m]
    __shared__ float Bs[2][TILE_K * TILE_N];  // [buffer][k * TILE_N + n]

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0.0f};
    float reg_a[MICRO_M];
    float reg_b[MICRO_N];

    constexpr int VECTOR_LENGTH = 4;

    // -------------------------
    // Preload tile 0 into buffer 0
    // -------------------------
    int buf = 0;
    {
        int tile_offset = 0;

        // copy A for tile 0
        #pragma unroll
        for (int i = tid; i < (TILE_M * TILE_K) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_K;   // [0, TILE_K), stride 4
            int shared_row = linear_idx / TILE_K;   // [0, TILE_M)

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(
                &A[global_row * lda + global_col]
            )[0];

            float *As_buf = &As[buf][0];

            As_buf[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
            As_buf[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
            As_buf[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
            As_buf[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
        }

        // copy B for tile 0
        #pragma unroll
        for (int i = tid; i < (TILE_K * TILE_N) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_N;   // [0, TILE_N), stride 4
            int shared_row = linear_idx / TILE_N;   // [0, TILE_K)

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(
                &B[global_row * ldb + global_col]
            )[0];

            float *Bs_buf = &Bs[buf][0];
            int base = shared_row * TILE_N + shared_col;
            Bs_buf[base + 0] = tmp.x;
            Bs_buf[base + 1] = tmp.y;
            Bs_buf[base + 2] = tmp.z;
            Bs_buf[base + 3] = tmp.w;
        }

        __syncthreads(); // tile 0 ready
    }

    // -------------------------
    // Main loop with double buffering
    // -------------------------
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        int next_tile = tile_id + 1;
        int next_buf  = buf ^ 1;

        // Start loading next tile into the other buffer (if any)
        if (next_tile < num_tiles) {
            int tile_offset = next_tile * TILE_K;

            // copy A for next tile into As[next_buf]
            #pragma unroll
            for (int i = tid; i < (TILE_M * TILE_K) / VECTOR_LENGTH; i += block_size) {
                int linear_idx = i * VECTOR_LENGTH;
                int shared_col = linear_idx % TILE_K;
                int shared_row = linear_idx / TILE_K;

                int global_col = tile_offset + shared_col;
                int global_row = block_row + shared_row;

                float4 tmp = reinterpret_cast<const float4 *>(
                    &A[global_row * lda + global_col]
                )[0];

                float *As_buf = &As[next_buf][0];

                As_buf[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
                As_buf[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
                As_buf[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
                As_buf[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
            }

            // copy B for next tile into Bs[next_buf]
            #pragma unroll
            for (int i = tid; i < (TILE_K * TILE_N) / VECTOR_LENGTH; i += block_size) {
                int linear_idx = i * VECTOR_LENGTH;
                int shared_col = linear_idx % TILE_N;
                int shared_row = linear_idx / TILE_N;

                int global_col = block_col + shared_col;
                int global_row = tile_offset + shared_row;

                float4 tmp = reinterpret_cast<const float4 *>(
                    &B[global_row * ldb + global_col]
                )[0];

                float *Bs_buf = &Bs[next_buf][0];
                int base = shared_row * TILE_N + shared_col;
                Bs_buf[base + 0] = tmp.x;
                Bs_buf[base + 1] = tmp.y;
                Bs_buf[base + 2] = tmp.z;
                Bs_buf[base + 3] = tmp.w;
            }
        }

        // ---- compute using current buffer (As[buf], Bs[buf]) ----
        float *As_cur = &As[buf][0];
        float *Bs_cur = &Bs[buf][0];

        for (int dot_idx = 0; dot_idx < TILE_K; ++dot_idx) {
            #pragma unroll
            for (int elt_idx = 0; elt_idx < MICRO_M; ++elt_idx) {
                reg_a[elt_idx] =
                    As_cur[dot_idx * TILE_M +
                           thread_row * MICRO_M + elt_idx];
            }

            #pragma unroll
            for (int elt_idx = 0; elt_idx < MICRO_N; ++elt_idx) {
                reg_b[elt_idx] =
                    Bs_cur[dot_idx * TILE_N +
                           thread_col * MICRO_N + elt_idx];
            }

            #pragma unroll
            for (int a_idx = 0; a_idx < MICRO_M; ++a_idx) {
                #pragma unroll
                for (int b_idx = 0; b_idx < MICRO_N; ++b_idx) {
                    reg_sums[a_idx][b_idx] +=
                        reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads(); // ensure next tile (if any) is fully loaded
        buf = next_buf;
    }

    #pragma unroll
    for (int a_idx = 0; a_idx < MICRO_M; ++a_idx) {
        #pragma unroll
        for (int b_idx = 0; b_idx < MICRO_N; ++b_idx) {
            int row = block_row + thread_row * MICRO_M + a_idx;
            int col = block_col + thread_col * MICRO_N + b_idx;
            if (row < M && col < N) {
                C[row * ldc + col] =
                    alpha * reg_sums[a_idx][b_idx] +
                    beta  * C[row * ldc + col];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// autotune things here
/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iomanip>

// A container for compile-time integers
template<int... Is>
struct ValueList {};

// A "Compile-Time For Loop"
// It takes a ValueList and a Lambda, and calls the Lambda for every value
template<int... Is, typename Func>
void static_for(ValueList<Is...>, Func&& f) {
    // C++17 Fold Expression: Expands to f.template operator()<Is>() ...
    (f.template operator()<Is>(), ...);
}

inline constexpr int quite_autotune = false;

//#define LARGE_AUTOTUNE_SEARCH_SPACE

#ifdef LARGE_AUTOTUNE_SEARCH_SPACE
// Block sizes to try
using BMs  = ValueList<64, 128>;
using BNs  = ValueList<64, 128>;
using BKs  = ValueList<8, 16>;

// Warp sizes to try
using WMs  = ValueList<16, 32, 64>;
using WNs  = ValueList<16, 32, 64>;

// Thread sizes to try
using TMs  = ValueList<1, 2, 4, 8, 16, 32>;
using TNs  = ValueList<1, 2, 4, 8, 16, 32>;
#else
using BMs  = ValueList<64, 128>;
using BNs  = ValueList<64, 128>;
using BKs  = ValueList<8, 16>;

// Warp sizes to try
using WMs  = ValueList<16, 32, 64>;
using WNs  = ValueList<16, 32, 64>;

// Thread sizes to try
using TMs  = ValueList<4, 8>;
using TNs  = ValueList<4, 8>;
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Config {
    int BM, BN, BK, TM, TN;
    double flops;
};

std::ostream& operator<<(std::ostream& os, const Config& c) {
    os << "Config{"
       << "BM=" << c.BM
       << ", BN=" << c.BN
       << ", BK=" << c.BK
       << ", TM=" << c.TM
       << ", TN=" << c.TN
       << ", flops=" << c.flops
       << "}";
    return os;
}

template<int BM, int BN, int BK, int MICRO_M, int MICRO_N>
double bench_config(cublasHandle_t handle,
                    const float* dA, const float* dB, float* dC,
                    int M, int N, int K, int iters, bool verify)
{
    static_assert(BN % MICRO_N == 0, "BN % MICRO_N != 0");
    static_assert(BM % MICRO_M == 0, "BM % MICRO_M != 0");

    dim3 block_size(BN / MICRO_N, BM / MICRO_M, 1);
    int threads = block_size.x * block_size.y;
    if (threads > 1024) return 0.0;  // invalid on many GPUs

    dim3 grid_size(
        CEIL_DIV(N, block_size.x * MICRO_N),
        CEIL_DIV(M, block_size.y * MICRO_M),
        1
    );

    bool verbose = false;
    double flops = run_custom_sgemm<sgemm_vectorize_smem<BM, BN, BK, MICRO_M, MICRO_N>>(
        handle, dA, dB, dC, M, N, K, iters,
        block_size, grid_size,
        "autotune", verify, verbose
    );
    return flops;
}

Config autotune_sgemm(cublasHandle_t handle,
                      const float* dA, const float* dB, float* dC,
                      int M, int N, int K, int iters, bool verify)
{
    Config best {0,0,0,0,0,0.0};

    std::cout << std::setprecision(3);

    auto update = [&](int BM, int BN, int BK, int TM, int TN, double flops) {
        if(!quite_autotune)
            std::cout << "Tested: " << BM << " " << BN << " " << BK << " | " << TM << " " << TN 
                  << " -> " << flops << " GFLOPS ";
        if (flops > best.flops) {
            if(!quite_autotune)
                std::cout << " (new best found!)" << std::endl;
            best = {BM, BN, BK, TM, TN, flops};
        } else {
            if(!quite_autotune)
                std::cout << std::endl;
        }
    };

    static_for(BMs{}, [&]<int BM>() {
        static_for(BNs{}, [&]<int BN>() {
            static_for(BKs{}, [&]<int BK>() {
                static_for(TMs{}, [&]<int TM>() {
                    static_for(TNs{}, [&]<int TN>() {
                        constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;
                        if constexpr (threads_fit) {
                            // Instantiate the kernel ONLY if valid
                            double g = bench_config<BM, BN, BK, TM, TN>(
                                handle, dA, dB, dC, M, N, K, iters, verify
                            );
                            update(BM, BN, BK, TM, TN, g);
                        }
                    });
                });
            });
        });
    });

    return best;
}

template<typename Cfg>
void run_autotuned(const Cfg& cfg,
                   cublasHandle_t handle,
                   const float* dA, const float* dB, float* dC,
                   int M, int N, int K,
                   int iters, std::string name, bool verify)
{
    bool launched = false;

    static_for(BMs{}, [&]<int BM>() {
        static_for(BNs{}, [&]<int BN>() {
            static_for(BKs{}, [&]<int BK>() {
                static_for(TMs{}, [&]<int TM>() {
                    static_for(TNs{}, [&]<int TN>() {
                        constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;

                        if constexpr (threads_fit) {
                            if (cfg.BM == BM && cfg.BN == BN && cfg.BK == BK && cfg.TM == TM && cfg.TN == TN) {
                                dim3 block_size(BN/TN, BM/TM, 1);
                                dim3 grid_size(CEIL_DIV(N, block_size.x * TN), CEIL_DIV(M, block_size.y * TM), 1);
                                run_custom_sgemm<sgemm_vectorize_smem<BM, BN, BK, TM, TN>>(
                                    handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, name, verify);

                                launched = true;
                            }
                        }
                    });
                });
            });
        });
    });

    if (!launched) {
        throw std::runtime_error("Autotuned cfg is not in the compiled search space");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// ditto for warp tiling
// have to boilerplate because of additonal WM/WN params

struct ConfigWarpTiling {
    int BM, BN, BK, WM, WN, TM, TN;
    double flops;
};

std::ostream& operator<<(std::ostream& os, const ConfigWarpTiling& c) {
    os << "Config{"
       << "BM=" << c.BM
       << ", BN=" << c.BN
       << ", BK=" << c.BK
       << ", WM=" << c.WM
       << ", WN=" << c.WN
       << ", TM=" << c.TM
       << ", TN=" << c.TN
       << ", flops=" << c.flops
       << "}";
    return os;
}

template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
double bench_config_wt(cublasHandle_t handle,
                       const float* dA, const float* dB, float* dC,
                       int M, int N, int K, int iters, bool verify)
{
    try {
        static_assert(BN % TN == 0, "BN % TN != 0");
        static_assert(BM % TM == 0, "BM % TM != 0");
        /*static_assert(BN % WN == 0, "BN % WN != 0");
        static_assert(BM % WM == 0, "BM % WM != 0");
        static_assert(WN % TM == 0, "WN % TM != 0");
        static_assert(WM % TM == 0, "WM % TM != 0");*/

        dim3 block_size(BN / TN, BM / TM, 1);
        int threads = block_size.x * block_size.y;
        if (threads > 1024) 
            return 0.0;  // invalid on many GPUs

        dim3 grid_size(
            CEIL_DIV(N, block_size.x * TN),
            CEIL_DIV(M, block_size.y * TM),
            1
        );

        bool verbose = false;
        double flops = run_custom_sgemm<sgemm_warp_tiling<BM, BN, BK, WM, WN, TM, TN>>(
            handle, dA, dB, dC, M, N, K, iters,
            block_size, grid_size,
            "autotune warp tilig", verify, verbose
        );
        return flops;
    } catch(...) {
        return 0.0;
    }
}

ConfigWarpTiling autotune_sgemm_wt(cublasHandle_t handle,
                         const float* dA, const float* dB, float* dC,
                         int M, int N, int K, int iters, bool verify)
{
    ConfigWarpTiling best {0,0,0,0,0,0,0,0.0};

    std::cout << std::setprecision(3);

    auto update = [&](int BM, int BN, int BK, int WM, int WN, int TM, int TN, double flops) {
        if(!quite_autotune)
            std::cout << "Tested: " << BM << " " << BN << " " << BK << " | " 
                    << WM << " " << WN << " | " << TM << " " << TN 
                    << " -> " << flops << " GFLOPS";
        if (flops > best.flops) {
            if(!quite_autotune)
                std::cout << " (new best found!)" << std::endl;
            best = {BM, BN, BK, WM, WN, TM, TN, flops};
        } else {
            if(!quite_autotune)
                std::cout << std::endl;
        }
    };

    static_for(BMs{}, [&]<int BM>() {
        static_for(BNs{}, [&]<int BN>() {
            static_for(BKs{}, [&]<int BK>() {
                static_for(WMs{}, [&]<int WM>() {
                    static_for(WNs{}, [&]<int WN>() {
                        static_for(TMs{}, [&]<int TM>() {
                            static_for(TNs{}, [&]<int TN>() {
                                constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;
                                constexpr bool warp_valid  = (WM / TM) * (WN / TN) == 32;
                                constexpr bool warp_fits_block = (BM >= WM) && (BN >= WN);

                                if constexpr (threads_fit && warp_valid && warp_fits_block) {
                                    // Instantiate the kernel ONLY if valid
                                    double g = bench_config_wt<BM, BN, BK, WM, WN, TM, TN>(
                                        handle, dA, dB, dC, M, N, K, iters, verify
                                    );
                                    update(BM, BN, BK, WM, WN, TM, TN, g);
                                }
                            });
                        });
                    });
                });
            });
        });
    });

    return best;
}

template<typename Cfg>
void run_autotuned_wt(const Cfg& cfg,
                      cublasHandle_t handle,
                      const float* dA, const float* dB, float* dC,
                      int M, int N, int K,
                      int iters, std::string name, bool verify)
{
    bool launched = false;

    static_for(BMs{}, [&]<int BM>() {
        static_for(BNs{}, [&]<int BN>() {
            static_for(BKs{}, [&]<int BK>() {
                static_for(WMs{}, [&]<int WM>() {
                    static_for(WNs{}, [&]<int WN>() {
                        static_for(TMs{}, [&]<int TM>() {
                            static_for(TNs{}, [&]<int TN>() {
                                constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;
                                constexpr bool warp_valid  = (WM / TM) * (WN / TN) == 32;
                                constexpr bool warp_fits_block = (BM >= WM) && (BN >= WN);

                                if constexpr (threads_fit && warp_valid && warp_fits_block) {
                                    if (cfg.BM == BM && cfg.BN == BN && cfg.BK == BK && cfg.WM == WM && cfg.WN == WN && cfg.TM == TM && cfg.TN == TN) {
                                        dim3 block_size(BN/TN, BM/TM, 1);
                                        dim3 grid_size(CEIL_DIV(N, block_size.x * TN), CEIL_DIV(M, block_size.y * TM), 1);
                                        run_custom_sgemm<sgemm_warp_tiling<BM, BN, BK, WM, WN, TM, TN>>(
                                            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, name, verify);
                                        launched = true;
                                    }
                                }
                            });
                        });
                    });
                });
            });
        });
    });

    if (!launched) {
        throw std::runtime_error("Autotuned cfg is not in the compiled search space");
    }
}

// --------------------------------------- Main -------------------------------------------

int main(int argc, char** argv)
{
    const bool verify = true;
    const bool run_slow = false;

    int size_sq = 4096;

    if (argc > 1)
        size_sq = std::atoi(argv[1]);
    int M = size_sq;
    int N = size_sq;
    int K = size_sq;

    int iters = 10;
    if (argc > 2)
        iters = std::atoi(argv[2]);

    std::cout << "SGEMM benchmark: C = A * B (row-major FP32)\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K
        << ", iters = " << iters << std::endl << std::endl;

    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    size_t bytesA = sizeA * sizeof(float);
    size_t bytesB = sizeB * sizeof(float);
    size_t bytesC = sizeC * sizeof(float);

    // Host buffers
    std::vector<float> hA(sizeA), hB(sizeB), hC(sizeC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i)
        hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i)
        hB[i] = dist(rng);
    std::fill(hC.begin(), hC.end(), 0.0f);

    // Device buffers
    float* dA, * dB, * dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cublas reference perf
    {
        run_cublas_gemm(handle, M, N, K, dA, dB, dC, iters);
    }

    if(run_slow)
    {
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_naive>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "naive global", verify);
    }

    if(run_slow)
    {
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_coalcesed>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "coalcesed global", verify);
    }

    if(run_slow)
    {
        const int TILE_SIZE = 32;
        dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_shared<TILE_SIZE>>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "shared", verify);
    }

    auto run_1d_blocking = [&]<int ELEM_PER_THREAD>() {
        std::cout << "using " << ELEM_PER_THREAD << " element per thread:\n";
        const int BM = 64;
        const int BN = 64;
        assert(BM == BN);
        const int BK = BM / ELEM_PER_THREAD;
        
        dim3 block_size(BM, BK, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y * ELEM_PER_THREAD), 1);
        run_custom_sgemm<sgemm_1D_blocking<BM, BN, BK, ELEM_PER_THREAD>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "1D blocking", verify);
    };

    run_1d_blocking.operator()<4>();
    run_1d_blocking.operator()<8>();
    run_1d_blocking.operator()<16>();
    run_1d_blocking.operator()<32>();

    {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_2D_blocking<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "2D blocking", verify);
    }

    {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_vectorize_smem<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "vectorize shmem", verify);
    }

    // do autotuning
    {   
        std::cout << "Autotuning in process..." << std::endl;
        auto cfg = autotune_sgemm(handle, dA, dB, dC, M, N, K, iters, verify);
        std::cout << "done!" << std::endl;
        std::cout << "best config: " << cfg << std::endl << std::endl;

        run_autotuned(cfg, handle, dA, dB, dC, M, N, K, iters, "autotuned", verify);
    }

    {   
        std::cout << "WARP TILING Autotuning in process..." << std::endl;
        auto cfg = autotune_sgemm_wt(handle, dA, dB, dC, M, N, K, iters, verify);
        std::cout << "done!" << std::endl;
        std::cout << "warp tiling best config: " << cfg << std::endl << std::endl;

        run_autotuned_wt(cfg, handle, dA, dB, dC, M, N, K, iters, "warp tiling with autotune", verify);
    }

    {   
        // results for RTX 3060
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;
        const int MICRO_M = 8;
        const int MICRO_N = 8;

        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_double_buffering<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "double buffering(DB)", verify);
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}