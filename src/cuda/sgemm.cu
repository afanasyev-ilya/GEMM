#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include "macros.cuh"
#include "bench_common.cuh"

double BASELINE_TFLOPS = 1;

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

#include "autotune.cuh"

// original large search space
/*using BMs  = ValueList<64, 128>;
using BNs  = ValueList<64, 128>;
using BKs  = ValueList<8, 16>;

using WMs  = ValueList<16, 32, 64>;
using WNs  = ValueList<16, 32, 64>;

using TMs  = ValueList<1, 2, 4, 8, 16, 32>;
using TNs  = ValueList<1, 2, 4, 8, 16, 32>;*/

// search space for faster compilation and demo
using BMs  = ValueList<128>;
using BNs  = ValueList<128>;
using BKs  = ValueList<16, 32>;

using WMs  = ValueList<16, 32>;
using WNs  = ValueList<16, 32>;

using TMs  = ValueList<4, 8>;
using TNs  = ValueList<4, 8>;

#include "autotune_specs.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

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
        run_cublas_rowmajor_gemm<float>(handle, M, N, K, dA, dB, dC, iters,  "cuBLAS SGEMM", /*update_baseline=*/true);
    }

    if(run_slow)
    {
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        
        auto launch = make_launcher<sgemm_naive>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "naive global", 1.0f, 0.0f, verify, true);
    }

    if(run_slow)
    {
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);

        auto launch = make_launcher<sgemm_coalcesed>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "coalcesed global", 1.0f, 0.0f, verify, true);
    }

    if(run_slow)
    {
        const int TILE_SIZE = 32;
        dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);

        auto launch = make_launcher<sgemm_shared<TILE_SIZE>>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "shared", 1.0f, 0.0f, verify, true);
    }

    auto run_1d_blocking = [&]<int ELEM_PER_THREAD>() {
        std::cout << "using " << ELEM_PER_THREAD << " element per thread:\n";
        const int BM = 64;
        const int BN = 64;
        assert(BM == BN);
        const int BK = BM / ELEM_PER_THREAD;
        
        dim3 block_size(BM, BK, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y * ELEM_PER_THREAD), 1);

        auto launch = make_launcher<sgemm_1D_blocking<BM, BN, BK, ELEM_PER_THREAD>>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "1D blocking", 1.0f, 0.0f, verify, true);
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

        auto launch = make_launcher<sgemm_2D_blocking<BM, BN, BK, MICRO_M, MICRO_N>>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "2D blocking", 1.0f, 0.0f, verify, true);
    }

    {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        
        auto launch = make_launcher<sgemm_vectorize_smem<BM, BN, BK, MICRO_M, MICRO_N>>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "vectorize shmem", 1.0f, 0.0f, verify, true);
    }

    // do autotuning for vectorize smem version
    {   
        auto cfg = autotune_generic<float, VecSmemSpec>(handle, dA, dB, dC, M, N, K, iters, verify);
        run_autotuned_generic<float, VecSmemSpec>(cfg, handle, dA, dB, dC, M, N, K, iters, "autotuned vectorized copy", verify);
    }

    // da autotune for warp tiling
    {   
        auto cfg_wt = autotune_generic<float, WarpTilingSpec>(handle, dA, dB, dC, M, N, K, iters, verify);
        run_autotuned_generic<float, WarpTilingSpec>(cfg_wt, handle, dA, dB, dC, M, N, K, iters, "warp tiling", verify);
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
        auto launch = make_launcher<sgemm_double_buffering<BM, BN, BK, MICRO_M, MICRO_N>>(grid_size, block_size);
        run_gemm_bench<float>(handle, M, N, K, dA, dB, dC, iters, launch, "double buffering(DB)", 1.0f, 0.0f, verify, true);
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}