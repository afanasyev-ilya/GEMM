#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include "macros.cuh"
#include "bench_common.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

// Tensor Core tile shape (Ampere)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Each block is a single warp (32 threads).
// Each warp computes one 16x16 tile of C.
__global__ void wmma_bf16_naive_kernel(const __nv_bfloat16* __restrict__ A,
                                       const __nv_bfloat16* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta)
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

__global__ void wmma_bf16_cta_kernel(const __nv_bfloat16* __restrict__ A,
                                     const __nv_bfloat16* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K,
                                     float alpha, float beta)
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
wmma_bf16_gemm_warp_tiling_kernel(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B,
                                  float      * __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta)
{
    static_assert(WM % WMMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % WMMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % WMMA_K == 0, "BK must be multiple of 16");

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
    constexpr int WARP_M_TILES = WM / WMMA_M;
    constexpr int WARP_N_TILES = WN / WMMA_N;

    // Shared memory: block tile of A and B
    __shared__ __nv_bfloat16 As[BM][BK];   // M x K
    __shared__ __nv_bfloat16 Bs[BK][BN];   // K x N

    const int num_k_tiles = (K + BK - 1) / BK;

    // Accumulator fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
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
        for (int kk = 0; kk < BK; kk += WMMA_K) {

            // A frags for each "row" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frags[WARP_M_TILES];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                int a_row = (warp_c_row - block_row) + mi * WMMA_M; // within As
                int a_col = kk;                                    // within As

                const __nv_bfloat16* a_ptr = &As[a_row][a_col];
                wmma::load_matrix_sync(a_frags[mi], a_ptr, BK);
            }

            // B frags for each "column" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> b_frags[WARP_N_TILES];

            #pragma unroll
            for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                int b_row = kk;                                    // within Bs
                int b_col = (warp_c_col - block_col) + nj * WMMA_N;

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
            int row = warp_c_row + mi * WMMA_M;
            int col = warp_c_col + nj * WMMA_N;

            // safest guard for WMMA load/store
            if (row + WMMA_M <= M && col + WMMA_N <= N) {
                float* c_ptr = &C[row * ldc + col];

                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
                wmma::load_matrix_sync(c_old, c_ptr, ldc, wmma::mem_row_major);

                #pragma unroll
                for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                    c_frags[mi][nj].x[e] =
                        alpha * c_frags[mi][nj].x[e] + beta * c_old.x[e];
                }

                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj], ldc, wmma::mem_row_major);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int BM, int BN, int BK, int WM, int WN>
__global__ void
wmma_bf16_gemm_vector_loads_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   float      * __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    static_assert(WM % WMMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % WMMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % WMMA_K == 0, "BK must be multiple of 16");

    // --- vectorization config for bf16 ---
    // unit4 is vector of 4 int elements, 4 * sizeof(int) = 16 byte
    // this means 16 / sizeof(bf16) = 8 vectorized element loads for one instruction
    // unit64 is just a struct, something like this:
    // typedef struct {
    //     unsigned int x, y, z, w;
    // } uint4;
    // this way we achieve 16 byte * 8 = 128 bit copy, longest possible in CUDA using sinle vector instruction
    constexpr int VECTOR_LENGTH = 8;          // 8 bf16 per 16-byte vector
    using Vec = uint4;                    // 16-byte raw vector

    static_assert(BK % VECTOR_LENGTH == 0, "BK must be multiple of vector width");
    static_assert(BN % VECTOR_LENGTH == 0, "BN must be multiple of vector width");

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
    constexpr int WARP_M_TILES = WM / WMMA_M;
    constexpr int WARP_N_TILES = WN / WMMA_N;

    // Shared memory: block tile of A and B
    __shared__ __align__(16) __nv_bfloat16 As[BM][BK];   // M x K
    __shared__ __align__(16) __nv_bfloat16 Bs[BK][BN];   // K x N

    const int num_k_tiles = (K + BK - 1) / BK;

    // Accumulator fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
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
            const int num_vec = (BM * BK) / VECTOR_LENGTH;

            #pragma unroll
            for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_threads) {
                int linear_elem = vec_idx * VECTOR_LENGTH;   // element index in [0, BM*BK)
                int row         = linear_elem / BK; // [0, BM)
                int col         = linear_elem % BK; // [0, BK), step VECTOR_LENGTH

                int global_row = block_row + row;
                int global_col = k_base    + col;

                // global address: A[g_row * lda + g_col]
                const Vec* src = reinterpret_cast<const Vec*>(&A[global_row * lda + global_col]);
                Vec v = *src;

                // shared address: As[row][col] (same row-major layout)
                Vec* dst = reinterpret_cast<Vec*>(&As[row][col]);
                *dst = v;
            }
        }

        // B: BK x BN, laid out row-major in global and in shared
        {
            const int num_vec = (BK * BN) / VECTOR_LENGTH;

            #pragma unroll
            for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_threads) {
                int linear_elem = vec_idx * VECTOR_LENGTH;    // element index in [0, BK*BN)
                int row         = linear_elem / BN;  // [0, BK)
                int col         = linear_elem % BN;  // [0, BN), step VECTOR_LENGTH

                int global_row = k_base    + row;
                int global_col = block_col + col;

                const Vec* src = reinterpret_cast<const Vec*>(&B[global_row * ldb + global_col]);
                Vec v = *src;

                Vec* dst = reinterpret_cast<Vec*>(&Bs[row][col]);
                *dst = v;
            }
        }

        __syncthreads();

        // ----------------------------
        // 2) warp-level MMA over this K-tile (unchanged)
        // ----------------------------
        for (int kk = 0; kk < BK; kk += WMMA_K) {

            // A frags for each "row" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frags[WARP_M_TILES];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                int a_row = (warp_c_row - block_row) + mi * WMMA_M; // within As
                int a_col = kk;                                    // within As

                const __nv_bfloat16* a_ptr = &As[a_row][a_col];
                wmma::load_matrix_sync(a_frags[mi], a_ptr, BK);
            }

            // B frags for each "column" of MMA tiles in this warp tile
            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> b_frags[WARP_N_TILES];

            #pragma unroll
            for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                int b_row = kk;                                    // within Bs
                int b_col = (warp_c_col - block_col) + nj * WMMA_N;

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
            int row = warp_c_row + mi * WMMA_M;
            int col = warp_c_col + nj * WMMA_N;

            // safest guard for WMMA load/store
            if (row + WMMA_M <= M && col + WMMA_N <= N) {
                float* c_ptr = &C[row * ldc + col];

                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
                wmma::load_matrix_sync(c_old, c_ptr, ldc, wmma::mem_row_major);

                #pragma unroll
                for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                    c_frags[mi][nj].x[e] =
                        alpha * c_frags[mi][nj].x[e] + beta * c_old.x[e];
                }

                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj], ldc, wmma::mem_row_major);
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

    std::cout << "BF16 GEMM benchmark: C = A * B (row-major FP32)\n";
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

    float alpha = 1.0;
    float beta = 0.0;

    {
        std::cout << "\n -------------- BF16 tests -------------- \n";

        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        run_cublas_bf16_tc_gemm(handle, M, N, K, dA, dB, dC, iters);
    }

    {
        dim3 naive_block(32, 1, 1);
        dim3 naive_grid((M - 1)/WMMA_M + 1, (N - 1)/WMMA_N + 1);

        auto launch = make_launcher<wmma_bf16_naive_kernel>(naive_grid, naive_block);
        run_gemm_bench<__nv_bfloat16>(handle, M, N, K, dA, dB, dC, iters, launch, "WMMA naive", alpha, beta);
    }

    {
        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 cta_block(32, 4, 4);
        dim3 cta_grid(1, (M - 1)/(WMMA_M*4) + 1, (N - 1)/(WMMA_N*4) + 1);

        auto launch = make_launcher<wmma_bf16_cta_kernel>(cta_grid, cta_block);
        run_gemm_bench<__nv_bfloat16>(handle, M, N, K, dA, dB, dC, iters, launch, "WMMA CTA", alpha, beta);
    }

    {
        //wmma_bf16_gemm_vector_loads_kernel
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;

        const int WM = 64;
        const int WN = 64;

        const int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);

        CHECK_CUDA(cudaMemset(dC, 0, bytesC));
        dim3 opt_block(WARP_SIZE, WARPS_PER_BLOCK);
        dim3 opt_grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

        {
            auto launch = make_launcher<wmma_bf16_gemm_warp_tiling_kernel<BM, BN, BK, WM, WN>>(opt_grid, opt_block);
            run_gemm_bench<__nv_bfloat16>(handle, M, N, K, dA, dB, dC, iters, launch, "WMMA warp tiling", alpha, beta);
        }

        {
            auto launch = make_launcher<wmma_bf16_gemm_vector_loads_kernel<BM, BN, BK, WM, WN>>(opt_grid, opt_block);
            run_gemm_bench<__nv_bfloat16>(handle, M, N, K, dA, dB, dC, iters, launch, "WMMA vector loads", alpha, beta);
        }
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}