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

#include "macros.cuh"
#include "bench_common.cuh"

// Tensor Core tile shape (Ampere)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

// Device helpers (Ampere has __nv_bfloat16 + conversions)
__device__ inline void fp32_to_bf16x3(float a,
                                      __nv_bfloat16 &a0,
                                      __nv_bfloat16 &a1,
                                      __nv_bfloat16 &a2)
{
    // Handle trivial / special cases explicitly
    if (a == 0.0f || !isfinite(a)) {
        a0 = __float2bfloat16(a);                // zero, inf, nan
        a1 = __float2bfloat16(0.0f);
        a2 = __float2bfloat16(0.0f);
        return;
    }

    // First digit: plain BF16 rounding
    a0 = __float2bfloat16(a);
    float f0 = __bfloat162float(a0);

    // Extract residual, shift left by 8 bits (because BF16 has 7 mantissa bits)
    float r1 = ldexpf(a - f0, 8);               // (a - a0)*2^8
    a1 = __float2bfloat16(r1);
    float f1 = __bfloat162float(a1);

    // Second residual, again shift by 8 bits
    float r2 = ldexpf(r1 - f1, 8);
    a2 = __float2bfloat16(r2);
    // Now: a  ≈  f0 + 2^-8 f1 + 2^-16 f2
    // and a0,a1,a2 are the BF16 encodings of f0,f1,f2
}


__global__ void
slice_fp32_to_bf16x3(const float* __restrict__ A,
                     __nv_bfloat16* __restrict__ A0,
                     __nv_bfloat16* __restrict__ A1,
                     __nv_bfloat16* __restrict__ A2,
                     int rows, int cols, int lda, int lda_bf16)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    float a = A[row * lda + col];

    __nv_bfloat16 a0, a1, a2;
    fp32_to_bf16x3(a, a0, a1, a2);

    int idx = row * lda_bf16 + col;
    A0[idx] = a0;
    A1[idx] = a1;
    A2[idx] = a2;
}

template<int BM, int BN, int BK, int WM, int WN>
__global__ void
wmma_bf16_gemm_async_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   float      * __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    constexpr int WARP_SIZE = 32;

    static_assert(WM % WMMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % WMMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % WMMA_K == 0, "BK must be multiple of 16");

    // --- vectorization config for bf16 ---
    constexpr int VECTOR_LENGTH = 8;     // 8 bf16 per 16-byte vector
    using Vec = uint4;                   // 16-byte raw vector

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

    // -----------------------------------------------------------------------
    // Shared memory: **double-buffered** block tiles of A and B
    // As[stage][BM][BK], Bs[stage][BK][BN], stage ∈ {0,1}
    // -----------------------------------------------------------------------
    __shared__ __align__(16) __nv_bfloat16 As[2][BM][BK];   // 2 x (BM x BK)
    __shared__ __align__(16) __nv_bfloat16 Bs[2][BK][BN];   // 2 x (BK x BN)

    const int num_k_tiles = (K + BK - 1) / BK;

    const int block_threads = blockDim.x * blockDim.y;
    const int num_vec_A = (BM * BK) / VECTOR_LENGTH;
    const int num_vec_B = (BK * BN) / VECTOR_LENGTH;

    // -----------------------------------------------------------------------
    // Accumulator fragments per warp
    // -----------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frags[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            wmma::fill_fragment(c_frags[mi][nj], 0.0f);
        }
    }

    // -----------------------------------------------------------------------
    // Lambda: issue cp.async copies for tile `tile_k` into shared buffer `stage`
    // (no commit/wait inside; those are done outside around the compute)
    // -----------------------------------------------------------------------

    auto cp_async_load_tile = [&](int stage, int tile_k) {
        const int k_base = tile_k * BK;

        // ---- A tile: BM x BK ----
        #pragma unroll
        for (int vec_idx = tid; vec_idx < num_vec_A; vec_idx += block_threads) {
            int linear_elem = vec_idx * VECTOR_LENGTH; // [0, BM*BK)
            int row         = linear_elem / BK;        // [0, BM)
            int col         = linear_elem % BK;        // [0, BK), step VECTOR_LENGTH

            int global_row = block_row + row;
            int global_col = k_base    + col;

            const __nv_bfloat16* g_ptr = &A[global_row * lda + global_col];
            const __nv_bfloat16* s_ptr = &As[stage][row][col];

            // shared memory address as 32-bit
            unsigned int smem_addr = static_cast<unsigned int>(
                __cvta_generic_to_shared(const_cast<__nv_bfloat16*>(s_ptr))
            );

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem_addr), "l"(g_ptr)
            );
        }

        // ---- B tile: BK x BN ----
        #pragma unroll
        for (int vec_idx = tid; vec_idx < num_vec_B; vec_idx += block_threads) {
            int linear_elem = vec_idx * VECTOR_LENGTH; // [0, BK*BN)
            int row         = linear_elem / BN;        // [0, BK)
            int col         = linear_elem % BN;        // [0, BN), step VECTOR_LENGTH

            int global_row = k_base    + row;
            int global_col = block_col + col;

            const __nv_bfloat16* g_ptr = &B[global_row * ldb + global_col];
            const __nv_bfloat16* s_ptr = &Bs[stage][row][col];

            unsigned int smem_addr = static_cast<unsigned int>(
                __cvta_generic_to_shared(const_cast<__nv_bfloat16*>(s_ptr))
            );

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem_addr), "l"(g_ptr)
            );
        }
    };

    // -----------------------------------------------------------------------
    // K-tile loop with double buffering and async copies
    // -----------------------------------------------------------------------

    if (num_k_tiles > 0) {
        // Preload first K-tile into stage 0
        cp_async_load_tile(/*stage=*/0, /*tile_k=*/0);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        const int stage       = tile_k & 1;        // current stage
        const int next_stage  = stage ^ 1;         // other stage

        // Preload next tile (tile_k + 1) into the other stage while we compute
        if (tile_k + 1 < num_k_tiles) {
            cp_async_load_tile(next_stage, tile_k + 1);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // ----------------------------
        // 2) warp-level MMA over this K-tile (uses As[stage], Bs[stage])
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

                const __nv_bfloat16* a_ptr = &As[stage][a_row][a_col];
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

                const __nv_bfloat16* b_ptr = &Bs[stage][b_row][b_col];
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

        // Ensure next tile's async copies are done before we use it
        if (tile_k + 1 < num_k_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
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

            if (row + WMMA_M <= M && col + WMMA_N <= N) {
                float* c_ptr = &C[row * ldc + col];

                if(beta == 0) {
                    #pragma unroll
                    for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                        c_frags[mi][nj].x[e] =
                            alpha * c_frags[mi][nj].x[e];
                    }
                } else {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
                    wmma::load_matrix_sync(c_old, c_ptr, ldc, wmma::mem_row_major);

                    #pragma unroll
                    for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                        c_frags[mi][nj].x[e] =
                            alpha * c_frags[mi][nj].x[e] + beta * c_old.x[e];
                    }
                }

                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj], ldc, wmma::mem_row_major);
            }
        }
    }
}


template<int BM, int BN, int BK, int WM, int WN>
__global__ void
wmma_fp32_emulated(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float      * __restrict__ C,
                   int M, int N, int K,
                   float alpha, float beta)
{
    constexpr int WARP_SIZE = 32;

    static_assert(WM % WMMA_M == 0, "WM must be multiple of 16");
    static_assert(WN % WMMA_N == 0, "WN must be multiple of 16");
    static_assert(BK % WMMA_K == 0, "BK must be multiple of 16");

    // --- vectorization config for fp32 ---
    constexpr int VECTOR_LENGTH = 4;     // 8 bf16 per 16-byte vector
    using Vec = uint4;                   // 16-byte raw vector

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

    __shared__ __align__(16) float As_f[2][BM][BK];
    __shared__ __align__(16) __nv_bfloat16 A0[2][BM][BK];
    __shared__ __align__(16) __nv_bfloat16 A1[2][BK][BN];
    __shared__ __align__(16) __nv_bfloat16 A2[2][BK][BN];

    __shared__ __align__(16) float Bs_f[2][BK][BN];
    __shared__ __align__(16) __nv_bfloat16 B0[2][BK][BN];
    __shared__ __align__(16) __nv_bfloat16 B1[2][BK][BN];
    __shared__ __align__(16) __nv_bfloat16 B2[2][BK][BN];

    const int num_k_tiles = (K + BK - 1) / BK;

    const int block_threads = blockDim.x * blockDim.y;
    const int num_vec_A = (BM * BK) / VECTOR_LENGTH;
    const int num_vec_B = (BK * BN) / VECTOR_LENGTH;

    // -----------------------------------------------------------------------
    // Accumulator fragments
    // -----------------------------------------------------------------------

    using acc_frag_t =
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    acc_frag_t c_frags [WARP_M_TILES][WARP_N_TILES]; // G0
    acc_frag_t g1_frags[WARP_M_TILES][WARP_N_TILES]; // G1 = A0B1 + A1B0
    acc_frag_t g2_frags[WARP_M_TILES][WARP_N_TILES]; // G2 = A0B2 + A1B1 + A2B0
    acc_frag_t g3_frags[WARP_M_TILES][WARP_N_TILES]; // G3 = A1B2 + A2B1
    acc_frag_t g4_frags[WARP_M_TILES][WARP_N_TILES]; // G4 = A2B2

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            wmma::fill_fragment(c_frags [mi][nj], 0.0f);
            wmma::fill_fragment(g1_frags[mi][nj], 0.0f);
            wmma::fill_fragment(g2_frags[mi][nj], 0.0f);
            wmma::fill_fragment(g3_frags[mi][nj], 0.0f);
            wmma::fill_fragment(g4_frags[mi][nj], 0.0f);
        }
    }

    // -----------------------------------------------------------------------
    // Lambda: issue cp.async copies for tile `tile_k` into shared buffer `stage`
    // (no commit/wait inside; those are done outside around the compute)
    // -----------------------------------------------------------------------
    auto cp_async_load_fp32 = [&](int stage, int tile_k) {
        const int k_base = tile_k * BK;

        // ---- A tile: BM x BK ----
        #pragma unroll
        for (int vec_idx = tid; vec_idx < num_vec_A; vec_idx += block_threads) {
            int linear_elem = vec_idx * VECTOR_LENGTH; // [0, BM*BK)
            int row         = linear_elem / BK;        // [0, BM)
            int col         = linear_elem % BK;        // [0, BK), step VECTOR_LENGTH

            int global_row = block_row + row;
            int global_col = k_base    + col;

            const float* g_ptr = &A[global_row * lda + global_col];
            const float* s_ptr = &As_f[stage][row][col];

            // shared memory address as 32-bit
            unsigned int smem_addr = static_cast<unsigned int>(
                __cvta_generic_to_shared(const_cast<float*>(s_ptr))
            );

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem_addr), "l"(g_ptr)
            );
        }

        // ---- B tile: BK x BN ----
        #pragma unroll
        for (int vec_idx = tid; vec_idx < num_vec_B; vec_idx += block_threads) {
            int linear_elem = vec_idx * VECTOR_LENGTH; // [0, BK*BN)
            int row         = linear_elem / BN;        // [0, BK)
            int col         = linear_elem % BN;        // [0, BN), step VECTOR_LENGTH

            int global_row = k_base    + row;
            int global_col = block_col + col;

            const float* g_ptr = &B[global_row * ldb + global_col];
            const float* s_ptr = &Bs_f[stage][row][col];

            unsigned int smem_addr = static_cast<unsigned int>(
                __cvta_generic_to_shared(const_cast<float*>(s_ptr))
            );

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem_addr), "l"(g_ptr)
            );
        }
    };

    auto convert_tile_to_digits = [&](int stage) {
        const int tile_elems_A = BM * BK;
        for (int idx = tid; idx < tile_elems_A; idx += block_threads) {
            int row = idx / BK;
            int col = idx % BK;

            float a = As_f[stage][row][col];

            __nv_bfloat16 d0, d1, d2;
            fp32_to_bf16x3(a, d0, d1, d2);

            A0[stage][row][col] = d0;
            A1[stage][row][col] = d1;
            A2[stage][row][col] = d2;
        }

        const int tile_elems_B = BK * BN;
        for (int idx = tid; idx < tile_elems_B; idx += block_threads) {
            int row = idx / BN;
            int col = idx % BN;

            float b = Bs_f[stage][row][col];

            __nv_bfloat16 d0, d1, d2;
            fp32_to_bf16x3(b, d0, d1, d2);

            B0[stage][row][col] = d0;
            B1[stage][row][col] = d1;
            B2[stage][row][col] = d2;
        }

        __syncthreads(); // ensure all digits are ready before WMMA uses them
    };

    // -----------------------------------------------------------------------
    // K-tile loop with double buffering and async copies
    // -----------------------------------------------------------------------

    if (num_k_tiles > 0) {
        // Preload first K-tile into stage 0
        cp_async_load_fp32(/*stage=*/0, /*tile_k=*/0);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        const int stage       = tile_k & 1;        // current stage
        const int next_stage  = stage ^ 1;         // other stage

        // Preload next tile (tile_k + 1) into the other stage while we compute
        if (tile_k + 1 < num_k_tiles) {
            cp_async_load_fp32(next_stage, tile_k + 1);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        convert_tile_to_digits(stage);

        // --- WMMA multiply-adds for this K-tile (stage) ---
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) { 
            // A frags per "row" of MMA tiles in this warp tile, 3 digits
            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> 
                a0_frags[WARP_M_TILES],
                a1_frags[WARP_M_TILES],
                a2_frags[WARP_M_TILES];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                int a_row = (warp_c_row - block_row) + mi * WMMA_M; // within [0, BM)
                int a_col = kk;                                     // within [0, BK)

                const __nv_bfloat16* a0_ptr = &A0[stage][a_row][a_col];
                const __nv_bfloat16* a1_ptr = &A1[stage][a_row][a_col];
                const __nv_bfloat16* a2_ptr = &A2[stage][a_row][a_col];

                wmma::load_matrix_sync(a0_frags[mi], a0_ptr, BK);
                wmma::load_matrix_sync(a1_frags[mi], a1_ptr, BK);
                wmma::load_matrix_sync(a2_frags[mi], a2_ptr, BK);
            }

            // B frags per "column" of MMA tiles in this warp tile, 3 digits
            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> 
                b0_frags[WARP_N_TILES],
                b1_frags[WARP_N_TILES],
                b2_frags[WARP_N_TILES];

            #pragma unroll
            for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                int b_row = kk;                                     // within [0, BK)
                int b_col = (warp_c_col - block_col) + nj * WMMA_N; // within [0, BN)

                const __nv_bfloat16* b0_ptr = &B0[stage][b_row][b_col];
                const __nv_bfloat16* b1_ptr = &B1[stage][b_row][b_col];
                const __nv_bfloat16* b2_ptr = &B2[stage][b_row][b_col];

                wmma::load_matrix_sync(b0_frags[nj], b0_ptr, BN);
                wmma::load_matrix_sync(b1_frags[nj], b1_ptr, BN);
                wmma::load_matrix_sync(b2_frags[nj], b2_ptr, BN);
            }

            // Now do all MMA updates reusing these fragments
            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < WARP_N_TILES; ++nj) {

                    // G0 = A0 * B0
                    wmma::mma_sync(c_frags[mi][nj],
                                   a0_frags[mi],
                                   b0_frags[nj],
                                   c_frags[mi][nj]);

                    // G1 = A0*B1 + A1*B0
                    wmma::mma_sync(g1_frags[mi][nj],
                                   a0_frags[mi],
                                   b1_frags[nj],
                                   g1_frags[mi][nj]);
                    wmma::mma_sync(g1_frags[mi][nj],
                                   a1_frags[mi],
                                   b0_frags[nj],
                                   g1_frags[mi][nj]);

                    // G2 = A0*B2 + A1*B1 + A2*B0
                    wmma::mma_sync(g2_frags[mi][nj],
                                   a0_frags[mi],
                                   b2_frags[nj],
                                   g2_frags[mi][nj]);
                    wmma::mma_sync(g2_frags[mi][nj],
                                   a1_frags[mi],
                                   b1_frags[nj],
                                   g2_frags[mi][nj]);
                    wmma::mma_sync(g2_frags[mi][nj],
                                   a2_frags[mi],
                                   b0_frags[nj],
                                   g2_frags[mi][nj]);

                    // G3 = A1*B2 + A2*B1
                    wmma::mma_sync(g3_frags[mi][nj],
                                   a1_frags[mi],
                                   b2_frags[nj],
                                   g3_frags[mi][nj]);
                    wmma::mma_sync(g3_frags[mi][nj],
                                   a2_frags[mi],
                                   b1_frags[nj],
                                   g3_frags[mi][nj]);

                    // G4 = A2 * B2
                    wmma::mma_sync(g4_frags[mi][nj],
                                   a2_frags[mi],
                                   b2_frags[nj],
                                   g4_frags[mi][nj]);
                }
            }

            // final scaling
            const float s1 = ldexpf(1.0f, -8);   // 2^-8
            const float s2 = ldexpf(1.0f, -16);  // 2^-16
            const float s3 = ldexpf(1.0f, -24);  // 2^-24
            const float s4 = ldexpf(1.0f, -32);  // 2^-32

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < WARP_N_TILES; ++nj) {
                    #pragma unroll
                    for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                        float g0 = c_frags [mi][nj].x[e];
                        float g1 = g1_frags[mi][nj].x[e];
                        float g2 = g2_frags[mi][nj].x[e];
                        float g3 = g3_frags[mi][nj].x[e];
                        float g4 = g4_frags[mi][nj].x[e];

                        c_frags[mi][nj].x[e] =
                            g0
                        + s1 * g1
                        + s2 * g2
                        + s3 * g3
                        + s4 * g4;
                    }
                }
            }
        }

        // Ensure next tile's async copies are done before we use it
        if (tile_k + 1 < num_k_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int nj = 0; nj < WARP_N_TILES; ++nj) {
            int row = warp_c_row + mi * WMMA_M;
            int col = warp_c_col + nj * WMMA_N;

            if (row + WMMA_M <= M && col + WMMA_N <= N) {
                float* c_ptr = &C[row * ldc + col];

                if(beta == 0) {
                    #pragma unroll
                    for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                        c_frags[mi][nj].x[e] =
                            alpha * c_frags[mi][nj].x[e];
                    }
                } else {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
                    wmma::load_matrix_sync(c_old, c_ptr, ldc, wmma::mem_row_major);

                    #pragma unroll
                    for (int e = 0; e < c_frags[mi][nj].num_elements; ++e) {
                        c_frags[mi][nj].x[e] =
                            alpha * c_frags[mi][nj].x[e] + beta * c_old.x[e];
                    }
                }

                wmma::store_matrix_sync(c_ptr, c_frags[mi][nj], ldc, wmma::mem_row_major);
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////

void emulate_sgemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;

    const int WM = 64;
    const int WN = 64;

    const int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);

    dim3 gemm_block(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 gemm_grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    bool fused = true;
    if(fused) {
        wmma_fp32_emulated<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        // 1. Allocate BF16 slices: A0..A2, B0..B2
        size_t szA = size_t(M) * K;
        size_t szB = size_t(K) * N;

        int lda = K;
        int ldb = N;
        int ldc = N;

        __nv_bfloat16 *A0, *A1, *A2;
        __nv_bfloat16 *B0, *B1, *B2;
        cudaMalloc(&A0, szA * sizeof(__nv_bfloat16));
        cudaMalloc(&A1, szA * sizeof(__nv_bfloat16));
        cudaMalloc(&A2, szA * sizeof(__nv_bfloat16));
        cudaMalloc(&B0, szB * sizeof(__nv_bfloat16));
        cudaMalloc(&B1, szB * sizeof(__nv_bfloat16));
        cudaMalloc(&B2, szB * sizeof(__nv_bfloat16));

        // 2. Slice A and B into 3 BF16 pieces each
        dim3 block(16, 16);
        dim3 gridA((K + block.x - 1)/block.x,
                (M + block.y - 1)/block.y);
        dim3 gridB((N + block.x - 1)/block.x,
                (K + block.y - 1)/block.y);

        slice_fp32_to_bf16x3<<<gridA, block, 0>>>(A, A0, A1, A2, M, K, lda, K);
        slice_fp32_to_bf16x3<<<gridB, block, 0>>>(B, B0, B1, B2, K, N, ldb, N);

        // 3. Compute C = beta*C + alpha * (full BF16x9 expansion)
        const float s1  = ldexpf(1.0f, -8);   // 2^-8
        const float s2  = ldexpf(1.0f, -16);  // 2^-16
        const float s3  = ldexpf(1.0f, -24);  // 2^-24
        const float s4  = ldexpf(1.0f, -32);  // 2^-32

        // Term group: A0B0
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A0, B0, C, M, N, K, alpha, beta);
        // C = beta*C + alpha*A0B0

        // Group 2: 2^-8 (A0B1 + A1B0)
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A0, B1, C, M, N, K, alpha * s1, 1.0f);
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A1, B0, C, M, N, K, alpha * s1, 1.0f);

        // Group 3: 2^-16 (A0B2 + A1B1 + A2B0)
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A0, B2, C, M, N, K, alpha * s2, 1.0f);
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A1, B1, C, M, N, K, alpha * s2, 1.0f);
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A2, B0, C, M, N, K, alpha * s2, 1.0f);

        // Group 4: 2^-24 (A1B2 + A2B1)
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A1, B2, C, M, N, K, alpha * s3, 1.0f);
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A2, B1, C, M, N, K, alpha * s3, 1.0f);

        // Group 5: 2^-32 (A2B2)
        wmma_bf16_gemm_async_kernel<BM, BM, BK, WM, WN><<<gemm_grid, gemm_block>>>(A2, B2, C, M, N, K, alpha * s4, 1.0f);

        // 4. Free slices
        cudaFree(A0); cudaFree(A1); cudaFree(A2);
        cudaFree(B0); cudaFree(B1); cudaFree(B2);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int size = 4096;
    int M = size;
    int N = size;
    int K = size;

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

    // do emulation multiplication
    emulate_sgemm(dA, dB, dC, M, N, K, 1.0, 0.0);

    // do verification
    bool verify = true;
    bool verbose = true;
    if (verify) {
        float* dC_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
        CHECK_CUDA(cudaMemset(dC_ref, 0, sizeC * sizeof(float)));

        cublas_ref_gemm<float>(handle, M, N, K, dA, dB, dC_ref);
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

        const double tol_abs = GemmTol<float>::abs;
        const double tol_rel = GemmTol<float>::rel;

        bool ok = (max_abs <= tol_abs)/* && (max_rel <= tol_rel) */;

        if (verbose) {
            std::cout << (ok ? "(OK)" : "(WARN)") << "\n";
        }
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}