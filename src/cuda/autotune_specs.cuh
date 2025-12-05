#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////
// Specs
/////////////////////////////////////////////////////////////////////////////////////////////////

inline constexpr int quite_autotune = false;

// forward declaration is needed
template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_vectorize_smem(const float * __restrict__ A,
                                     const float * __restrict__ B, 
                                     float * __restrict__ C, 
                                     int M, int N, int K,
                                     float alpha, float beta);

struct VecSmemSpec {
    using Config = ::Config;

    static Config empty() { return {0,0,0,0,0,0.0}; }

    static constexpr std::string_view bench_name() {
        return "autotune";
    }

    template<int BM, int BN, int BK, int TM, int TN>
    static constexpr auto Kernel = sgemm_vectorize_smem<BM, BN, BK, TM, TN>;

    template<typename Func>
    static void for_each(Func&& f) {
        static_for_product<BMs, BNs, BKs, TMs, TNs>(std::forward<Func>(f));
    }

    template<int BM, int BN, int BK, int TM, int TN>
    static constexpr bool valid() {
        constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;
        return threads_fit;
    }

    template<int BM, int BN, int BK, int TM, int TN>
    static dim3 block() {
        static_assert(BN % TN == 0, "BN % TN != 0");
        static_assert(BM % TM == 0, "BM % TM != 0");
        return dim3(BN / TN, BM / TM, 1);
    }

    template<int BM, int BN, int BK, int TM, int TN>
    static dim3 grid(int M, int N) {
        dim3 b = block<BM, BN, BK, TM, TN>();
        return dim3(
            CEIL_DIV(N, int(b.x) * TN),
            CEIL_DIV(M, int(b.y) * TM),
            1
        );
    }

    static bool match(const Config& c, int BM, int BN, int BK, int TM, int TN) {
        return c.BM == BM && c.BN == BN && c.BK == BK &&
            c.TM == TM && c.TN == TN;
    }

    static void update_best(Config& best, double g,
                            int BM, int BN, int BK, int TM, int TN)
    {
        if (!quite_autotune) {
            std::cout << "Tested: " << BM << " " << BN << " " << BK
                      << " | " << TM << " " << TN
                      << " -> " << g << " TFLOP/s ";
        }

        if (g > best.flops) {
            if (!quite_autotune) std::cout << "(new best found!)\n";
            best = {BM, BN, BK, TM, TN, g};
        } else {
            if (!quite_autotune) std::cout << "\n";
        }
    }
};

// forward declaration is needed
template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void
sgemm_warp_tiling(const float * __restrict__ A,
                                  const float * __restrict__ B,
                                  float * __restrict__  C,
                                  int M, int N, int K,
                                  float alpha, float beta);

struct WarpTilingSpec {
    using Config = ::ConfigWarpTiling;

    static Config empty() { return {0,0,0,0,0,0,0,0.0}; }

    static constexpr std::string_view bench_name() {
        return "autotune warp tiling";
    }

    template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    static constexpr auto Kernel = sgemm_warp_tiling<BM, BN, BK, WM, WN, TM, TN>;

    template<typename Func>
    static void for_each(Func&& f) {
        static_for_product<BMs, BNs, BKs, WMs, WNs, TMs, TNs>(std::forward<Func>(f));
    }

    template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    static constexpr bool valid() {
        constexpr bool threads_fit = ((BM * BN) / (TM * TN)) <= 256;
        //constexpr bool warp_valid  = (WM / TM) * (WN / TN) == 32; // TODO fix ME
        constexpr bool warp_fits_block = (BM >= WM) && (BN >= WN);
        return threads_fit && warp_fits_block;
    }

    template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    static dim3 block() {
        static_assert(BN % TN == 0, "BN % TN != 0");
        static_assert(BM % TM == 0, "BM % TM != 0");
        return dim3(BN / TN, BM / TM, 1);
    }

    template<int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    static dim3 grid(int M, int N) {
        dim3 b = block<BM, BN, BK, WM, WN, TM, TN>();
        return dim3(
            CEIL_DIV(N, int(b.x) * TN),
            CEIL_DIV(M, int(b.y) * TM),
            1
        );
    }

    static bool match(const Config& c, int BM, int BN, int BK, int WM, int WN, int TM, int TN) {
        return c.BM == BM && c.BN == BN && c.BK == BK &&
               c.WM == WM && c.WN == WN && c.TM == TM && c.TN == TN;
    }

    static void update_best(Config& best, double g,
                            int BM, int BN, int BK, int WM, int WN, int TM, int TN)
    {
        if (!quite_autotune) {
            std::cout << "Tested: " << BM << " " << BN << " " << BK
                      << " | " << WM << " " << WN
                      << " | " << TM << " " << TN
                      << " -> " << g << " TFLOP/s ";
        }

        if (g > best.flops) {
            if (!quite_autotune) std::cout << "(new best found!)\n";
            best = {BM, BN, BK, WM, WN, TM, TN, g};
        } else {
            if (!quite_autotune) std::cout << "\n";
        }
    }
};

// forward declaration is needed
template<int BM, int BN, int BK, int WM, int WN>
__global__ void
wmma_bf16_gemm_vector_loads_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   float      * __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta);

struct WMMASpec {
    using Config = ::ConfigWMMA;

    static Config empty() { return {0,0,0,0,0,0.0}; }

    static constexpr std::string_view bench_name() {
        return "autotune WMMA";
    }

    template<int BM, int BN, int BK, int WM, int WN>
    static constexpr auto Kernel = wmma_bf16_gemm_vector_loads_kernel<BM, BN, BK, WM, WN>;

    template<typename Func>
    static void for_each(Func&& f) {
        static_for_product<BMs, BNs, BKs, WMs, WNs>(std::forward<Func>(f));
    }

    template<int BM, int BN, int BK, int WM, int WN>
    static constexpr bool valid() {
        return true;
    }

    template<int BM, int BN, int BK, int WM, int WN>
    static dim3 block() {
        return dim3(1, 1, 1);
    }

    template<int BM, int BN, int BK, int WM, int WN>
    static dim3 grid(int M, int N) {
        dim3 b = block<BM, BN, BK, WM, WN>();
        return dim3(1, 1);
    }

    static bool match(const Config& c, int BM, int BN, int BK, int WM, int WN) {
        return c.BM == BM && c.BN == BN && c.BK == BK &&
               c.WM == WM && c.WN == WN;
    }

    static void update_best(Config& best, double g,
                            int BM, int BN, int BK, int WM, int WN)
    {
        if (!quite_autotune) {
            std::cout << "Tested: " << BM << " " << BN << " " << BK
                      << " | " << WM << " " << WN
                      << " -> " << g << " TFLOP/s ";
        }

        if (g > best.flops) {
            if (!quite_autotune) std::cout << "(new best found!)\n";
            best = {BM, BN, BK, WM, WN, g};
        } else {
            if (!quite_autotune) std::cout << "\n";
        }
    }
};

