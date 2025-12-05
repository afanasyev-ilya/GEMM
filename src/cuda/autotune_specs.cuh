#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////
// Specs
/////////////////////////////////////////////////////////////////////////////////////////////////

inline constexpr int quite_autotune = false;

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
        constexpr bool warp_valid  = (WM / TM) * (WN / TN) == 32;
        constexpr bool warp_fits_block = (BM >= WM) && (BN >= WN);
        return threads_fit && warp_valid && warp_fits_block;
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
