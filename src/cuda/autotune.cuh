#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <string_view>
#include <type_traits>
#include <utility>
#include <stdexcept>

#include "bench_common.cuh"
#include "macros.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time lists + loops
/////////////////////////////////////////////////////////////////////////////////////////////////

template<int... Is>
struct ValueList {};

template<int... Is, typename Func>
void static_for(ValueList<Is...>, Func&& f) {
    (f.template operator()<Is>(), ...);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Configs
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Config {
    int BM, BN, BK, TM, TN;
    double flops; // actually TFLOP/s from run_gemm_bench
};

inline std::ostream& operator<<(std::ostream& os, const Config& c) {
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

struct ConfigWarpTiling {
    int BM, BN, BK, WM, WN, TM, TN;
    double flops; // TFLOP/s
};

inline std::ostream& operator<<(std::ostream& os, const ConfigWarpTiling& c) {
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

struct ConfigWMMA {
    int BM, BN, BK, WM, WN;
    double flops; // TFLOP/s
};

inline std::ostream& operator<<(std::ostream& os, const ConfigWMMA& c) {
    os << "Config{"
       << "BM=" << c.BM
       << ", BN=" << c.BN
       << ", BK=" << c.BK
       << ", WM=" << c.WM
       << ", WN=" << c.WN
       << ", flops=" << c.flops
       << "}";
    return os;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// static_for_product (cartesian product)
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Func, int... Prefix>
struct StaticProduct {
    template<typename List, typename... Rest>
    static void run(Func&& f) {
        static_for(List{}, [&]<int I>() {
            if constexpr (sizeof...(Rest) == 0) {
                f.template operator()<Prefix..., I>();
            } else {
                StaticProduct<Func, Prefix..., I>::template run<Rest...>(std::forward<Func>(f));
            }
        });
    }
};

template<typename... Lists, typename Func>
void static_for_product(Func&& f) {
    StaticProduct<Func>::template run<Lists...>(std::forward<Func>(f));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Generic bench helper
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename FloatType, typename Spec, int... P>
double bench_config_generic(cublasHandle_t handle,
                            const FloatType* dA, const FloatType* dB, float* dC,
                            int M, int N, int K, int iters, bool verify)
{
    dim3 block = Spec::template block<P...>();
    int threads = int(block.x) * int(block.y) * int(block.z);
    if (threads > 1024) return 0.0;

    dim3 grid = Spec::template grid<P...>(M, N);

    bool verbose = false;

    auto launch = make_launcher<Spec::template Kernel<P...>>(grid, block);

    return run_gemm_bench<FloatType>(
        handle, M, N, K,
        dA, dB, dC,
        iters,
        launch,
        Spec::bench_name(),
        1.0f, 0.0f,
        verify, verbose
    );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Generic autotune engine
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename FloatType, typename Spec>
typename Spec::Config autotune_generic(cublasHandle_t handle,
                                       const FloatType* dA, const FloatType* dB, float* dC,
                                       int M, int N, int K, int iters, bool verify)
{
    using Cfg = typename Spec::Config;
    Cfg best = Spec::empty();

    std::cout << std::setprecision(3);

    Spec::for_each([&]<int... P>() {
        if constexpr (Spec::template valid<P...>()) {
            double g = bench_config_generic<FloatType, Spec, P...>(
                handle, dA, dB, dC, M, N, K, iters, verify
            );
            Spec::update_best(best, g, P...);
        }
    });

    return best;
}

template<typename Spec>
void run_autotuned_generic(const typename Spec::Config cfg,
                           cublasHandle_t handle,
                           const float* dA, const float* dB, float* dC,
                           int M, int N, int K, int iters,
                           std::string_view name, bool verify)
{
    bool launched = false;

    Spec::for_each([&]<int... P>() {
        if constexpr (Spec::template valid<P...>()) {
            if (Spec::match(cfg, P...)) {
                dim3 block = Spec::template block<P...>();
                dim3 grid  = Spec::template grid<P...>(M, N);

                auto launch = make_launcher<Spec::template Kernel<P...>>(grid, block);

                run_gemm_bench<float>(
                    handle, M, N, K,
                    dA, dB, dC,
                    iters,
                    launch,
                    name,
                    1.0f, 0.0f,
                    verify, true
                );

                launched = true;
            }
        }
    });

    if (!launched) {
        throw std::runtime_error("Autotuned cfg is not in the compiled search space");
    }
}

