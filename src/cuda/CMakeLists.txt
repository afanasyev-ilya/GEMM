add_library(gemm_cuda_common INTERFACE)

# Force these .cu files to be treated as CUDA
set_source_files_properties(
  sgemm.cu
  bf16_gemm.cu
  PROPERTIES
    LANGUAGE CUDA
)

target_link_libraries(gemm_cuda_common
  INTERFACE
    gemm_options
    CUDA::cudart
)

target_include_directories(gemm_cuda_common
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gemm_cuda_common
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --use_fast_math
      --expt-relaxed-constexpr
      --expt-extended-lambda
    >
)

add_executable(sgemm sgemm.cu)
target_link_libraries(sgemm
  PRIVATE
    gemm_cuda_common
    CUDA::cublas
)

add_executable(bf16_gemm bf16_gemm.cu)
target_link_libraries(bf16_gemm
  PRIVATE
    gemm_cuda_common
    CUDA::cublas
)
