#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t status_ = (call);                                      \
    if (status_ != cudaSuccess)                                        \
    {                                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(status_)       \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

#define CHECK_CUBLAS(call)                                             \
  do                                                                   \
  {                                                                    \
    cublasStatus_t status_ = (call);                                   \
    if (status_ != CUBLAS_STATUS_SUCCESS)                              \
    {                                                                  \
      std::cerr << "cuBLAS error: " << status_                         \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

#define CEIL_DIV(a, b) ( ((a) - 1) / (b) + 1 )

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}
