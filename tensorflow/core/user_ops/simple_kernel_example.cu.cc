// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "simple_kernel_example.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


using GPUDevice = Eigen::GpuDevice;

/*

// Partially specialize functor for GpuDevice.
template <typename GPUDevice, typename T>
struct ExampleFunctor {
  void operator()(const GPUDevice& d, int size, const T* in, T* out);
};

using GPUDevice = Eigen::GpuDevice;

*/

// Define the CUDA kernel.
__global__ void ExampleCudaKernel(const int size, const float* in, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <>
void ExampleFunctor<GPUDevice>::operator()(
    const GPUDevice& d, int size, const float* in, float* out) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}


#endif  // GOOGLE_CUDA
