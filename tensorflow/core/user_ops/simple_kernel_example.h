// kernel_example.h
#ifndef SIMPLE_KERNEL_EXAMPLE_H_
#define SIMPLE_KERNEL_EXAMPLE_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;



    /*
template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};
*/


template <typename Device>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const float* in, float* out);
};



#if GOOGLE_CUDA

/*

using GPUDevice = Eigen::GpuDevice;

// Partially specialize functor for GpuDevice.
struct ExampleFunctorGPU {
  void operator()(const Eigen::GpuDevice& d, int size, const int* in, int* out);
};

*/

#endif




#endif // SIMPLE_KERNEL_EXAMPLE_H_