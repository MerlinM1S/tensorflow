#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("TestEmpty")
    .Input("in_vel_grid: float")
    .Output("out_vel_grid: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;




template <typename Device>
class TestEmptyOp : public OpKernel {
 public:
  explicit TestEmptyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      // Create an output tensor
      Tensor* output_vel_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(), &output_vel_tensor));
  }
};


// Register the CPU kernels.


REGISTER_KERNEL_BUILDER(Name("TestEmpty").Device(DEVICE_CPU), TestEmptyOp<CPUDevice>);






