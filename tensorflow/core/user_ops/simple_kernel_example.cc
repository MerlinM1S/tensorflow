#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "simple_kernel_example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("SimpleKernelExample")
    .Input("to_zero: float")
    .Output("zeroed: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });



using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <>
struct ExampleFunctor<CPUDevice> {
  void operator()(const CPUDevice& d, int size, const float* in, float* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};



// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device>
class SimpleKernelExampleOP : public OpKernel {
 public:
  explicit SimpleKernelExampleOP(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<float>().data(),
        output_tensor->flat<float>().data());
  }
};


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(
      Name("SimpleKernelExample").Device(DEVICE_CPU), SimpleKernelExampleOP<CPUDevice>);




#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
        Name("SimpleKernelExample").Device(DEVICE_GPU), SimpleKernelExampleOP<GPUDevice>);

#endif
