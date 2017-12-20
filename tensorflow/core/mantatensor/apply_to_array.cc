#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("ApplyToArray")
    .Input("in_array: float")
    .Input("flags_grid: int32")
    .Input("mask: bool")
    .Input("value: float")
    .Output("out_array: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <typename Device>
struct ApplyToArray {
  void operator()(const CPUDevice& d, const FluidGrid* fluidGrid, const float* in_array, const bool* mask, const float value, float* out_array, int size);
};

// CPU specialization of actual computation.
template <>
void ApplyToArray<CPUDevice>::operator()(const CPUDevice& d, const FluidGrid* fluidGrid, const float* in_array, const bool* mask, const float value, float* out_array, int size) {
    for (int i = 0; i < size; i++) {
        if(mask[i] && !fluidGrid.isObstacle(i))
            out_array[i] = value;
        else
            out_array[i] = in_array[i];
    }
}



template <typename Device>
class ApplyToArrayOp : public OpKernel {
    public:
        explicit ApplyToArrayOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& in_tensor = context->input(0);
            TensorShape in_shape = in_tensor.shape();

            const Tensor& flag_grid_tensor = context->input(1);
            TensorShape flag_grid_shape = flag_grid_tensor.shape();
            auto flag_grid_flat = flag_grid_tensor.flat<int>();

            const Tensor& mask_tensor = context->input(2);
            TensorShape mask_shape = mask_tensor.shape();

            const Tensor& value_tensor = context->input(1);
            TensorShape value_tensor_shape = apply_tensor.shape();



            OP_REQUIRES(context, (in_shape == apply_tensor_shape) &&  (in_shape == mask_shape),
                        errors::InvalidArgument("ApplyArrayOp expects all parameters to have the same shape"));
            OP_REQUIRES(context, in_tensor.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument("Too many elements in tensor"));

            // Create an output tensor
            Tensor* out_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, in_tensor.shape(), &out_tensor));


            // Do the computation.
            ApplyToArray<Device>()(
                context->eigen_device<Device>(),
                in_tensor.flat<float>().data(),
                apply_tensor.flat<float>().data(),
                mask_tensor.flat<bool>().data(),
                out_tensor->flat<float>().data(),
                static_cast<int>(in_tensor.NumElements()));
        }
};


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("ApplyToArray").Device(DEVICE_CPU), ApplyToArrayOp<CPUDevice>);






