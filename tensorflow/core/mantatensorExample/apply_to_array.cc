#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "fluid_grid_functor.h"
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
    FluidGridFunctor fluidGridFunctor(fluidGrid);

    for (int i = 0; i < size; i++) {
        if(mask[i] && !fluidGridFunctor.isObstacle(i))
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
            auto mask_flat = mask_tensor.flat<bool>();

            const Tensor& value_tensor = context->input(3);
            TensorShape value_tensor_shape = value_tensor.shape();


            OP_REQUIRES(context, in_tensor.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument("Too many elements in tensor"));

            // Create an output tensor
            Tensor* out_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, in_tensor.shape(), &out_tensor));


            const FluidGrid fluidGrid = {
                in_shape.dim_size(0),              // batches
                in_shape.dim_size(1),              // width
                in_shape.dim_size(2),              // height
                in_shape.dim_size(3),              // depth
                in_shape.dim_size(4),              // dim - currently wrong for in_shape.dims() < 5

                nullptr,
                nullptr,
                flag_grid_flat.data()
            };

            const float value = value_tensor.scalar<float>().data()[0];


            // Do the computation.
            ApplyToArray<Device>()(
                context->eigen_device<Device>(),
                &fluidGrid,
                in_tensor.flat<float>().data(),
                mask_flat.data(),
                value,
                out_tensor->flat<float>().data(),
                static_cast<int>(in_tensor.NumElements()));

        }


};


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("ApplyToArray").Device(DEVICE_CPU), ApplyToArrayOp<CPUDevice>);






