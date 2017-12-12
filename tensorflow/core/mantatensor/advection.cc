 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fluid_grid.h"


using namespace tensorflow;

template <typename Device>
struct Advection {
  void operator()(const Device& d, const FluidGrid* fluidGrid, const float dt, const float* in_grid, float* out_grid);
};







#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "fluid_grid_functor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("Advection")
    .Input("in_grid: float")
    .Input("flags_grid: int32")
    .Input("vel_grid: float")
    .Input("dt: float")
    .Output("out_grid: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;




// CPU specialization of actual computation.
template <>
struct Advection<CPUDevice> {
  void operator()(const CPUDevice& d, const FluidGrid* fluidGrid, const float dt, const float* in_grid, float* out_grid) {
      FluidGridFunctor fluidGridFunctor(fluidGrid);
      FluidGridFunctor* pFluidGridFunctor = &fluidGridFunctor;
      // traceback position
      //Vec3 pos = Vec3(i+0.5f,j+0.5f,k+0.5f) - vel.getCentered(i,j,k) * dt;
      //dst(i,j,k) = src.getInterpolatedHi(pos, orderSpace);
      // 		case 1:  return interpol     <T>(mData, mSize, mStrideZ, pos);



      for (int b = 0; b < pFluidGridFunctor->getBatches(); b++) {
          int i_b = b * pFluidGridFunctor->getWidth() * pFluidGridFunctor->getHeight() * pFluidGridFunctor->getDepth();
          for (int x = 0; x < pFluidGridFunctor->getWidth(); x++) {
              int i_bx = i_b + x*pFluidGridFunctor->getHeight() * pFluidGridFunctor->getDepth();
              bool xInside = x >= 1 && x < pFluidGridFunctor->getWidth() - 1;
              for (int y = 0; y < pFluidGridFunctor->getHeight(); y++) {
                  int i_bxy = i_bx + y * pFluidGridFunctor->getDepth();
                  bool yInside = y >= 1 && y < pFluidGridFunctor->getHeight() - 1;
                  for (int z = 0; z < pFluidGridFunctor->getDepth(); z++) {
                      int i_bxyz = i_bxy + z;
                      bool zInside = z >= 1 && z < pFluidGridFunctor->getDepth() - 1;

                      if(xInside && yInside && zInside) {
                          Vec3 pos = Vec3(x+0.5f,y+0.5f,z+0.5f) - pFluidGridFunctor->getCenteredVel(i_bxyz) * dt;
                          out_grid[i_bxyz] = pFluidGridFunctor->interpolate(in_grid, pos, b);
                      } else {
                            out_grid[i_bxyz] = 0;
                      }
                  }
              }
          }
      }

  }
};


template <typename Device>
class AdvectionOp : public OpKernel {
 public:
  explicit AdvectionOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor


    const Tensor& input_grid_tensor = context->input(0);
    TensorShape grid_shape = input_grid_tensor.shape();
    auto input_grid_flat = input_grid_tensor.flat<float>();


    const Tensor& flag_grid_tensor = context->input(1);
    TensorShape flag_grid_shape = flag_grid_tensor.shape();
    auto flag_grid_flat = flag_grid_tensor.flat<int>();

    const Tensor& input_vel_tensor = context->input(2);
    TensorShape vel_shape = input_vel_tensor.shape();
    auto input_vel_flat = input_vel_tensor.flat<float>();

    const Tensor& dt_tensor = context->input(3);
    TensorShape dt_shape = dt_tensor.shape();

    /*

    OP_REQUIRES(context, vel_shape.dims() == 5,
                 errors::InvalidArgument("AddBuoyancy expects as first parameter a 5-D float velocity array: batches, width, height, depth, dimension"));
    OP_REQUIRES(context, flag_grid_shape.dims() == 4,
                 errors::InvalidArgument("AddBuoyancy expects as second parameter a 4-D int flags array: batches, width, height, depth"));
    OP_REQUIRES(context, density_shape.dims() == 4,
                 errors::InvalidArgument("AddBuoyancy expects as third parameter a 4-D float density array: batches, width, height, depth"));
    OP_REQUIRES(context, force_shape.dims() == 1,
                 errors::InvalidArgument("AddBuoyancy expects as fourth parameter a 1-D float force array: dimension"));


    OP_REQUIRES(context, isSameValue({vel_shape.dim_size(0), flag_grid_shape.dim_size(0), density_shape.dim_size(0)}),
                 errors::InvalidArgument("AddBuoyancy expects that the batch size is equal for all inputs"));
    OP_REQUIRES(context, isSameValue({vel_shape.dim_size(1), flag_grid_shape.dim_size(1), density_shape.dim_size(1)}),
                 errors::InvalidArgument("AddBuoyancy expects that the width size is equal for all inputs"));
    OP_REQUIRES(context, isSameValue({vel_shape.dim_size(2), flag_grid_shape.dim_size(2), density_shape.dim_size(2)}),
                 errors::InvalidArgument("AddBuoyancy expects that the height size is equal for all inputs"));
    OP_REQUIRES(context, isSameValue({vel_shape.dim_size(3), flag_grid_shape.dim_size(3), density_shape.dim_size(3)}),
                 errors::InvalidArgument("AddBuoyancy expects that the depth size is equal for all inputs"));
    OP_REQUIRES(context, isSameValue({vel_shape.dim_size(4), force_shape.dim_size(0)}),
                 errors::InvalidArgument("AddBuoyancy expects that the dimension size is equal for all inputs"));

                 */


    const FluidGrid fluidGrid = {
        vel_shape.dim_size(0),              // batches
        vel_shape.dim_size(1),              // width
        vel_shape.dim_size(2),              // height
        vel_shape.dim_size(3),              // depth
        vel_shape.dim_size(4),              // dim

        input_vel_flat.data(),
        nullptr,
        flag_grid_flat.data()
    };


    // Create an output tensor
    Tensor* output_grid_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grid_shape, &output_grid_tensor));


    Advection<Device>()(
        context->eigen_device<Device>(),
        &fluidGrid,
        dt_tensor.flat<float>().data()[0],
        input_grid_flat.data(),
        output_grid_tensor->flat<float>().data());

  }
};


// Register the CPU kernels.


                                        \
REGISTER_KERNEL_BUILDER(                                       \
      Name("Advection").Device(DEVICE_CPU), AdvectionOp<CPUDevice>);


