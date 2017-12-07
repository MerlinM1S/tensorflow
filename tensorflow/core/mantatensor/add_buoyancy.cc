#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "fluid_grid_functor.h"
#include "add_buoyancy.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("AddBuoyancy")
    .Input("in_vel_grid: float")
    .Input("flags_grid: int32")
    .Input("density_grid: float")
    .Input("force: float")
    .Output("out_vel_grid: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/*
void fKernel (Size size, void (*f)(int, int), int border, void (*fBorder)(int) ) {
    int idx = 0;
    int idxi = 0;

    for (int x = 0; x < size.width; x++) {
        bool xInside = x >= border && x < size.width - border;
        for (int y = 0; y < size.height; y++) {
            bool yInside = y >= border && y < size.height - border;
            for (int z = 0; z < size.depth; z++) {
                bool zInside = z > 0 && z < size.depth - 1;
                for(int i = 0; i < size.dim; i++) {
                    if(xInside && yInside && zInside) {
                        f(idx, idxi);
                    } else {
                        fBorder(idx);
                    }
                    idxi++;
                }
                idx++;
            }
        }
    }
}

*/

/*
  void doBuo(const CPUDevice& d, FluidGrid* fluidGrid, const float* force, float* out_vel) {
      int idx = 0;
      int idxi = 0;

      for (int x = 0; x < fluidGrid->getWidth(); x++) {
          bool xInside = x > 0 && x < fluidGrid->getWidth() - 1;

          for (int y = 0; y < fluidGrid->getHeight(); y++) {
              bool yInside = y > 0 && y < fluidGrid->getHeight() - 1;

              for (int z = 0; z < fluidGrid->getDepth(); z++) {
                  bool zInside = z > 0 && z < fluidGrid->getDepth() - 1;

                  bool isIdxFluid = fluidGrid->isFluid(idx);
                  //bool isIdxFluid = isFluid(flag_grid_flat.data(), idx);
                  for(int i = 0; i < fluidGrid->getDimension(); i++) {
                      float value = fluidGrid->vel[idxi];

                      int idxNeighbour = idx + fluidGrid->getDimOffset(i);
                      bool isNeighbourFluid = fluidGrid->isFluid(idxNeighbour);
                      if(xInside && yInside && zInside && isIdxFluid && isNeighbourFluid) {
                          value += (0.5f*force[i]) * (fluidGrid->den[idx] + fluidGrid->den[idxNeighbour]);
                      }

                      out_vel[idxi] = value;
                      idxi++;
                  }


                  idx++;
              }
          }
      }
  }
*/

/*
class FluidGridBuoyancy : public FluidGrid {
    float* out_vel;
    const float* force;

    protected:
        bool kernelCondition(int idx) {
            return isFluid(idx);
        }
        void kernelFunction(int idx, int idxi, int i) {
            float value = vel[idxi];

            int idxNeighbour = idx + getDimOffset(i);
            if(isFluid(idxNeighbour)) {
                value += (0.5f*force[i]) * (den[idx] + den[idxNeighbour]);
            }

            out_vel[idxi] = value;
        }
        void kernelIdentity(int idxi) {
            out_vel[idxi] = vel[idxi];
        }

    public:
        void setParameter(float* out_vel, const float* force) {
            this->out_vel = out_vel;
            this->force = force;
        }
};

*/

// CPU specialization of actual computation.
template <>
struct AddBuoyancy<CPUDevice> {
  void operator()(const CPUDevice& d, FluidGrid* fluidGrid, const float* force, float* out_vel) {
      FluidGridFunctor fluidGridFunctor(fluidGrid);
      FluidGridFunctor* pFluidGridFunctor = &fluidGridFunctor;

      int idx = 0;
      int idxi = 0;

      for (int b = 0; b < pFluidGridFunctor->getBatches(); b++) {
          for (int x = 0; x < pFluidGridFunctor->getWidth(); x++) {
              bool xInside = x > 0 && x < pFluidGridFunctor->getWidth() - 1;
              for (int y = 0; y < pFluidGridFunctor->getHeight(); y++) {
                  bool yInside = y > 0 && y < pFluidGridFunctor->getHeight() - 1;
                  for (int z = 0; z < pFluidGridFunctor->getDepth(); z++) {
                      bool zInside = z > 0 && z < pFluidGridFunctor->getDepth() - 1;
                      bool isIdxFluid = pFluidGridFunctor->isFluid(idx);
                      for(int i = 0; i < pFluidGridFunctor->getDim(); i++) {
                          float value = pFluidGridFunctor->getVel()[idxi];

                          int idxNeighbour = idx +  pFluidGridFunctor->getDimOffset(i);
                          bool isNeighbourFluid = pFluidGridFunctor->isFluid(idxNeighbour);
                          if(xInside && yInside && zInside && isIdxFluid && isNeighbourFluid) {
                              value += (0.5f*force[i]) * (pFluidGridFunctor->getDen()[idx] + pFluidGridFunctor->getDen()[idxNeighbour]);
                          }

                          out_vel[idxi] = value;
                          idxi++;
                      }
                      idx++;
                  }
              }
          }
      }
  }
};

#include <iostream>
#include <initializer_list>


bool isSameValue(std::initializer_list<long> list)
{

    for (int i = 1; i < list.size(); i++)
        if(list.begin()[0] != list.begin()[i])
            return false;
    return true;
}

template <typename Device>
class AddBuoyancyOp : public OpKernel {
 public:
  explicit AddBuoyancyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor


    const Tensor& input_vel_tensor = context->input(0);
    TensorShape vel_shape = input_vel_tensor.shape();
    auto input_vel_flat = input_vel_tensor.flat<float>();

    const Tensor& flag_grid_tensor = context->input(1);
    TensorShape flag_grid_shape = flag_grid_tensor.shape();
    auto flag_grid_flat = flag_grid_tensor.flat<int>();

    const Tensor& density_tensor = context->input(2);
    TensorShape density_shape = density_tensor.shape();
    auto density_flat = density_tensor.flat<float>();

    const Tensor& force_tensor = context->input(3);
    TensorShape force_shape = force_tensor.shape();

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

    FluidGrid fluidGrid;
    fluidGrid.vel = input_vel_flat.data();
    fluidGrid.den =  density_flat.data();
    fluidGrid.flags = flag_grid_flat.data();

    fluidGrid.batches   = vel_shape.dim_size(0);
    fluidGrid.width     = vel_shape.dim_size(1);
    fluidGrid.height    = vel_shape.dim_size(2);
    fluidGrid.depth     = vel_shape.dim_size(3);
    fluidGrid.dim       = force_shape.dim_size(0);


    // Create an output tensor
    Tensor* output_vel_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, vel_shape, &output_vel_tensor));


    AddBuoyancy<Device>()(
        context->eigen_device<Device>(),
        &fluidGrid,
        force_tensor.flat<float>().data(),
        output_vel_tensor->flat<float>().data());

    /*

    FluidGrid fluidGrid(flag_grid_flat.data(), density_flat.data(), input_vel_flat.data());
    fluidGrid.setSize(vel_shape.dim_size(0), vel_shape.dim_size(1), vel_shape.dim_size(2));
    fluidGrid.setDimension(force_shape.dim_size(0));

    AddBuoyancy<Device>()(
        context->eigen_device<Device>(),
        &fluidGrid,
        force_tensor.flat<float>().data(),
        output_vel_tensor->flat<float>().data());

        */

  }
};


// Register the CPU kernels.
                                        \
REGISTER_KERNEL_BUILDER(                                       \
      Name("AddBuoyancy").Device(DEVICE_CPU), AddBuoyancyOp<CPUDevice>);






#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
        Name("AddBuoyancy").Device(DEVICE_GPU), AddBuoyancyOp<GPUDevice>);

#endif






