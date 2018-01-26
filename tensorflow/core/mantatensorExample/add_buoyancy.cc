#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "fluid_grid_functor.h"
#include "add_buoyancy.h"
#include "kernel_base.h"
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


class KernelBuoyancy : public KernelBase {
    private:
        float* out_vel;
        const float* force;

    protected:
        inline void kernelFunction(int i_bxyz) {
            if(!fluidGridFunctor.isFluid(i_bxyz)) {
                kernelIdentity(i_bxyz);
                return;
            }

            for(int d = 0; d < fluidGridFunctor.getDim(); d++) {
                int i_bxyzd = get_i_bxyzd(i_bxyz, d);
                float value = fluidGridFunctor.getVelGrid()[i_bxyzd];

                int idxNeighbour = i_bxyz - fluidGridFunctor.gridData1D.getOffset(d);
                if(fluidGridFunctor.isFluid(idxNeighbour)) {
                    value += (0.5f*force[d]) * (fluidGridFunctor.getDenGrid()[i_bxyz] + fluidGridFunctor.getDenGrid()[idxNeighbour]);
                }

                out_vel[i_bxyzd] = value;
            }
        }

        inline void kernelIdentityDim(int i_bxyzd) {
            out_vel[i_bxyzd] = fluidGridFunctor.getVelGrid()[i_bxyzd];
        }

    public:
        KernelBuoyancy(const FluidGridFunctor& fluidGridFunctor, const float* force,  float* out_vel) : KernelBase(fluidGridFunctor, 1) {
            this->force = force;
            this->out_vel = out_vel;
        }
};

inline void calcWithInit(const FluidGridFunctor& fluidGridFunctor, const float* force, float* out_vel) {
    const int size = fluidGridFunctor.getBatches() * fluidGridFunctor.getWidth() * fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth() * fluidGridFunctor.getDim();
    for (int i = 0; i < size; i++) {
        out_vel[i] = fluidGridFunctor.getVelGrid()[i];
    }


    const int border = 1;
    for (int b = 0; b < fluidGridFunctor.getBatches(); b++) {
        int i_b = b * fluidGridFunctor.getWidth() * fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();

        for (int x = border; x < fluidGridFunctor.getWidth() - border; x++) {
            int i_bx = i_b + x*fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();
            for (int y = border; y < fluidGridFunctor.getHeight() - border; y++) {
                int i_bxy = i_bx + y * fluidGridFunctor.getDepth();
                for (int z = border; z < fluidGridFunctor.getDepth() - border; z++) {
                    int i_bxyz = i_bxy + z;

                    for(int d = 0; d < fluidGridFunctor.getDim(); d++) {
                        int i_bxyzd = i_bxyz * fluidGridFunctor.getDim() + d;

                        int idxNeighbour = i_bxyz - fluidGridFunctor.gridData1D.getOffset(d);
                        if(fluidGridFunctor.isFluid(i_bxyz) && fluidGridFunctor.isFluid(idxNeighbour)) {
                            out_vel[i_bxyzd] += 0.5f*force[d] * (fluidGridFunctor.getDenGrid()[i_bxyz] + fluidGridFunctor.getDenGrid()[idxNeighbour]);
                        }
                    }
                }
            }
        }
    }
}

inline void calcWithoutInit(const FluidGridFunctor& fluidGridFunctor, const float* force, float* out_vel) {
    const int border = 1;
    for (int b = 0; b < fluidGridFunctor.getBatches(); b++) {
        int i_b = b * fluidGridFunctor.getWidth() * fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();

        for (int x = 0; x < fluidGridFunctor.getWidth(); x++) {
            int i_bx = i_b + x*fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();
            bool xInside = x >= border && x < fluidGridFunctor.getWidth() - border;
            for (int y = 0; y < fluidGridFunctor.getHeight(); y++) {
                int i_bxy = i_bx + y * fluidGridFunctor.getDepth();
                bool yInside = y >= border && y < fluidGridFunctor.getHeight() - border;
                for (int z = 0; z < fluidGridFunctor.getDepth(); z++) {
                    int i_bxyz = i_bxy + z;
                    bool zInside = z >= border && z < fluidGridFunctor.getDepth() - border;

                    for(int d = 0; d < fluidGridFunctor.getDim(); d++) {
                        int i_bxyzd = i_bxyz * fluidGridFunctor.getDim() + d;

                        int idxNeighbour = i_bxyz - fluidGridFunctor.gridData1D.getOffset(d);
                        if(xInside && yInside && zInside && fluidGridFunctor.isFluid(i_bxyz) && fluidGridFunctor.isFluid(idxNeighbour)) {
                            out_vel[i_bxyzd] = fluidGridFunctor.getVelGrid()[i_bxyzd] + 0.5f*force[d] * (fluidGridFunctor.getDenGrid()[i_bxyz] + fluidGridFunctor.getDenGrid()[idxNeighbour]);
                        } else {
                            out_vel[i_bxyzd] = fluidGridFunctor.getVelGrid()[i_bxyzd];
                        }
                    }
                }
            }
        }
    }
}

// CPU specialization of actual computation.
template <>
void AddBuoyancy<CPUDevice>::operator()(const CPUDevice& d, const FluidGrid* fluidGrid, const float* force, float* out_vel) {
    FluidGridFunctor fluidGridFunctor(fluidGrid);

    //KernelBuoyancy(fluidGridFunctor, force, out_vel).run();
    calcWithInit(fluidGridFunctor, force, out_vel);
    //calcWithoutInit(fluidGridFunctor, force, out_vel);
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
    OP_REQUIRES(context, flag_grid_shape.dims() == 5,
                 errors::InvalidArgument("AddBuoyancy expects as second parameter a 5-D int flags array: batches, width, height, depth, 1"));
    OP_REQUIRES(context, density_shape.dims() == 5,
                 errors::InvalidArgument("AddBuoyancy expects as third parameter a 5-D float density array: batches, width, height, depth, 1"));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(force_shape),
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
                 errors::InvalidArgument("AddBuoyancy expects that the dimension size is equal for the velocity and force input"));
    OP_REQUIRES(context, isSameValue({flag_grid_shape.dim_size(4), density_shape.dim_size(0), 1L}),
                 errors::InvalidArgument("AddBuoyancy expects that the dimension size is equal to 1 for flags and density input"));

    const FluidGrid fluidGrid = {
        vel_shape.dim_size(0),              // batches
        vel_shape.dim_size(1),              // width
        vel_shape.dim_size(2),              // height
        vel_shape.dim_size(3),              // depth
        force_shape.dim_size(0),            // dim

        input_vel_flat.data(),
        density_flat.data(),
        flag_grid_flat.data()
    };


    // Create an output tensor
    Tensor* output_vel_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, vel_shape, &output_vel_tensor));


    AddBuoyancy<Device>()(
        context->eigen_device<Device>(),
        &fluidGrid,
        force_tensor.flat<float>().data(),
        output_vel_tensor->flat<float>().data());

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


