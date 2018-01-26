#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "dim_size.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "MantaCPU/grid.h"
#include "MantaCPU/extforces.h"

using namespace tensorflow;
using namespace Manta;



template <typename Device>
struct addBuoyancy_Functor {
        void operator()(const Device& d, const DimSize dimSize, const int* in_flags, const float* in_density, const float* in_vel, float* out_vel, const float* in_gravity, const float* in_coefficient);
};

REGISTER_OP("AutoBuo")
        .Input("in_flags: int32")
        .Input("in_density: float")
        .Input("in_vel: float")
        .Input("in_gravity: float")
        .Input("in_coefficient: float")
        .Output("out_vel: float")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                c->set_output(0, c->input(2));
                return Status::OK();
        });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <>
void addBuoyancy_Functor<CPUDevice>::operator()(const CPUDevice& d, const DimSize dimSize, const int* in_flags, const float* in_density, const float* in_vel, float* out_vel, const float* in_gravity, const float* in_coefficient) {
        for(int i = 0; i < dimSize.lengthOf(5); i++) {
                out_vel[i] = in_vel[i];
        }

        for(int i_b = 0; i_b < dimSize.batches; i_b++) {
            FluidSolver fluidSolver = FluidSolver(Vec3i(dimSize.width, dimSize.height, dimSize.depth));

            const FlagGrid flags = FlagGrid(&fluidSolver, const_cast<int*>(in_flags + dimSize.batchToIndex(4, i_b)), 3, true);
            const Grid<float> density = Grid<float>(&fluidSolver, const_cast<float*>(in_density + dimSize.batchToIndex(4, i_b)), true);
            MACGrid vel = MACGrid(&fluidSolver, (Vec3*) (out_vel + dimSize.batchToIndex(5, i_b)), true);
            const Vec3 gravity = Vec3(in_gravity + (3 * i_b));
            const float coefficient = in_coefficient[i_b];


            addBuoyancy(flags, density, vel, gravity, coefficient);
        }
}


template <typename Device>
class addBuoyancy_OP : public OpKernel {
public:
        explicit addBuoyancy_OP(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& in_flags_tensor = context->input(0);
            const int* in_flags = in_flags_tensor.flat<int>().data();

            const Tensor& in_density_tensor = context->input(1);
            const float* in_density = in_density_tensor.flat<float>().data();

            const Tensor& in_vel_tensor = context->input(2);
            const float* in_vel = in_vel_tensor.flat<float>().data();

            const Tensor& in_gravity_tensor = context->input(3);
            const float* in_gravity = in_gravity_tensor.flat<float>().data();

            const Tensor& in_coefficient_tensor = context->input(4);
            const float* in_coefficient = in_coefficient_tensor.flat<float>().data();

            Tensor* out_vel_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, in_vel_tensor.shape(), &out_vel_tensor));
            float* out_vel = out_vel_tensor->flat<float>().data();

            long batches = in_vel_tensor.shape().dim_size(0);
            long width = in_vel_tensor.shape().dim_size(1);
            long height = in_vel_tensor.shape().dim_size(2);
            long depth = in_vel_tensor.shape().dim_size(3);
            long dim = in_vel_tensor.shape().dim_size(4);
            DimSize dimSize = DimSize(batches, width, depth, height, dim);

            addBuoyancy_Functor<Device>()(
                    context->eigen_device<Device>(),
                    dimSize,
                    in_flags, in_density, in_vel, out_vel, in_gravity, in_coefficient);
        }
};


REGISTER_KERNEL_BUILDER(Name("AutoBuo").Device(DEVICE_CPU), addBuoyancy_OP<CPUDevice>);
