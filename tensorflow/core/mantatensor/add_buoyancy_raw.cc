#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

namespace mantaflow {


REGISTER_OP("AddBuoyancyRaw")
    .Input("flags_grid: int32")
    .Input("density_grid: float")
    .Input("in_vel_grid: float")
    .Input("force: float")
    .Output("out_vel_grid: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

//! types of cells, in/outflow can be combined, e.g., TypeFluid|TypeInflow
enum CellType {
    TypeNone     = 0,
    TypeFluid    = 1,
    TypeObstacle = 2,
    TypeEmpty    = 4,
    TypeInflow   = 8,
    TypeOutflow  = 16,
    TypeOpen     = 32,
    TypeStick    = 64,
    // internal use only, for fast marching
    TypeReserved = 256,
    // 2^10 - 2^14 reserved for moving obstacles
};


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <typename Device>
class AddBuoyancyOp : public OpKernel {
 public:
  explicit AddBuoyancyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& flag_grid_tensor = context->input(0);
    auto flag_grid_flat = flag_grid_tensor.flat<int>();

    const Tensor& density_tensor = context->input(1);
    auto density_flat = density_tensor.flat<float>();

    const Tensor& input_vel_tensor = context->input(2);
    TensorShape vel_shape = input_vel_tensor.shape();
    auto input_vel_flat = input_vel_tensor.flat<float>();

    const Tensor& force_tensor = context->input(3);
    TensorShape force_shape = force_tensor.shape();
    auto force_flat = force_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_vel_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, vel_shape, &output_vel_tensor));

    auto output_vel_flat = output_vel_tensor->flat<float>();


    int width = vel_shape.dim_size(0);
    int height = vel_shape.dim_size(1);
    int depth = vel_shape.dim_size(2);

    int dim = force_shape.dim_size(0);


    int idx = 0;                        // int index = x + y*width + z*width*height;
    int idxi = 0;                       // int index = idx*dim + i



    int dimOffset[3] = {-width*height, -width, -1 };

    for (int x = 0; x < width; x++) {
        bool xInside = x > 0 && x < width - 1;
        for (int y = 0; y < height; y++) {
            bool yInside = y > 0 && y < height - 1;
            for (int z = 0; z < depth; z++) {
                bool zInside = z > 0 && z < depth - 1;
                bool isIdxFluid = flag_grid_flat(idx) & TypeFluid;
                for(int i = 0; i < dim; i++) {
                    float value = input_vel_flat(idxi);

                    int idxNeighbour = idx +  dimOffset[i];
                    bool isNeighbourFluid = flag_grid_flat(idxNeighbour) & TypeFluid;
                    if(xInside && yInside && zInside && isIdxFluid && isNeighbourFluid) {
                        value += (0.5f*force_flat(i)) * (density_flat(idx) + density_flat(idxNeighbour));
                    }

                    output_vel_flat(idxi) = value;
                    idxi++;
                }
                idx++;
            }
        }
    }
  }
};

// Register the CPU kernels.
                                        \
REGISTER_KERNEL_BUILDER(                                       \
      Name("AddBuoyancyRaw").Device(DEVICE_CPU), AddBuoyancyOp<CPUDevice>);
}
