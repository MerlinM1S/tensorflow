#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "fluid_grid_functor.cu.h"
#include "add_buoyancy.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


using GPUDevice = Eigen::GpuDevice;

// TODO: Use __ldg
// Define the CUDA kernel.
__global__ void AddBuoyancyKernel(const CudaFluidGridFunctor fluidGridFunctor,  const float* force, float* out_vel) {


//= blockIdx.x * blockDim.x + threadIdx.x; x < pFluidGridFunctor->getWidth(); x += blockDim.x * gridDim.x
    for (int b = 0; b < fluidGridFunctor.getBatches(); b++) {
        int i_b = b * fluidGridFunctor.getWidth() * fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();

        for (int x = 0; x < fluidGridFunctor.getWidth(); x++) {
            int i_bx = i_b + x*fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();
            bool xInside = x >= 1 && x < fluidGridFunctor.getWidth() - 1;
            for (int y = 0; y < fluidGridFunctor.getHeight(); y++) {
                int i_bxy = i_bx + y * fluidGridFunctor.getDepth();
                bool yInside = y >= 1 && y < fluidGridFunctor.getHeight() - 1;
                for (int z = 0; z < fluidGridFunctor.getDepth(); z++) {
                    int i_bxyz = i_bxy + z;
                    bool zInside = z >= 1 && z < fluidGridFunctor.getDepth() - 1;
                    bool isIdxFluid = fluidGridFunctor.isFluid(i_bxyz);

                    for(int d = 0; d < fluidGridFunctor.getDim(); d++) {

                        int i_bxyzd = d + i_bxyz*3;
                        float value = fluidGridFunctor.getVelGrid()[i_bxyzd];

                        int idxNeighbour = i_bxyz - fluidGridFunctor.gridData1D.getOffset(d);
                        bool isNeighbourFluid = fluidGridFunctor.isFluid(idxNeighbour);
                        if(xInside && yInside && zInside && isIdxFluid && isNeighbourFluid) {
                            value += (0.5f*force[d]) * (fluidGridFunctor.getDenGrid()[i_bxyz] + fluidGridFunctor.getDenGrid()[idxNeighbour]);
                        }

                        out_vel[i_bxyzd] = value;

                    }
                }
            }
        }
    }

}


// Define the GPU implementation that launches the CUDA kernel.
template <>
void AddBuoyancy<GPUDevice>::operator()(
    const GPUDevice& d, const FluidGrid* pFluidGrid, const float* force, float* out_vel) {

    const CudaFluidGridFunctor fluidGridFunctor(pFluidGrid);

  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  AddBuoyancyKernel
      <<<block_count, thread_per_block, 0, d.stream()>>>(fluidGridFunctor, force, out_vel);
}


#endif  // GOOGLE_CUDA
