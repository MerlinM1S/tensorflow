#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "fluid_grid_functor.cu.h"
#include "add_buoyancy.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


using GPUDevice = Eigen::GpuDevice;

// TODO: idx wont work like that!
// TODO: Use __ldg
// Define the CUDA kernel.
__global__ void AddBuoyancyKernel(FluidGridFunctorGPU* pFluidGridFunctor, const float* force, float* out_vel) {
    int idx = 0;
    int idxi = 0;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < pFluidGridFunctor->getBatches(); b += blockDim.x * gridDim.x) {
        for (int x = 0; x < pFluidGridFunctor->getWidth(); x ++) {
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

// Define the GPU implementation that launches the CUDA kernel.
template <>
void AddBuoyancy<GPUDevice>::operator()(
    const GPUDevice& d, FluidGrid* fluidGrid, const float* force, float* out_vel) {

    FluidGridFunctorGPU fluidGridFunctor(fluidGrid);

  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  AddBuoyancyKernel
      <<<block_count, thread_per_block, 0, d.stream()>>>(&fluidGridFunctor, force, out_vel);
}


#endif  // GOOGLE_CUDA
