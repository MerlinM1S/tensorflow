#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "fluid_grid_functor.cu.h"
#include "interpolation.cu.h"
#include "advection.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


using GPUDevice = Eigen::GpuDevice;



// TODO: Use __ldg
// Define the CUDA kernel.
__global__ void advect1DKernel( const CudaFluidGridFunctor fluidGridFunctor, const float dt, const float* in_grid, float* out_grid, const int orderSpace) {
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

                    if(xInside && yInside && zInside) {
                        CudaVec3 pos = CudaVec3(x+0.5f, y+0.5f, z+0.5f) - fluidGridFunctor.getCenteredVel(i_bxyz) * dt;
                        switch(orderSpace) {
                        case 1:
                            out_grid[i_bxyz] = cudaInterpolate(fluidGridFunctor.gridData1D, in_grid, pos, b);
                            break;
                        case 2:
                            //out_grid[i_bxyz] = interpolateCubic(fluidGridFunctor.gridInfo1D, in_grid, pos, b);
                            break;
                        }
                    } else {
                      out_grid[i_bxyz] = 0;
                    }
                }
            }
        }
    }
}




// TODO: Use __ldg
// Define the CUDA kernel.
__global__ void advectMACKernel( const CudaFluidGridFunctor fluidGridFunctor, const float dt, const float* in_grid, float* out_grid, const int orderSpace) {
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

                    int i_bxyzd = i_bxyz * fluidGridFunctor.getDim();

                    if(xInside && yInside && zInside) {
                        CudaVec3 posOffH = CudaVec3(x+0.5f, y+0.5f, z+0.5f);

                        CudaVec3 xpos = posOffH - fluidGridFunctor.getVelMACX(i_bxyzd) * dt;
                        out_grid[i_bxyzd + 0] = cudaInterpolate(fluidGridFunctor.gridData, in_grid, xpos, b, 0);

                        CudaVec3 ypos = posOffH - fluidGridFunctor.getVelMACY(i_bxyzd) * dt;
                        out_grid[i_bxyzd + 1] = cudaInterpolate(fluidGridFunctor.gridData, in_grid, ypos, b, 1);

                        CudaVec3 zpos = posOffH - fluidGridFunctor.getVelMACZ(i_bxyzd) * dt;
                        out_grid[i_bxyzd + 2] = cudaInterpolate(fluidGridFunctor.gridData, in_grid, zpos, b, 2);
                    } else {
                        out_grid[i_bxyzd + 0] = 0;
                        out_grid[i_bxyzd + 1] = 0;
                        out_grid[i_bxyzd + 2] = 0;
                    }
                }
            }
        }
    }
}




template <>
void Advection<GPUDevice>::operator()(const GPUDevice& d, const FluidGrid* pFluidGrid, const float dt, const float* in_grid, float* out_grid, const int orderSpace, const AdvectionType advectionType) {

    const CudaFluidGridFunctor fluidGridFunctor(pFluidGrid);

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;

    switch(advectionType) {
    case AdvectionType1D:
        advect1DKernel<<<block_count, thread_per_block, 0, d.stream()>>>(fluidGridFunctor, dt, in_grid, out_grid, orderSpace);
        break;
    case AdvectionTypeMAC:
        advectMACKernel<<<block_count, thread_per_block, 0, d.stream()>>>(fluidGridFunctor, dt, in_grid, out_grid, orderSpace);
        break;
    }

}





#endif  // GOOGLE_CUDA
