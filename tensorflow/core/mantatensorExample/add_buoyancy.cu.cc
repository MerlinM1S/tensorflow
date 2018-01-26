#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "kernel_base.cu.h"
#include "fluid_grid_functor.cu.h"
#include "add_buoyancy.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


using GPUDevice = Eigen::GpuDevice;



class CudaKernelBuoyancy : public CudaKernelBase {
    private:
        float* out_vel;
        const float* force;

    protected:
        __device__ void kernelFunction(int i_bxyz) {
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

        __device__ void kernelIdentityDim(int i_bxyzd) {
            out_vel[i_bxyzd] = fluidGridFunctor.getVelGrid()[i_bxyzd];
        }

    public:
        __device__ CudaKernelBuoyancy(const CudaFluidGridFunctor& fluidGridFunctor, float* out_vel, const float* force) : CudaKernelBase(fluidGridFunctor, 1) {
            this->out_vel = out_vel;
            this->force = force;
        }
};





// TODO: Use __ldg
// Define the CUDA kernel.
__global__ void addBuoyancyKernel( const CudaFluidGridFunctor fluidGridFunctor, float* out_vel, const float* force) {
    //CudaKernelBuoyancy(fluidGridFunctor, out_vel, force).run();

    int border = 1;
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
    //cudaKernelBuoyancy.run<<<block_count, thread_per_block, 0, d.stream()>>>();
    addBuoyancyKernel<<<block_count, thread_per_block, 0, d.stream()>>>(fluidGridFunctor, out_vel, force);
}

#endif  // GOOGLE_CUDA
