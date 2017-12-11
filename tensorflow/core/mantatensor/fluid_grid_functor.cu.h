#ifndef FLUID_GRID_FUNCTOR_CU_H_
#define FLUID_GRID_FUNCTOR_CU_H_

#include "fluid_grid.h"

// TODO: Use __ldg
class FluidGridFunctorGPU {
    private:
        int dimOffset[3];
        
        long batches;
        long width;
        long height;
        long depth;
        long dim;
        

        const float* vel;
        const float* den;
        const int* flags;
        
    public: 
        __device__ int getBatches() const { return batches; };
        __device__ int getWidth()   const { return width; };
        __device__ int getHeight()  const { return height; };
        __device__ int getDepth()   const { return depth; };
        __device__ int getDim()     const { return dim; };
        

        __device__ const float* getVel() const { return vel; };
        __device__ const float* getDen() const { return den; };
        
        __device__ int getDimOffset(int dim) const { return dimOffset[dim];}
        
        __device__ bool isFluid(int idx)    const { return flags[idx] & TypeFluid; }
        __device__ bool isObstacle(int idx) const { return flags[idx] & TypeObstacle; }
        __device__ bool isInflow(int idx)   const { return flags[idx] & TypeInflow; }
        __device__ bool isEmpty(int idx)    const { return flags[idx] & TypeEmpty; }
        __device__ bool isOutflow(int idx)  const { return flags[idx] & TypeOutflow; }
        __device__ bool isOpen(int idx)     const { return flags[idx] & TypeOpen; }
        __device__ bool isStick(int idx)    const { return flags[idx] & TypeStick; }
        
    public:
        FluidGridFunctorGPU (const FluidGrid* pFluidGrid);
        
};

FluidGridFunctorGPU::FluidGridFunctorGPU (const FluidGrid* pFluidGrid) {  
    this->batches   = pFluidGrid->batches;
    this->width     = pFluidGrid->width;
    this->height    = pFluidGrid->height;
    this->depth     = pFluidGrid->depth;
    this->dim       = pFluidGrid->dim;
    
    this->vel       = pFluidGrid->vel;
    this->den       = pFluidGrid->den;
    this->flags     = pFluidGrid->flags;
    
    dimOffset[0] = -pFluidGrid->depth*pFluidGrid->height;
    dimOffset[1] = -pFluidGrid->depth;
    dimOffset[2] = -1;
    

}

#endif // FLUID_GRID_FUNCTOR_CU_H_