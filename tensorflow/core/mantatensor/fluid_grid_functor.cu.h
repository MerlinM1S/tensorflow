#ifndef FLUID_GRID_FUNCTOR_CU_H_
#define FLUID_GRID_FUNCTOR_CU_H_

#include "fluid_grid.h"

// TODO: Use __ldg
class FluidGridFunctorGPU {
    private:
        int dimOffset[3];
        
    public: 
        FluidGrid* fluidGrid;
        
        __device__ int getBatches() { return fluidGrid->batches; };
        __device__ int getWidth()   { return fluidGrid->width; };
        __device__ int getHeight()  { return fluidGrid->height; };
        __device__ int getDepth()   { return fluidGrid->depth; };
        __device__ int getDim()     { return fluidGrid->dim; };
        

        __device__ const float* getVel() { return fluidGrid->vel; };
        __device__ const float* getDen() { return fluidGrid->den; };
        
        __device__ int getDimOffset(int dim) { return dimOffset[dim];}
        
        __device__ bool isFluid(int idx)    const { return fluidGrid->flags[idx] & TypeFluid; }
        __device__ bool isObstacle(int idx) const { return fluidGrid->flags[idx] & TypeObstacle; }
        __device__ bool isInflow(int idx)   const { return fluidGrid->flags[idx] & TypeInflow; }
        __device__ bool isEmpty(int idx)    const { return fluidGrid->flags[idx] & TypeEmpty; }
        __device__ bool isOutflow(int idx)  const { return fluidGrid->flags[idx] & TypeOutflow; }
        __device__ bool isOpen(int idx)     const { return fluidGrid->flags[idx] & TypeOpen; }
        __device__ bool isStick(int idx)    const { return fluidGrid->flags[idx] & TypeStick; }
        
    public:
        FluidGridFunctorGPU (FluidGrid* fluidGrid);
        
};

FluidGridFunctorGPU::FluidGridFunctorGPU (FluidGrid* fluidGrid) {
    this->fluidGrid = fluidGrid;
    
    dimOffset[0] = -fluidGrid->depth*fluidGrid->height;
    dimOffset[1] = -fluidGrid->depth;
    dimOffset[2] = -1;
}

#endif // FLUID_GRID_FUNCTOR_CU_H_