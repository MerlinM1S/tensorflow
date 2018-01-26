#ifndef FLUID_GRID_FUNCTOR_CU_H_
#define FLUID_GRID_FUNCTOR_CU_H_

#include "fluid_grid.h"
#include "vec3.cu.h"

struct CudaGridData {
    public:
        long batches;
        long width;
        long height;
        long depth;
        long dim;
        
        int indexOffset[7];
        
        CudaGridData() {}
    
        CudaGridData (long batches, long width, long height, long depth, long dim) {
            this->batches   = batches;   
            this->width     = width;   
            this->height    = height;   
            this->depth     = depth;   
            this->dim       = dim;   
            
            indexOffset[(int) DirectionRight]   = dim * depth * height;
            indexOffset[(int) DirectionUp]      = dim * depth;
            indexOffset[(int) DirectionForward] = dim;
        
            indexOffset[(int) DirectionRightUp]         = indexOffset[DirectionRight] +  indexOffset[DirectionUp];
            indexOffset[(int) DirectionRightForward]    = indexOffset[DirectionRight] +  indexOffset[DirectionForward];
            indexOffset[(int) DirectionUpForward]       = indexOffset[DirectionUp]    +  indexOffset[DirectionForward];
        
            indexOffset[(int) DirectionRightUpForward]  = indexOffset[DirectionRight] +  indexOffset[DirectionUp] + indexOffset[DirectionForward];
        }
        
        __device__ int getOffset(int direction) const { return indexOffset[direction]; }
        __device__ int offsetIndex(int index, int direction) const { return index + indexOffset[direction]; }
        __device__ int offsetIndex(int index, int x_d, int y_d, int z_d) const { return index + x_d * indexOffset[(int) DirectionRight] + y_d * indexOffset[(int) DirectionUp] + z_d * indexOffset[(int) DirectionForward]; }
        
        __device__ int convertToIndex(int b_i, int x_i, int y_i, int z_i, int d_i)  const { return d_i + dim * (z_i + depth * (y_i + height * (x_i + b_i * width))); }
};

// TODO: Use __ldg
class CudaFluidGridFunctor {
    private:
        const long batches;
        const long width;
        const long height;
        const long depth;
        const long dim;
        

        const float* vel;
        const float* den;
        const int* flags;
        
    public: 
        CudaGridData gridData1D;
        CudaGridData gridData;
        
        __device__ int getBatches() const { return batches; };
        __device__ int getWidth()   const { return width; };
        __device__ int getHeight()  const { return height; };
        __device__ int getDepth()   const { return depth; };
        __device__ int getDim()     const { return dim; };
        

        __device__ const float* getVelGrid() const { return vel; };
        __device__ const float* getDenGrid() const { return den; };
        
        __device__ bool isFluid(int idx)    const { return flags[idx] & TypeFluid; }
        __device__ bool isObstacle(int idx) const { return flags[idx] & TypeObstacle; }
        __device__ bool isInflow(int idx)   const { return flags[idx] & TypeInflow; }
        __device__ bool isEmpty(int idx)    const { return flags[idx] & TypeEmpty; }
        __device__ bool isOutflow(int idx)  const { return flags[idx] & TypeOutflow; }
        __device__ bool isOpen(int idx)     const { return flags[idx] & TypeOpen; }
        __device__ bool isStick(int idx)    const { return flags[idx] & TypeStick; }
        
        __device__ CudaVec3 getCenteredVel(int i_bxyz) const;
        
        __device__ CudaVec3 getVelMACX(int i_bxyz) const;
        __device__ CudaVec3 getVelMACY(int i_bxyz) const;
        __device__ CudaVec3 getVelMACZ(int i_bxyz) const;
        
    public:
        CudaFluidGridFunctor (const FluidGrid* pFluidGrid);
        
};

CudaFluidGridFunctor::CudaFluidGridFunctor (const FluidGrid* pFluidGrid) : batches(pFluidGrid->batches), width(pFluidGrid->width), height(pFluidGrid->height), depth(pFluidGrid->depth), dim(pFluidGrid->dim) {  
    this->vel       = pFluidGrid->vel;
    this->den       = pFluidGrid->den;
    this->flags     = pFluidGrid->flags;
    
    
    gridData1D = CudaGridData(pFluidGrid->batches, pFluidGrid->width, pFluidGrid->height, pFluidGrid->depth, 1);
    gridData = CudaGridData(pFluidGrid->batches, pFluidGrid->width, pFluidGrid->height, pFluidGrid->depth, pFluidGrid->dim);
}

// "+ 0" = ".x"; "+ 1" = ".y"; "+ 2" = ".z"
__device__ CudaVec3 CudaFluidGridFunctor::getCenteredVel(int i_bxyz) const {
    int i_bxyzd = i_bxyz * this->getDim();
    
    float x = 0.5f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, DirectionRight) + 0]);             
    float y = 0.5f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, DirectionUp) + 1]);
    float z = 0.5f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, DirectionForward) + 2]);
    return CudaVec3(x, y, z);
}



__device__ CudaVec3 CudaFluidGridFunctor::getVelMACX(int i_bxyzd) const {
    float x = getVelGrid()[i_bxyzd + 0];
    float y = 0.25f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, -1, 0, 0) + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 1, 0) + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, -1, 1, 0) + 1]);
    float z = 0.25f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, -1, 0, 0) + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 0, 1) + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, -1, 0, 1) + 2]);
    
    return CudaVec3(x, y, z);
}

__device__ CudaVec3 CudaFluidGridFunctor::getVelMACY(int i_bxyzd) const {
    float x = 0.25f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, -1, 0) + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 1, 0, 0) + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 1, -1, 0) + 0]);
    float y = getVelGrid()[i_bxyzd + 1];
    float z = 0.25f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, -1, 0) + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 0, 1) + 2] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, -1, 1) + 2]);
    
    return CudaVec3(x, y, z);
}


__device__ CudaVec3 CudaFluidGridFunctor::getVelMACZ(int i_bxyzd) const {
    float x = 0.25f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 0, -1) + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 1, 0, 0) + 0] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 1, 0, -1) + 0]);
    float y = 0.25f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 0, -1) + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 1, 0) + 1] + getVelGrid()[gridData.offsetIndex(i_bxyzd, 0, 1, -1) + 1]);
    float z = getVelGrid()[i_bxyzd + 2];
    
    return CudaVec3(x, y, z);
}


#endif // FLUID_GRID_FUNCTOR_CU_H_