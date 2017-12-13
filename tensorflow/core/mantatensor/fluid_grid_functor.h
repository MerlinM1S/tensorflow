#ifndef FLUID_GRID_FUNCTOR_H_
#define FLUID_GRID_FUNCTOR_H_

#include "fluid_grid.h"

struct GridInfo {
    public:
        long batches;
        long width;
        long height;
        long depth;
        long dim;
        
        int indexOffset[7];
        
        GridInfo() {}
    
        GridInfo (long batches, long width, long height, long depth, long dim) {
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
        
        inline int offsetIndex(int index, int direction) const { return index + indexOffset[direction]; }
        inline int offsetIndex(int index, int x_d, int y_d, int z_d) const { return index + x_d * indexOffset[(int) DirectionRight] + y_d * indexOffset[(int) DirectionUp] + z_d * indexOffset[(int) DirectionForward]; }
        
        inline int convertToIndex(int b_i, int x_i, int y_i, int z_i, int d_i)  const { return d_i + dim * (z_i + depth * (y_i + height * (x_i + b_i * width))); }
};

class FluidGridFunctor {
    private:
        inline float cubicInterp(const float interp, const float* points);
        
    public: 
        const FluidGrid* pFluidGrid;
        GridInfo gridInfo1D;
        GridInfo gridInfo;
        
        inline int getBatches()   const { return pFluidGrid->batches; }
        inline int getWidth()     const { return pFluidGrid->width; }
        inline int getHeight()    const { return pFluidGrid->height; }
        inline int getDepth()     const { return pFluidGrid->depth; }
        inline int getDim()       const { return pFluidGrid->dim; }
        

        inline const float* getVelGrid() const { return pFluidGrid->vel; }
        inline const float* getDenGrid() const { return pFluidGrid->den; }
        
        inline bool isFluid(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeFluid; }
        inline bool isObstacle(int i_bxyz) const { return pFluidGrid->flags[i_bxyz] & TypeObstacle; }
        inline bool isInflow(int i_bxyz)   const { return pFluidGrid->flags[i_bxyz] & TypeInflow; }
        inline bool isEmpty(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeEmpty; }
        inline bool isOutflow(int i_bxyz)  const { return pFluidGrid->flags[i_bxyz] & TypeOutflow; }
        inline bool isOpen(int i_bxyz)     const { return pFluidGrid->flags[i_bxyz] & TypeOpen; }
        inline bool isStick(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeStick; }
        
        inline Vec3 getCenteredVel(int i_bxyz) const;
        
        inline Vec3 getVelMACX(int i_bxyz) const;
        inline Vec3 getVelMACY(int i_bxyz) const;
        inline Vec3 getVelMACZ(int i_bxyz) const;
        
    public:
        FluidGridFunctor (const FluidGrid* pFluidGrid);
        
};

FluidGridFunctor::FluidGridFunctor (const FluidGrid* pFluidGrid) {
    this->pFluidGrid = pFluidGrid;
    
    gridInfo1D = GridInfo(pFluidGrid->batches, pFluidGrid->width, pFluidGrid->height, pFluidGrid->depth, 1);
    gridInfo = GridInfo(pFluidGrid->batches, pFluidGrid->width, pFluidGrid->height, pFluidGrid->depth, pFluidGrid->dim);
}


// "+ 0" = ".x"; "+ 1" = ".y"; "+ 2" = ".z"
inline Vec3 FluidGridFunctor::getCenteredVel(int i_bxyz) const {
    int i_bxyzd = i_bxyz * this->getDim();
    
    float x = 0.5f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, DirectionRight) + 0]);             
    float y = 0.5f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, DirectionUp) + 1]);
    float z = 0.5f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, DirectionForward) + 2]);
    return Vec3(x, y, z);
}



inline Vec3 FluidGridFunctor::getVelMACX(int i_bxyzd) const {
    float x = getVelGrid()[i_bxyzd + 0];
    float y = 0.25f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, -1, 0, 0) + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 1, 0) + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, -1, 1, 0) + 1]);
    float z = 0.25f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, -1, 0, 0) + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 0, 1) + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, -1, 0, 1) + 2]);
    
    return Vec3(x, y, z);
}

inline Vec3 FluidGridFunctor::getVelMACY(int i_bxyzd) const {
    float x = 0.25f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, -1, 0) + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 1, 0, 0) + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 1, -1, 0) + 0]);
    float y = getVelGrid()[i_bxyzd + 1];
    float z = 0.25f * (getVelGrid()[i_bxyzd + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, -1, 0) + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 0, 1) + 2] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, -1, 1) + 2]);
    
    return Vec3(x, y, z);
}


inline Vec3 FluidGridFunctor::getVelMACZ(int i_bxyzd) const {
    float x = 0.25f * (getVelGrid()[i_bxyzd + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 0, -1) + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 1, 0, 0) + 0] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 1, 0, -1) + 0]);
    float y = 0.25f * (getVelGrid()[i_bxyzd + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 0, -1) + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 1, 0) + 1] + getVelGrid()[gridInfo.offsetIndex(i_bxyzd, 0, 1, -1) + 1]);
    float z = getVelGrid()[i_bxyzd + 2];
    
    return Vec3(x, y, z);
}




#include <iostream>
#include <initializer_list>


bool isSameValue(std::initializer_list<long> list)
{
    for (int i = 1; i < list.size(); i++)
        if(list.begin()[0] != list.begin()[i])
            return false;
    return true;
}


#endif // FLUID_GRID_FUNCTOR_H_


