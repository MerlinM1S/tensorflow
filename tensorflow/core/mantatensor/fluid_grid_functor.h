#ifndef FLUID_GRID_FUNCTOR_H_
#define FLUID_GRID_FUNCTOR_H_

#include "fluid_grid.h"

class FluidGridFunctor {
    private:
        int dimOffset[3];
        
    public: 
        const FluidGrid* pFluidGrid;
        
        inline int getBatches()    { return pFluidGrid->batches; }
        inline int getWidth()      { return pFluidGrid->width; }
        inline int getHeight()     { return pFluidGrid->height; }
        inline int getDepth()      { return pFluidGrid->depth; }
        inline int getDim()        { return pFluidGrid->dim; }
        

        inline const float* getVel() { return pFluidGrid->vel; }
        inline const float* getDen() { return pFluidGrid->den; }
        
        inline int getDimOffset(int dim) { return dimOffset[dim];}
        
        inline bool isFluid(int idx)    const { return pFluidGrid->flags[idx] & TypeFluid; }
        inline bool isObstacle(int idx) const { return pFluidGrid->flags[idx] & TypeObstacle; }
        inline bool isInflow(int idx)   const { return pFluidGrid->flags[idx] & TypeInflow; }
        inline bool isEmpty(int idx)    const { return pFluidGrid->flags[idx] & TypeEmpty; }
        inline bool isOutflow(int idx)  const { return pFluidGrid->flags[idx] & TypeOutflow; }
        inline bool isOpen(int idx)     const { return pFluidGrid->flags[idx] & TypeOpen; }
        inline bool isStick(int idx)    const { return pFluidGrid->flags[idx] & TypeStick; }
        
    public:
        FluidGridFunctor (const FluidGrid* pFluidGrid);
        
};

FluidGridFunctor::FluidGridFunctor (const FluidGrid* pFluidGrid) {
    this->pFluidGrid = pFluidGrid;
    
    dimOffset[0] = -pFluidGrid->depth*pFluidGrid->height;
    dimOffset[1] = -pFluidGrid->depth;
    dimOffset[2] = -1;
}

#endif // FLUID_GRID_FUNCTOR_H_