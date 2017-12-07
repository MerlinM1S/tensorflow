#ifndef FLUID_GRID_FUNCTOR_H_
#define FLUID_GRID_FUNCTOR_H_

#include "fluid_grid.h"

class FluidGridFunctor {
    private:
        int dimOffset[3];
        
    public: 
        FluidGrid* fluidGrid;
        
        inline int getBatches()  { return fluidGrid->batches; };
        inline int getWidth()      { return fluidGrid->width; };
        inline int getHeight()     { return fluidGrid->height; };
        inline int getDepth()      { return fluidGrid->depth; };
        inline int getDim()        { return fluidGrid->dim; };
        

        inline const float* getVel() { return fluidGrid->vel; };
        inline const float* getDen() { return fluidGrid->den; };
        
        inline int getDimOffset(int dim) { return dimOffset[dim];}
        
        inline bool isFluid(int idx)    const { return fluidGrid->flags[idx] & TypeFluid; }
        inline bool isObstacle(int idx) const { return fluidGrid->flags[idx] & TypeObstacle; }
        inline bool isInflow(int idx)   const { return fluidGrid->flags[idx] & TypeInflow; }
        inline bool isEmpty(int idx)    const { return fluidGrid->flags[idx] & TypeEmpty; }
        inline bool isOutflow(int idx)  const { return fluidGrid->flags[idx] & TypeOutflow; }
        inline bool isOpen(int idx)     const { return fluidGrid->flags[idx] & TypeOpen; }
        inline bool isStick(int idx)    const { return fluidGrid->flags[idx] & TypeStick; }
        
    public:
        FluidGridFunctor (FluidGrid* fluidGrid);
        
};

FluidGridFunctor::FluidGridFunctor (FluidGrid* fluidGrid) {
    this->fluidGrid = fluidGrid;
    
    dimOffset[0] = -fluidGrid->depth*fluidGrid->height;
    dimOffset[1] = -fluidGrid->depth;
    dimOffset[2] = -1;
}

#endif // FLUID_GRID_FUNCTOR_H_