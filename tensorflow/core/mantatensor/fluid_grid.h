#ifndef FLUID_GRID_H_
#define FLUID_GRID_H_

        //! types of cells, in/outflow can be combined, e.g., TypeFluid|TypeInflow
        enum CellType {
            TypeNone     = 0,
            TypeFluid    = 1,
            TypeObstacle = 2,
            TypeEmpty    = 4,
            TypeInflow   = 8,
            TypeOutflow  = 16,
            TypeOpen     = 32,
            TypeStick    = 64,
            // internal use only, for fast marching
            TypeReserved = 256,
            // 2^10 - 2^14 reserved for moving obstacles
        };

struct FluidGrid {
    int batches;
    int width;
    int height;
    int depth;
    int dim;
        

    const float* vel;
    const float* den;
    const int* flags;
};

#endif // FLUID_GRID_H_
