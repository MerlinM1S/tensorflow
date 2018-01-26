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
    long batches;
    long width;
    long height;
    long depth;
    long dim;
        

    const float* vel;
    const float* den;
    const int* flags;
};



/*
struct GridVec3 {
    float* base
    
    Vec3(float* base) {
        this->base = base;
    }
    
    
    // Read
    const Vec3& operator()(int i_bxyz) const { return Vec3(base[i_bxyz + 0], base[i_bxyz + 1], base[i_bxyz + 2]); }
    
    // Read
    const float& operator()(int i_bxyz, int d) const { return base[i_bxyz*3 + d]; }
};
*/

enum Direction {
    DirectionRight          = 0, 
    DirectionUp             = 1, 
    DirectionForward        = 2,
    
    DirectionRightUp        = 3,
    DirectionRightForward   = 4,
    DirectionUpForward      = 5,
    DirectionRightUpForward = 6,    
};




#endif // FLUID_GRID_H_
