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



struct Vec3 {
    float x;
    float y;
    float z;
    
    Vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

//! Multiplication operator
inline Vec3 operator* ( const Vec3& v, float s ) {
	return Vec3 ( v.x*s, v.y*s, v.z*s );
}

//! Subtraction operator
inline Vec3 operator- ( const Vec3 &v1, const Vec3 &v2 ) {
	return Vec3 ( v1.x-v2.x, v1.y-v2.y, v1.z-v2.z );
}


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



#include <iostream>
#include <initializer_list>


bool isSameValue(std::initializer_list<long> list)
{
    for (int i = 1; i < list.size(); i++)
        if(list.begin()[0] != list.begin()[i])
            return false;
    return true;
}


#endif // FLUID_GRID_H_
