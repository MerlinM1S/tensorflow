#ifndef INTERPOLATION_CU_H_
#define INTERPOLATION_CU_H_

#include "fluid_grid_functor.cu.h"


__device__ float cudaInterpolate(const CudaGridData& gridData, const float* grid, const CudaVec3& pos, const int b_i, const int d_i = 0) {
    float px = pos.x - 0.5f;
    float py = pos.y - 0.5f;
    float pz = pos.z - 0.5f; 
    
    int x_i = (int)px; 
    int y_i = (int)py; 
    int z_i = (int)pz; 
    
    float s1 = px - x_i; 
    float t1 = py - y_i;
    float f1 = pz - z_i;
    
    float s0 = 1.0f - s1; 
    float t0 = 1.0f - t1; 
    float f0 = 1.0f - f1; 
    
    // clamp to border 
    if (px < 0.0f) { x_i = 0; s0 = 1.0f; s1 = 0.0f; } 
    if (py < 0.0f) { y_i = 0; t0 = 1.0f; t1 = 0.0f; } 
    if (pz < 0.0f) { z_i = 0; f0 = 1.0f; f1 = 0.0f; } 
    
    if (x_i >= gridData.width - 1)  { x_i = gridData.width - 2;     s0 = 0.0f; s1 = 1.0f; } 
    if (y_i >= gridData.height - 1) { y_i = gridData.height - 2;    t0 = 0.0f; t1 = 1.0f; } 
    if (z_i >= gridData.depth - 1)  { z_i = gridData.depth - 2;     f0 = 0.0f; f1 = 1.0f; }
    
    int index = gridData.convertToIndex(b_i, x_i, y_i, z_i, d_i);
    
    return  ((grid[index]                                               *t0 + grid[gridData.offsetIndex(index, DirectionUp)]             *t1) * s0
           + (grid[gridData.offsetIndex(index, DirectionRight)]         *t0 + grid[gridData.offsetIndex(index, DirectionRightUp)]        *t1) * s1) * f0
           +((grid[gridData.offsetIndex(index, DirectionForward)]       *t0 + grid[gridData.offsetIndex(index, DirectionUpForward)]      *t1) * s0
           + (grid[gridData.offsetIndex(index, DirectionRightForward)]  *t0 + grid[gridData.offsetIndex(index, DirectionRightUpForward)] *t1) * s1) * f1;
}


#endif // INTERPOLATION_CU_H_