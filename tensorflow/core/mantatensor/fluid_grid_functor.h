#ifndef FLUID_GRID_FUNCTOR_H_
#define FLUID_GRID_FUNCTOR_H_

#include "fluid_grid.h"

class FluidGridFunctor {
    private:
        int m_i_bxyz_offset[7];
        int m_i_bxyzd_offset[7];
        
    public: 
        const FluidGrid* pFluidGrid;
        
        inline int getBatches()    { return pFluidGrid->batches; }
        inline int getWidth()      { return pFluidGrid->width; }
        inline int getHeight()     { return pFluidGrid->height; }
        inline int getDepth()      { return pFluidGrid->depth; }
        inline int getDim()        { return pFluidGrid->dim; }
        

        inline const float* getVelGrid() { return pFluidGrid->vel; }
        inline const float* getDenGrid() { return pFluidGrid->den; }
        
        inline int i_bxyz_offset(int direction)     { return m_i_bxyz_offset[direction];}
        inline int i_bxyzd_offset(int direction)    { return m_i_bxyzd_offset[direction];}
        
        inline bool isFluid(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeFluid; }
        inline bool isObstacle(int i_bxyz) const { return pFluidGrid->flags[i_bxyz] & TypeObstacle; }
        inline bool isInflow(int i_bxyz)   const { return pFluidGrid->flags[i_bxyz] & TypeInflow; }
        inline bool isEmpty(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeEmpty; }
        inline bool isOutflow(int i_bxyz)  const { return pFluidGrid->flags[i_bxyz] & TypeOutflow; }
        inline bool isOpen(int i_bxyz)     const { return pFluidGrid->flags[i_bxyz] & TypeOpen; }
        inline bool isStick(int i_bxyz)    const { return pFluidGrid->flags[i_bxyz] & TypeStick; }
        
        inline Vec3 getCenteredVel(int i_bxyz);
        inline float interpolate(const float* grid, const Vec3& pos, int b);
        
        
        inline int convert_i_bxyz(int i_b, int i_x, int i_y, int i_z) { return i_z + getDepth() * (i_y + getHeight() * (i_x + i_b * getBatches())); }
        
        
        
    public:
        FluidGridFunctor (const FluidGrid* pFluidGrid);
        
};

FluidGridFunctor::FluidGridFunctor (const FluidGrid* pFluidGrid) {
    this->pFluidGrid = pFluidGrid;
    
    {
        m_i_bxyz_offset[(int) DirectionRight]   = pFluidGrid->depth*pFluidGrid->height;
        m_i_bxyz_offset[(int) DirectionUp]      = pFluidGrid->depth;
        m_i_bxyz_offset[(int) DirectionForward] = 1;
        
        m_i_bxyz_offset[(int) DirectionRightUp]         = i_bxyz_offset(DirectionRight) +  i_bxyz_offset(DirectionUp);
        m_i_bxyz_offset[(int) DirectionRightForward]    = i_bxyz_offset(DirectionRight) +  i_bxyz_offset(DirectionForward);
        m_i_bxyz_offset[(int) DirectionUpForward]       = i_bxyz_offset(DirectionUp)    +  i_bxyz_offset(DirectionForward);
        
        m_i_bxyz_offset[(int) DirectionRightUpForward]  = i_bxyz_offset(DirectionRight) +  i_bxyz_offset(DirectionUp) + i_bxyz_offset(DirectionForward);
    }
    
    for(int d = 0; d < getDim(); d++) {
        m_i_bxyzd_offset[d] = m_i_bxyz_offset[d] * getDim();
    }
}




inline Vec3 FluidGridFunctor::getCenteredVel(int i_bxyz) {
    int i_bxyzd = i_bxyz * 3;
    float x = 0.5f* (getVelGrid()[i_bxyzd + 0] + getVelGrid()[i_bxyzd + 0 + i_bxyzd_offset(DirectionRight)]);
    float y = 0.5f* (getVelGrid()[i_bxyzd + 1] + getVelGrid()[i_bxyzd + 1 + i_bxyzd_offset(DirectionUp)]);
    float z = 0.5f* (getVelGrid()[i_bxyzd + 2] + getVelGrid()[i_bxyzd + 2 + i_bxyzd_offset(DirectionForward)]);
    return Vec3(x, y, z);
}



inline float FluidGridFunctor::interpolate(const float* grid, const Vec3& pos, int i_b) {
    float px=pos.x - 0.5f;
    float py=pos.y - 0.5f;
    float pz=pos.z - 0.5f; 
    
    int i_x = (int)px; 
    int i_y = (int)py; 
    int i_z = (int)pz; 
    
    float s1 = px - i_x; 
    float t1 = py - i_y;
    float f1 = pz - i_z;
    
    float s0 = 1.0f - s1; 
    float t0 = 1.0f - t1; 
    float f0 = 1.0f - f1; 
    
    // clamp to border 
    if (px < 0.0f) { i_x = 0; s0 = 1.0f; s1 = 0.0f; } 
    if (py < 0.0f) { i_y = 0; t0 = 1.0f; t1 = 0.0f; } 
    if (pz < 0.0f) { i_z = 0; f0 = 1.0f; f1 = 0.0f; } 
    
    if (i_x >= getWidth() - 1) { i_x = getWidth() - 2; s0 = 0.0f; s1 = 1.0f; } 
    if (i_y >= getHeight() - 1) { i_y = getHeight() - 2; t0 = 0.0f; t1 = 1.0f; } 
    if (i_z >= getDepth() - 1) { i_z = getDepth() - 2; f0 = 0.0f; f1 = 1.0f; }
    
    int i_bxyz = convert_i_bxyz(i_b, i_x, i_y, i_z);
    
    return  ((grid[i_bxyz]                                          *t0 + grid[i_bxyz + i_bxyz_offset(DirectionUp)]             *t1) * s0
           + (grid[i_bxyz + i_bxyz_offset(DirectionRight)]          *t0 + grid[i_bxyz + i_bxyz_offset(DirectionRightUp)]        *t1) * s1) * f0
           +((grid[i_bxyz + i_bxyz_offset(DirectionForward)]        *t0 + grid[i_bxyz + i_bxyz_offset(DirectionUpForward)]      *t1) * s0
           + (grid[i_bxyz + i_bxyz_offset(DirectionRightForward)]   *t0 + grid[i_bxyz + i_bxyz_offset(DirectionRightUpForward)] *t1) * s1) * f1;
}









#endif // FLUID_GRID_FUNCTOR_H_


