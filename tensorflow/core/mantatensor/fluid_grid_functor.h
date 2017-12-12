#ifndef FLUID_GRID_FUNCTOR_H_
#define FLUID_GRID_FUNCTOR_H_

#include "fluid_grid.h"

class FluidGridFunctor {
    private:
        int m_i_bxyz_offset[7];
        int m_i_bxyzd_offset[7];
        
        
        inline float cubicInterp(const float interp, const float* points);
        
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
        inline float interpolateCubic(const float* grid, const Vec3& pos, int i_b);
        
        
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



inline float FluidGridFunctor::cubicInterp(const float interp, const float* points) { 
  float d0 = (points[2] - points[0]) * 0.5;
  float d1 = (points[3] - points[1]) * 0.5; 
  float deltak = (points[2] - points[1]);

  // disabled: if (deltak * d0 < 0.0) d0 = 0;
  // disabled: if (deltak * d1 < 0.0) d1 = 0;

  float a0 = points[1];
  float a1 = d0;
  float a2 = 3.0f * deltak - 2.0f * d0 - d1;
  float a3 = -2.0f * deltak + d0 + d1;

  float squared = interp * interp;
  float cubed = squared * interp;
  return a3 * cubed + a2 * squared + a1 * interp + a0;
}

inline float FluidGridFunctor::interpolateCubic(const float* data, const Vec3& pos, int i_b) { 
	const float px = pos.x - 0.5f;
        const float py = pos.y - 0.5f;
        const float pz = pos.z - 0.5f; 

	const int x1 = (int)px;
	const int x2    = x1 + 1;
	const int x3    = x1 + 2;
	const int x0    = x1 - 1;

	const int y1 = (int)py;
	const int y2    = y1 + 1;
	const int y3    = y1 + 2;
	const int y0    = y1 - 1;

	const int z1 = (int)pz;
	const int z2    = z1 + 1;
	const int z3    = z1 + 2;
	const int z0    = z1 - 1;

	if (x0 < 0 || y0 < 0 || z0 < 0 || x3 >= getWidth() || y3 >= getHeight() || z3 >= getDepth()) {
            return interpolate(data, pos, i_b);
	}

	const float xInterp = px - x1;
	const float yInterp = py - y1;
	const float zInterp = pz - z1;

	const int slabsize = getWidth() * getHeight();
	const int z0Slab = z0 * slabsize;
	const int z1Slab = z1 * slabsize;
	const int z2Slab = z2 * slabsize;
	const int z3Slab = z3 * slabsize;

	const int y0x = y0 * getWidth();
	const int y1x = y1 * getWidth();
	const int y2x = y2 * getWidth();
	const int y3x = y3 * getWidth();

	const int y0z0 = y0x + z0Slab;
	const int y1z0 = y1x + z0Slab;
	const int y2z0 = y2x + z0Slab;
	const int y3z0 = y3x + z0Slab;

	const int y0z1 = y0x + z1Slab;
	const int y1z1 = y1x + z1Slab;
	const int y2z1 = y2x + z1Slab;
	const int y3z1 = y3x + z1Slab;

	const int y0z2 = y0x + z2Slab;
	const int y1z2 = y1x + z2Slab;
	const int y2z2 = y2x + z2Slab;
	const int y3z2 = y3x + z2Slab;

	const int y0z3 = y0x + z3Slab;
	const int y1z3 = y1x + z3Slab;
	const int y2z3 = y2x + z3Slab;
	const int y3z3 = y3x + z3Slab;

	// get the z0 slice
	const float p0[]  = {data[x0 + y0z0], data[x1 + y0z0], data[x2 + y0z0], data[x3 + y0z0]};
	const float p1[]  = {data[x0 + y1z0], data[x1 + y1z0], data[x2 + y1z0], data[x3 + y1z0]};
	const float p2[]  = {data[x0 + y2z0], data[x1 + y2z0], data[x2 + y2z0], data[x3 + y2z0]};
	const float p3[]  = {data[x0 + y3z0], data[x1 + y3z0], data[x2 + y3z0], data[x3 + y3z0]};

	// get the z1 slice
	const float p4[]  = {data[x0 + y0z1], data[x1 + y0z1], data[x2 + y0z1], data[x3 + y0z1]};
	const float p5[]  = {data[x0 + y1z1], data[x1 + y1z1], data[x2 + y1z1], data[x3 + y1z1]};
	const float p6[]  = {data[x0 + y2z1], data[x1 + y2z1], data[x2 + y2z1], data[x3 + y2z1]};
	const float p7[]  = {data[x0 + y3z1], data[x1 + y3z1], data[x2 + y3z1], data[x3 + y3z1]};

	// get the z2 slice
	const float p8[]  = {data[x0 + y0z2], data[x1 + y0z2], data[x2 + y0z2], data[x3 + y0z2]};
	const float p9[]  = {data[x0 + y1z2], data[x1 + y1z2], data[x2 + y1z2], data[x3 + y1z2]};
	const float p10[] = {data[x0 + y2z2], data[x1 + y2z2], data[x2 + y2z2], data[x3 + y2z2]};
	const float p11[] = {data[x0 + y3z2], data[x1 + y3z2], data[x2 + y3z2], data[x3 + y3z2]};

	// get the z3 slice
	const float p12[] = {data[x0 + y0z3], data[x1 + y0z3], data[x2 + y0z3], data[x3 + y0z3]};
	const float p13[] = {data[x0 + y1z3], data[x1 + y1z3], data[x2 + y1z3], data[x3 + y1z3]};
	const float p14[] = {data[x0 + y2z3], data[x1 + y2z3], data[x2 + y2z3], data[x3 + y2z3]};
	const float p15[] = {data[x0 + y3z3], data[x1 + y3z3], data[x2 + y3z3], data[x3 + y3z3]};

	// interpolate
	const float z0Points[] = {cubicInterp(xInterp, p0),  cubicInterp(xInterp, p1),  cubicInterp(xInterp, p2),  cubicInterp(xInterp, p3)};
	const float z1Points[] = {cubicInterp(xInterp, p4),  cubicInterp(xInterp, p5),  cubicInterp(xInterp, p6),  cubicInterp(xInterp, p7)};
	const float z2Points[] = {cubicInterp(xInterp, p8),  cubicInterp(xInterp, p9),  cubicInterp(xInterp, p10), cubicInterp(xInterp, p11)};
	const float z3Points[] = {cubicInterp(xInterp, p12), cubicInterp(xInterp, p13), cubicInterp(xInterp, p14), cubicInterp(xInterp, p15)};

	const float finalPoints[] = {cubicInterp(yInterp, z0Points), cubicInterp(yInterp, z1Points), cubicInterp(yInterp, z2Points), cubicInterp(yInterp, z3Points)};

	return cubicInterp(zInterp, finalPoints);
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


