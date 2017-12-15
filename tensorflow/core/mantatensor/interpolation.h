#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_

#include "fluid_grid_functor.h"


inline float interpolate(const GridData& gridData, const float* grid, const Vec3& pos, const int b_i, const int d_i = 0) {
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


/*

inline float cubicInterp(const float interp, const float* points) { 
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

inline float interpolateCubic(const GridData& gridData, const float* data, const Vec3& pos, const int i_b) { 
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

	if (x0 < 0 || y0 < 0 || z0 < 0 || x3 >= gridData.width || y3 >= gridData.height || z3 >= gridData.depth) {
            return interpolate(gridData, data, pos, i_b);
	}

	const float xInterp = px - x1;
	const float yInterp = py - y1;
	const float zInterp = pz - z1;

	const int slabsize = gridData.width * gridData.height;
	const int z0Slab = z0 * slabsize;
	const int z1Slab = z1 * slabsize;
	const int z2Slab = z2 * slabsize;
	const int z3Slab = z3 * slabsize;

	const int y0x = y0 * gridData.width;
	const int y1x = y1 * gridData.width;
	const int y2x = y2 * gridData.width;
	const int y3x = y3 * gridData.width;

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


*/


#endif // INTERPOLATION_H_