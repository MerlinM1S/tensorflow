#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fluid_grid.h"

#ifndef ADVECTION_H_
#define ADVECTION_H_

using namespace tensorflow;

template <typename Device>
struct Advection {
    void advect1D(const Device& d, const FluidGrid* fluidGrid, const float dt, const float* in_grid, float* out_grid, const int orderSpace);
    void advectMAC(const Device& d, const FluidGrid* fluidGrid, const float dt, const float* in_grid, float* out_grid, const int orderSpace);
};



#endif // ADVECTION_H_