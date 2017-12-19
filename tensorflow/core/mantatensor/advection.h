#ifndef ADVECTION_H_
#define ADVECTION_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fluid_grid.h"

using namespace tensorflow;

enum AdvectionType {
    AdvectionType1D,
    AdvectionTypeMAC
};

template <typename Device>
struct Advection {
    void operator()(const Device& d, const FluidGrid* fluidGrid, const float dt, const float* in_grid, float* out_grid, const int orderSpace, const AdvectionType advectionType );
};



#endif // ADVECTION_H_