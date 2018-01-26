#ifndef ADD_BUOYANCY_H_
#define ADD_BUOYANCY_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fluid_grid.h"

using namespace tensorflow;

template <typename Device>
struct AddBuoyancy {
  void operator()(const Device& d, const FluidGrid* fluidGrid, const float* force, float* out_vel);
};

#endif // ADD_BUOYANCY_H_
