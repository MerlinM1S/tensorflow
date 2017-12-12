load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "add_buoyancy.so",
    srcs = ["add_buoyancy.cc", "fluid_grid.h", "fluid_grid_functor.h", "add_buoyancy.h"],
    gpu_srcs = ["add_buoyancy.cu.cc", "fluid_grid.h", "fluid_grid_functor.cu.h", "add_buoyancy.h"],
)

tf_custom_op_library(
    name = "advection.so",
    srcs = ["advection.cc", "fluid_grid.h", "fluid_grid_functor.h"],
)