load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "simple_kernel_example.so",
    srcs = ["simple_kernel_example.cc", "simple_kernel_example.h"],
    gpu_srcs = ["simple_kernel_example.cu.cc", "simple_kernel_example.h"],
)
