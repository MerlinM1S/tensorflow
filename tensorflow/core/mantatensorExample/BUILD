load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "auto_buo.so",
    srcs = ["dim_size.h",
    
        "MantaCPU/general.h", "MantaCPU/grid.h", "MantaCPU/manta.h", "MantaCPU/vectorbase.h", "MantaCPU/pclass.h", "MantaCPU/fluidsolver.h", "MantaCPU/vector4d.h", "MantaCPU/interpol.h",
        "MantaCPU/interpolHigh.h", "MantaCPU/kernel.h", "MantaCPU/commonkernels.h", "MantaCPU/particle.h", "MantaCPU/integrator.h", "MantaCPU/randomstream.h", "MantaCPU/grid4d.h", "MantaCPU/gitinfo.h",
        "MantaCPU/levelset.h", "MantaCPU/fileio.h", "MantaCPU/mesh.h", "MantaCPU/fastmarch.h", "MantaCPU/vortexsheet.h", "MantaCPU/mcubes.h", "MantaCPU/solvana.h", "MantaCPU/shapes.h",
        "MantaCPU/noisefield.h",
        
        "MantaCPU/grid.cpp", "MantaCPU/general.cpp", "MantaCPU/vectorbase.cpp", "MantaCPU/pclass.cpp", "MantaCPU/fluidsolver.cpp", "MantaCPU/vector4d.cpp", 
        "MantaCPU/kernel.cpp", "MantaCPU/particle.cpp", "MantaCPU/grid4d.cpp",         
        "MantaCPU/levelset.cpp", "MantaCPU/fileio.cpp", "MantaCPU/mesh.cpp", "MantaCPU/fastmarch.cpp", "MantaCPU/vortexsheet.cpp", "MantaCPU/shapes.cpp",
        "MantaCPU/noisefield.cpp",
        
        "MantaCPU/extforces.h",
        "MantaCPU/extforces.cpp",
        
        "auto_buo.cc"],
)
