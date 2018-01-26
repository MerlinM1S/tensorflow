load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "auto_buo.so",
    srcs = ["dim_size.h",
    
        "general.h", "grid.h", "manta.h", "vectorbase.h", "pclass.h", "fluidsolver.h", "vector4d.h", "interpol.h",
        "interpolHigh.h", "kernel.h", "commonkernels.h", "particle.h", "integrator.h", "randomstream.h", "grid4d.h", "gitinfo.h",
        "levelset.h", "fileio.h", "mesh.h", "fastmarch.h", "vortexsheet.h", "mcubes.h", "solvana.h", "shapes.h",
        "noisefield.h",
        
        "grid.cpp", "general.cpp", "vectorbase.cpp", "pclass.cpp", "fluidsolver.cpp", "vector4d.cpp", 
        "kernel.cpp", "particle.cpp", "grid4d.cpp",         
        "levelset.cpp", "fileio.cpp", "mesh.cpp", "fastmarch.cpp", "vortexsheet.cpp", "shapes.cpp",
        "noisefield.cpp",
        
        "extforces.cpp"],
)
