import tensorflow as tf
import numpy as np

zero_out_module = tf.load_op_library('/home/ansorge/tensorflow/bazel-bin/tensorflow/core/mantatensorExample/auto_buo.so')

with tf.Session(''):
    batches = 1
    width = 4
    height = 6
    depth = 4
    
    vel = np.ones((batches, width, height, depth, 3), dtype=np.float32)
    flags = np.ones((batches, width, height, depth, 1), dtype=np.int32)
    density = np.ones((batches, width, height, depth, 1), dtype=np.float32) 
    
    force = np.array([0, -0.2, 0])
    coeffient = np.array([1])
    
    
    print density
    
    print vel
    
    print ""
    print "Result"
    
    
    vel = zero_out_module.auto_buo(flags, density, vel, force, coeffient).eval()

    print vel
