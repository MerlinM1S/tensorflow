#ifndef CUDA_VEC_3_H_
#define CUDA_VEC_3_H_

struct CudaVec3 {
    float x;
    float y;
    float z;
    
    __device__ CudaVec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};


//! Multiplication operator
__device__ CudaVec3 operator* ( const CudaVec3& v, float s ) {
	return CudaVec3 ( v.x*s, v.y*s, v.z*s );
}

//! Subtraction operator
__device__ CudaVec3 operator- ( const CudaVec3 &v1, const CudaVec3 &v2 ) {
	return CudaVec3 ( v1.x-v2.x, v1.y-v2.y, v1.z-v2.z );
}

#endif // CUDA_VEC_3_H_
