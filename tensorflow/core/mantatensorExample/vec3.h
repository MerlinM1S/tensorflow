#ifndef VEC_3_H_
#define VEC_3_H_

struct Vec3 {
    float x;
    float y;
    float z;
    
    Vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};


//! Multiplication operator
inline Vec3 operator* ( const Vec3& v, float s ) {
	return Vec3 ( v.x*s, v.y*s, v.z*s );
}

//! Subtraction operator
inline Vec3 operator- ( const Vec3 &v1, const Vec3 &v2 ) {
	return Vec3 ( v1.x-v2.x, v1.y-v2.y, v1.z-v2.z );
}

#endif // VEC_3_H_