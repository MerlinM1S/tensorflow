




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/grid4d.h"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Grid representation
 *
 ******************************************************************************/

#ifndef _GRID4D_H
#define _GRID4D_H

#include "manta.h"
#include "vectorbase.h"
#include "vector4d.h"
#include "kernel.h"


namespace Manta {

	
//! Base class for all grids

class Grid4dBase : public PbClass {public:
	enum Grid4dType { TypeNone = 0, TypeReal = 1, TypeInt = 2, TypeVec3 = 4, TypeVec4 = 8 };
		
	Grid4dBase(FluidSolver* parent);
	
	//! Get the grids X dimension
	inline int getSizeX() const { return mSize.x; }
	//! Get the grids Y dimension
	inline int getSizeY() const { return mSize.y; }
	//! Get the grids Z dimension
	inline int getSizeZ() const { return mSize.z; }
	//! Get the grids T dimension
	inline int getSizeT() const { return mSize.t; }
	//! Get the grids dimensions
	inline Vec4i getSize() const { return mSize; }
	
	//! Get Stride in X dimension
	inline IndexInt getStrideX() const { return 1; }
	//! Get Stride in Y dimension
	inline IndexInt getStrideY() const { return mSize.x; }
	//! Get Stride in Z dimension
	inline IndexInt getStrideZ() const { return mStrideZ; }
	//! Get Stride in T dimension
	inline IndexInt getStrideT() const { return mStrideT; }
	
	inline Real getDx() { return mDx; }
	
	//! Check if indices are within bounds, otherwise error (should only be called when debugging)
	inline void checkIndex(int i, int j, int k, int t) const;
	//! Check if indices are within bounds, otherwise error (should only be called when debugging)
	inline void checkIndex(IndexInt idx) const;
	//! Check if index is within given boundaries
	inline bool isInBounds(const Vec4i& p, int bnd) const;
	//! Check if index is within given boundaries
	inline bool isInBounds(const Vec4i& p) const;
	//! Check if index is within given boundaries
	inline bool isInBounds(const Vec4& p, int bnd = 0) const { return isInBounds(toVec4i(p), bnd); }
	//! Check if linear index is in the range of the array
	inline bool isInBounds(IndexInt idx) const;
	
	//! Get the type of grid
	inline Grid4dType getType() const { return mType; }
	//! Check dimensionality
	inline bool is3D() const { return true; }
	inline bool is4D() const { return true; }

	//! 3d compatibility
	inline bool isInBounds(int i,int j, int k, int t, int bnd) const { return isInBounds( Vec4i(i,j,k,t), bnd ); }
	
	//! Get index into the data
	inline IndexInt index(int i, int j, int k, int t) const { DEBUG_ONLY(checkIndex(i,j,k,t)); return (IndexInt)i + (IndexInt)mSize.x * j + (IndexInt)mStrideZ * k + (IndexInt)mStrideT * t; }
	//! Get index into the data
	inline IndexInt index(const Vec4i& pos) const    { DEBUG_ONLY(checkIndex(pos.x,pos.y,pos.z,pos.t)); return (IndexInt)pos.x + (IndexInt)mSize.x * pos.y + (IndexInt)mStrideZ * pos.z + (IndexInt)mStrideT * pos.t; }
protected:
	
	Grid4dType mType;
	Vec4i      mSize;
	Real       mDx;
	// precomputed Z,T shift: to ensure 2D compatibility, always use this instead of sx*sy !
	IndexInt   mStrideZ;  	IndexInt   mStrideT;  }
;

//! Grid class


template<class T> class Grid4d : public Grid4dBase {public:
	//! init new grid, values are set to zero
	Grid4d(FluidSolver* parent, bool show = true);
	//! create new & copy content from another grid
	Grid4d(const Grid4d<T>& a);
	//! return memory to solver
	virtual ~Grid4d();
	
	typedef T BASETYPE;
	typedef Grid4dBase BASETYPE_GRID;
	
	void save(std::string name);
	void load(std::string name);
	
	//! set all cells to zero
	void clear();
	
	//! all kinds of access functions, use grid(), grid[] or grid.get()
	//! access data
	inline T get(int i,int j, int k, int t) const         { return mData[index(i,j,k,t)]; }
	//! access data
	inline T& get(int i,int j, int k, int t)              { return mData[index(i,j,k,t)]; }
	//! access data
	inline T get(IndexInt idx) const                           { DEBUG_ONLY(checkIndex(idx)); return mData[idx]; }
	//! access data
	inline T get(const Vec4i& pos) const                  { return mData[index(pos)]; }
	//! access data
	inline T& operator()(int i, int j, int k, int t)      { return mData[index(i, j, k,t)]; }
	//! access data
	inline T operator()(int i, int j, int k, int t) const { return mData[index(i, j, k,t)]; }
	//! access data
	inline T& operator()(IndexInt idx)                  { DEBUG_ONLY(checkIndex(idx)); return mData[idx]; }
	//! access data
	inline T operator()(IndexInt idx) const             { DEBUG_ONLY(checkIndex(idx)); return mData[idx]; }
	//! access data
	inline T& operator()(const Vec4i& pos)         { return mData[index(pos)]; }
	//! access data
	inline T operator()(const Vec4i& pos) const    { return mData[index(pos)]; }
	//! access data
	inline T& operator[](IndexInt idx)                  { DEBUG_ONLY(checkIndex(idx)); return mData[idx]; }
	//! access data
	inline const T operator[](IndexInt idx) const       { DEBUG_ONLY(checkIndex(idx)); return mData[idx]; }
	
	// interpolated access
	inline T    getInterpolated(const Vec4& pos) const { return interpol4d<T>(mData, mSize, mStrideZ, mStrideT, pos); }
	
	// assignment / copy

	//! warning - do not use "=" for grids in python, this copies the reference! not the grid content...
	//Grid4d<T>& operator=(const Grid4d<T>& a);
	//! copy content from other grid (use this one instead of operator= !)
	Grid4d<T>& copyFrom(const Grid4d<T>& a, bool copyType=true ); // old: { *this = a; }

	// helper functions to work with grids in scene files 

	//! add/subtract other grid
	void add(const Grid4d<T>& a);
	void sub(const Grid4d<T>& a);
	//! set all cells to constant value
	void setConst(T s);
	//! add constant to all grid cells
	void addConst(T s);
	//! add scaled other grid to current one (note, only "Real" factor, "T" type not supported here!)
	void addScaled(const Grid4d<T>& a, const T& factor); 
	//! multiply contents of grid
	void mult( const Grid4d<T>& a);
	//! multiply each cell by a constant scalar value
	void multConst(T s);
	//! clamp content to range (for vec3, clamps each component separately)
	void clamp(Real min, Real max);
	
	// common compound operators
	//! get absolute max value in grid 
	Real getMaxAbs();
	//! get max value in grid 
	Real getMax();
	//! get min value in grid 
	Real getMin();    
	//! set all boundary cells to constant value (Dirichlet)
	void setBound(T value, int boundaryWidth=1);
	//! set all boundary cells to last inner value (Neumann)
	void setBoundNeumann(int boundaryWidth=1);

	//! debugging helper, print grid from Python
	void printGrid(int zSlice=-1, int tSlice=-1, bool printIndex=false, int bnd=0); 

	// c++ only operators
	template<class S> Grid4d<T>& operator+=(const Grid4d<S>& a);
	template<class S> Grid4d<T>& operator+=(const S& a);
	template<class S> Grid4d<T>& operator-=(const Grid4d<S>& a);
	template<class S> Grid4d<T>& operator-=(const S& a);
	template<class S> Grid4d<T>& operator*=(const Grid4d<S>& a);
	template<class S> Grid4d<T>& operator*=(const S& a);
	template<class S> Grid4d<T>& operator/=(const Grid4d<S>& a);
	template<class S> Grid4d<T>& operator/=(const S& a);
	Grid4d<T>& safeDivide(const Grid4d<T>& a);    
	
	//! Swap data with another grid (no actual data is moved)
	void swap(Grid4d<T>& other);

protected: 	T* mData; }
;

// Python doesn't know about templates: explicit aliases needed






//! helper to compute grid conversion factor between local coordinates of two grids
inline Vec4 calcGridSizeFactor4d(Vec4i s1, Vec4i s2) {
	return Vec4( Real(s1[0])/s2[0], Real(s1[1])/s2[1], Real(s1[2])/s2[2] , Real(s1[3])/s2[3] );
}
inline Vec4 calcGridSizeFactor4d(Vec4 s1, Vec4 s2) {
	return Vec4( s1[0]/s2[0], s1[1]/s2[1], s1[2]/s2[2] , s1[3]/s2[3] );
}

// prototypes for grid plugins
void getComponent4d(const Grid4d<Vec4>& src, Grid4d<Real>& dst, int c);
void setComponent4d(const Grid4d<Real>& src, Grid4d<Vec4>& dst, int c);


//******************************************************************************
// Implementation of inline functions

inline void Grid4dBase::checkIndex(int i, int j, int k, int t) const {
	if ( i<0 || j<0  || i>=mSize.x || j>=mSize.y || k<0|| k>= mSize.z ||
         t<0|| t>= mSize.t ) {
		std::ostringstream s;
		s << "Grid4d " << mName << " dim " << mSize << " : index " << i << "," << j << "," << k << ","<<t<<" out of bound ";
		errMsg(s.str());
	}
}

inline void Grid4dBase::checkIndex(IndexInt idx) const {
	if (idx<0 || idx >= mSize.x * mSize.y * mSize.z * mSize.t) {
		std::ostringstream s;
		s << "Grid4d " << mName << " dim " << mSize << " : index " << idx << " out of bound ";
		errMsg(s.str());
	}
}

bool Grid4dBase::isInBounds(const Vec4i& p) const { 
	return (p.x >= 0 && p.y >= 0 && p.z >= 0 && p.t >= 0 && 
			p.x < mSize.x && p.y < mSize.y && p.z < mSize.z && p.t < mSize.t); 
}

bool Grid4dBase::isInBounds(const Vec4i& p, int bnd) const { 
	bool ret = (p.x >= bnd && p.y >= bnd && p.x < mSize.x-bnd && p.y < mSize.y-bnd);
	ret &= (p.z >= bnd && p.z < mSize.z-bnd); 
	ret &= (p.t >= bnd && p.t < mSize.t-bnd); 
	return ret;
}
//! Check if linear index is in the range of the array
bool Grid4dBase::isInBounds(IndexInt idx) const {
	if (idx<0 || idx >= mSize.x * mSize.y * mSize.z * mSize.t) {
		return false;
	}
	return true;
}

// note - ugly, mostly copied from normal GRID!

template <class T, class S>  struct Grid4dAdd : public KernelBase { Grid4dAdd(Grid4d<T>& me, const Grid4d<S>& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<S>& other )  { me[idx] += other[idx]; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<S>& getArg1() { return other; } typedef Grid4d<S> type1; void runMessage() { debMsg("Executing kernel Grid4dAdd ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const Grid4d<S>& other;   };
#line 259 "grid4d.h"


template <class T, class S>  struct Grid4dSub : public KernelBase { Grid4dSub(Grid4d<T>& me, const Grid4d<S>& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<S>& other )  { me[idx] -= other[idx]; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<S>& getArg1() { return other; } typedef Grid4d<S> type1; void runMessage() { debMsg("Executing kernel Grid4dSub ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const Grid4d<S>& other;   };
#line 260 "grid4d.h"


template <class T, class S>  struct Grid4dMult : public KernelBase { Grid4dMult(Grid4d<T>& me, const Grid4d<S>& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<S>& other )  { me[idx] *= other[idx]; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<S>& getArg1() { return other; } typedef Grid4d<S> type1; void runMessage() { debMsg("Executing kernel Grid4dMult ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const Grid4d<S>& other;   };
#line 261 "grid4d.h"


template <class T, class S>  struct Grid4dDiv : public KernelBase { Grid4dDiv(Grid4d<T>& me, const Grid4d<S>& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<S>& other )  { me[idx] /= other[idx]; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<S>& getArg1() { return other; } typedef Grid4d<S> type1; void runMessage() { debMsg("Executing kernel Grid4dDiv ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const Grid4d<S>& other;   };
#line 262 "grid4d.h"


template <class T, class S>  struct Grid4dAddScalar : public KernelBase { Grid4dAddScalar(Grid4d<T>& me, const S& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const S& other )  { me[idx] += other; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const S& getArg1() { return other; } typedef S type1; void runMessage() { debMsg("Executing kernel Grid4dAddScalar ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const S& other;   };
#line 263 "grid4d.h"


template <class T, class S>  struct Grid4dMultScalar : public KernelBase { Grid4dMultScalar(Grid4d<T>& me, const S& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const S& other )  { me[idx] *= other; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const S& getArg1() { return other; } typedef S type1; void runMessage() { debMsg("Executing kernel Grid4dMultScalar ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const S& other;   };
#line 264 "grid4d.h"


template <class T, class S>  struct Grid4dScaledAdd : public KernelBase { Grid4dScaledAdd(Grid4d<T>& me, const Grid4d<T>& other, const S& factor) :  KernelBase(&me,0) ,me(me),other(other),factor(factor)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<T>& other, const S& factor )  { me[idx] += factor * other[idx]; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<T>& getArg1() { return other; } typedef Grid4d<T> type1;inline const S& getArg2() { return factor; } typedef S type2; void runMessage() { debMsg("Executing kernel Grid4dScaledAdd ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other,factor);  }   } Grid4d<T>& me; const Grid4d<T>& other; const S& factor;   };
#line 265 "grid4d.h"



template <class T>  struct Grid4dSafeDiv : public KernelBase { Grid4dSafeDiv(Grid4d<T>& me, const Grid4d<T>& other) :  KernelBase(&me,0) ,me(me),other(other)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, const Grid4d<T>& other )  { me[idx] = safeDivide(me[idx], other[idx]); }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline const Grid4d<T>& getArg1() { return other; } typedef Grid4d<T> type1; void runMessage() { debMsg("Executing kernel Grid4dSafeDiv ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other);  }   } Grid4d<T>& me; const Grid4d<T>& other;   };
#line 267 "grid4d.h"


template <class T>  struct Grid4dSetConst : public KernelBase { Grid4dSetConst(Grid4d<T>& me, T value) :  KernelBase(&me,0) ,me(me),value(value)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, T value )  { me[idx] = value; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline T& getArg1() { return value; } typedef T type1; void runMessage() { debMsg("Executing kernel Grid4dSetConst ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,value);  }   } Grid4d<T>& me; T value;   };
#line 268 "grid4d.h"



template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator+= (const Grid4d<S>& a) {
	Grid4dAdd<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator+= (const S& a) {
	Grid4dAddScalar<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator-= (const Grid4d<S>& a) {
	Grid4dSub<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator-= (const S& a) {
	Grid4dAddScalar<T,S> (*this, -a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator*= (const Grid4d<S>& a) {
	Grid4dMult<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator*= (const S& a) {
	Grid4dMultScalar<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator/= (const Grid4d<S>& a) {
	Grid4dDiv<T,S> (*this, a);
	return *this;
}
template<class T> template<class S> Grid4d<T>& Grid4d<T>::operator/= (const S& a) {
	S rez((S)1.0 / a);
	Grid4dMultScalar<T,S> (*this, rez);
	return *this;
}


//******************************************************************************
// Other helper functions

inline Vec4 getGradient4d(const Grid4d<Real>& data, int i, int j, int k, int t) {
	Vec4 v;
	if (i > data.getSizeX()-2) i= data.getSizeX()-2;
	if (j > data.getSizeY()-2) j= data.getSizeY()-2;
	if (k > data.getSizeZ()-2) k= data.getSizeZ()-2;
	if (t > data.getSizeT()-2) t= data.getSizeT()-2;
	if (i < 1) i = 1;
	if (j < 1) j = 1;
	if (k < 1) k = 1;
	if (t < 1) t = 1;
	v = Vec4( data(i+1,j  ,k  ,t  ) - data(i-1,j  ,k  ,t  ) ,
			  data(i  ,j+1,k  ,t  ) - data(i  ,j-1,k  ,t  ) , 
			  data(i  ,j  ,k+1,t  ) - data(i  ,j  ,k-1,t  ) , 
			  data(i  ,j  ,k  ,t+1) - data(i  ,j  ,k  ,t-1) );
	return v;
}



template <class S>  struct KnInterpolateGrid4dTempl : public KernelBase { KnInterpolateGrid4dTempl(Grid4d<S>& target, Grid4d<S>& source, const Vec4& sourceFactor , Vec4 offset) :  KernelBase(&target,0) ,target(target),source(source),sourceFactor(sourceFactor),offset(offset)   { runMessage(); run(); }   inline void op(int i, int j, int k, int t, Grid4d<S>& target, Grid4d<S>& source, const Vec4& sourceFactor , Vec4 offset )  {
	Vec4 pos = Vec4(i,j,k,t) * sourceFactor + offset;
	if(!source.is3D()) pos[2] = 0.; // allow 2d -> 3d
	if(!source.is4D()) pos[3] = 0.; // allow 3d -> 4d
	target(i,j,k,t) = source.getInterpolated(pos);
}    inline Grid4d<S>& getArg0() { return target; } typedef Grid4d<S> type0;inline Grid4d<S>& getArg1() { return source; } typedef Grid4d<S> type1;inline const Vec4& getArg2() { return sourceFactor; } typedef Vec4 type2;inline Vec4& getArg3() { return offset; } typedef Vec4 type3; void runMessage() { debMsg("Executing kernel KnInterpolateGrid4dTempl ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   " t "<< minT<<" - "<< maxT  , 4); }; void run() {   const int _maxX = maxX; const int _maxY = maxY; if (maxT > 1) { const int _maxZ = maxZ; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int t=0; t < maxT; t++) for (int k=0; k < _maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,sourceFactor,offset);  } } else if (maxZ > 1) { const int t=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,sourceFactor,offset);  } } else { const int t=0; const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,sourceFactor,offset);  } }   } Grid4d<S>& target; Grid4d<S>& source; const Vec4& sourceFactor; Vec4 offset;   };
#line 327 "grid4d.h"

 

} //namespace
#endif


