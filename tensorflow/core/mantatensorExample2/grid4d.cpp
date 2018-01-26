




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/grid4d.cpp"
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

#include <limits>
#include <sstream>
#include <cstring>
#include "fileio.h"

#include "grid4d.h"
#include "levelset.h"
#include "kernel.h"

using namespace std;
namespace Manta {


//******************************************************************************
// GridBase members

Grid4dBase::Grid4dBase (FluidSolver* parent) 
	: PbClass(parent), mType(TypeNone)
{
	checkParent();
}


//******************************************************************************
// Grid4d<T> members

// helpers to set type
template<class T> inline Grid4dBase::Grid4dType typeList()        { return Grid4dBase::TypeNone; }
template<>        inline Grid4dBase::Grid4dType typeList<Real>()  { return Grid4dBase::TypeReal; }
template<>        inline Grid4dBase::Grid4dType typeList<int>()   { return Grid4dBase::TypeInt;  }
template<>        inline Grid4dBase::Grid4dType typeList<Vec3>()  { return Grid4dBase::TypeVec3; }
template<>        inline Grid4dBase::Grid4dType typeList<Vec4>()  { return Grid4dBase::TypeVec4; }


template<class T>
Grid4d<T>::Grid4d(FluidSolver* parent, bool show)
	: Grid4dBase(parent)
{
	assertMsg( parent->is3D() && parent->supports4D(), "To use 4d grids create a 3d solver with fourthDim>0");

	mType = typeList<T>();
	Vec3i s = parent->getGridSize();
	mSize = Vec4i(s.x, s.y, s.z, parent->getFourthDim() ); 
	mData = parent->getGrid4dPointer<T>();
	assertMsg( mData, "Couldnt allocate data pointer!");
	
	mStrideZ = (mSize.x * mSize.y);
	mStrideT = (mStrideZ * mSize.z);

	Real sizemax = (Real)mSize[0];
	for(int c=1; c<3; ++c) if(mSize[c]>sizemax) sizemax = mSize[c];
	// note - the 4d component is ignored for dx! keep same scaling as for 3d...
	mDx = 1.0 / sizemax;

	clear();
	setHidden(!show);
}

template<class T>
Grid4d<T>::Grid4d(const Grid4d<T>& a) : Grid4dBase(a.getParent()) {
	mSize = a.mSize;
	mType = a.mType;
	mStrideZ = a.mStrideZ;
	mStrideT = a.mStrideT;
	mDx = a.mDx;
	FluidSolver *gp = a.getParent();
	mData = gp->getGrid4dPointer<T>();
	assertMsg( mData, "Couldnt allocate data pointer!");

	memcpy(mData, a.mData, sizeof(T) * a.mSize.x * a.mSize.y * a.mSize.z * a.mSize.t);
}

template<class T>
Grid4d<T>::~Grid4d() {
	mParent->freeGrid4dPointer<T>(mData);
}

template<class T>
void Grid4d<T>::clear() {
	memset(mData, 0, sizeof(T) * mSize.x * mSize.y * mSize.z * mSize.t);
}

template<class T>
void Grid4d<T>::swap(Grid4d<T>& other) {
	if (other.getSizeX() != getSizeX() || other.getSizeY() != getSizeY() || other.getSizeZ() != getSizeZ() || other.getSizeT() != getSizeT())
		errMsg("Grid4d::swap(): Grid4d dimensions mismatch.");
	
	T* dswap = other.mData;
	other.mData = mData;
	mData = dswap;
}

template<class T>
void Grid4d<T>::load(string name) {
	if (name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if (ext == ".uni")
		readGrid4dUni(name, this);
	else if (ext == ".raw")
		readGrid4dRaw(name, this);
	else
		errMsg("file '" + name +"' filetype not supported");
}

template<class T>
void Grid4d<T>::save(string name) {
	if (name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if (ext == ".uni")
		writeGrid4dUni(name, this);
	else if (ext == ".raw")
		writeGrid4dRaw(name, this);
	else
		errMsg("file '" + name +"' filetype not supported");
}

//******************************************************************************
// Grid4d<T> operators

//! Kernel: Compute min value of Real Grid4d

 struct kn4dMinReal : public KernelBase { kn4dMinReal(Grid4d<Real>& val) :  KernelBase(&val,0) ,val(val) ,minVal(std::numeric_limits<Real>::max())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<Real>& val ,Real& minVal)  {
	if (val[idx] < minVal)
		minVal = val[idx];
}    inline operator Real () { return minVal; } inline Real  & getRet() { return minVal; }  inline Grid4d<Real>& getArg0() { return val; } typedef Grid4d<Real> type0; void runMessage() { debMsg("Executing kernel kn4dMinReal ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  Real minVal = std::numeric_limits<Real>::max(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,minVal); 
#pragma omp critical
{this->minVal = min(minVal, this->minVal); } }   } Grid4d<Real>& val;  Real minVal;  };
#line 137 "grid4d.cpp"



//! Kernel: Compute max value of Real Grid4d

 struct kn4dMaxReal : public KernelBase { kn4dMaxReal(Grid4d<Real>& val) :  KernelBase(&val,0) ,val(val) ,maxVal(-std::numeric_limits<Real>::max())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<Real>& val ,Real& maxVal)  {
	if (val[idx] > maxVal)
		maxVal = val[idx];
}    inline operator Real () { return maxVal; } inline Real  & getRet() { return maxVal; }  inline Grid4d<Real>& getArg0() { return val; } typedef Grid4d<Real> type0; void runMessage() { debMsg("Executing kernel kn4dMaxReal ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  Real maxVal = -std::numeric_limits<Real>::max(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,maxVal); 
#pragma omp critical
{this->maxVal = max(maxVal, this->maxVal); } }   } Grid4d<Real>& val;  Real maxVal;  };
#line 144 "grid4d.cpp"



//! Kernel: Compute min value of int Grid4d

 struct kn4dMinInt : public KernelBase { kn4dMinInt(Grid4d<int>& val) :  KernelBase(&val,0) ,val(val) ,minVal(std::numeric_limits<int>::max())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<int>& val ,int& minVal)  {
	if (val[idx] < minVal)
		minVal = val[idx];
}    inline operator int () { return minVal; } inline int  & getRet() { return minVal; }  inline Grid4d<int>& getArg0() { return val; } typedef Grid4d<int> type0; void runMessage() { debMsg("Executing kernel kn4dMinInt ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  int minVal = std::numeric_limits<int>::max(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,minVal); 
#pragma omp critical
{this->minVal = min(minVal, this->minVal); } }   } Grid4d<int>& val;  int minVal;  };
#line 151 "grid4d.cpp"



//! Kernel: Compute max value of int Grid4d

 struct kn4dMaxInt : public KernelBase { kn4dMaxInt(Grid4d<int>& val) :  KernelBase(&val,0) ,val(val) ,maxVal(std::numeric_limits<int>::min())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<int>& val ,int& maxVal)  {
	if (val[idx] > maxVal)
		maxVal = val[idx];
}    inline operator int () { return maxVal; } inline int  & getRet() { return maxVal; }  inline Grid4d<int>& getArg0() { return val; } typedef Grid4d<int> type0; void runMessage() { debMsg("Executing kernel kn4dMaxInt ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  int maxVal = std::numeric_limits<int>::min(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,maxVal); 
#pragma omp critical
{this->maxVal = max(maxVal, this->maxVal); } }   } Grid4d<int>& val;  int maxVal;  };
#line 158 "grid4d.cpp"



//! Kernel: Compute min norm of vec Grid4d

template <class VEC>  struct kn4dMinVec : public KernelBase { kn4dMinVec(Grid4d<VEC>& val) :  KernelBase(&val,0) ,val(val) ,minVal(std::numeric_limits<Real>::max())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<VEC>& val ,Real& minVal)  {
	const Real s = normSquare(val[idx]);
	if (s < minVal)
		minVal = s;
}    inline operator Real () { return minVal; } inline Real  & getRet() { return minVal; }  inline Grid4d<VEC>& getArg0() { return val; } typedef Grid4d<VEC> type0; void runMessage() { debMsg("Executing kernel kn4dMinVec ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  Real minVal = std::numeric_limits<Real>::max(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,minVal); 
#pragma omp critical
{this->minVal = min(minVal, this->minVal); } }   } Grid4d<VEC>& val;  Real minVal;  };
#line 165 "grid4d.cpp"



//! Kernel: Compute max norm of vec Grid4d

template <class VEC>  struct kn4dMaxVec : public KernelBase { kn4dMaxVec(Grid4d<VEC>& val) :  KernelBase(&val,0) ,val(val) ,maxVal(-std::numeric_limits<Real>::max())  { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<VEC>& val ,Real& maxVal)  {
	const Real s = normSquare(val[idx]);
	if (s > maxVal)
		maxVal = s;
}    inline operator Real () { return maxVal; } inline Real  & getRet() { return maxVal; }  inline Grid4d<VEC>& getArg0() { return val; } typedef Grid4d<VEC> type0; void runMessage() { debMsg("Executing kernel kn4dMaxVec ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  Real maxVal = -std::numeric_limits<Real>::max(); 
#pragma omp for nowait  
  for (IndexInt i = 0; i < _sz; i++) op(i,val,maxVal); 
#pragma omp critical
{this->maxVal = max(maxVal, this->maxVal); } }   } Grid4d<VEC>& val;  Real maxVal;  };
#line 173 "grid4d.cpp"




template<class T> Grid4d<T>& Grid4d<T>::safeDivide (const Grid4d<T>& a) {
	Grid4dSafeDiv<T> (*this, a);
	return *this;
}
template<class T> Grid4d<T>& Grid4d<T>::copyFrom (const Grid4d<T>& a, bool copyType ) {
	assertMsg (a.mSize.x == mSize.x && a.mSize.y == mSize.y && a.mSize.z == mSize.z && a.mSize.t == mSize.t, "different Grid4d resolutions "<<a.mSize<<" vs "<<this->mSize );
	memcpy(mData, a.mData, sizeof(T) * mSize.x * mSize.y * mSize.z * mSize.t);
	if(copyType) mType = a.mType; // copy type marker
	return *this;
}
/*template<class T> Grid4d<T>& Grid4d<T>::operator= (const Grid4d<T>& a) {
	note: do not use , use copyFrom instead
}*/

template <class T>  struct kn4dSetConstReal : public KernelBase { kn4dSetConstReal(Grid4d<T>& me, T val) :  KernelBase(&me,0) ,me(me),val(val)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, T val )  { me[idx]  = val; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline T& getArg1() { return val; } typedef T type1; void runMessage() { debMsg("Executing kernel kn4dSetConstReal ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,val);  }   } Grid4d<T>& me; T val;   };
#line 194 "grid4d.cpp"


template <class T>  struct kn4dAddConstReal : public KernelBase { kn4dAddConstReal(Grid4d<T>& me, T val) :  KernelBase(&me,0) ,me(me),val(val)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, T val )  { me[idx] += val; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline T& getArg1() { return val; } typedef T type1; void runMessage() { debMsg("Executing kernel kn4dAddConstReal ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,val);  }   } Grid4d<T>& me; T val;   };
#line 195 "grid4d.cpp"


template <class T>  struct kn4dMultConst : public KernelBase { kn4dMultConst(Grid4d<T>& me, T val) :  KernelBase(&me,0) ,me(me),val(val)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, T val )  { me[idx] *= val; }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline T& getArg1() { return val; } typedef T type1; void runMessage() { debMsg("Executing kernel kn4dMultConst ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,val);  }   } Grid4d<T>& me; T val;   };
#line 196 "grid4d.cpp"


template <class T>  struct kn4dClamp : public KernelBase { kn4dClamp(Grid4d<T>& me, T min, T max) :  KernelBase(&me,0) ,me(me),min(min),max(max)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid4d<T>& me, T min, T max )  { me[idx] = clamp( me[idx], min, max); }    inline Grid4d<T>& getArg0() { return me; } typedef Grid4d<T> type0;inline T& getArg1() { return min; } typedef T type1;inline T& getArg2() { return max; } typedef T type2; void runMessage() { debMsg("Executing kernel kn4dClamp ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,min,max);  }   } Grid4d<T>& me; T min; T max;   };
#line 197 "grid4d.cpp"



template<class T> void Grid4d<T>::add(const Grid4d<T>& a) {
	Grid4dAdd<T,T>(*this, a);
}
template<class T> void Grid4d<T>::sub(const Grid4d<T>& a) {
	Grid4dSub<T,T>(*this, a);
}
template<class T> void Grid4d<T>::addScaled(const Grid4d<T>& a, const T& factor) { 
	Grid4dScaledAdd<T,T> (*this, a, factor); 
}
template<class T> void Grid4d<T>::setConst(T a) {
	kn4dSetConstReal<T>( *this, T(a) );
}
template<class T> void Grid4d<T>::addConst(T a) {
	kn4dAddConstReal<T>( *this, T(a) );
}
template<class T> void Grid4d<T>::multConst(T a) {
	kn4dMultConst<T>( *this, a );
}

template<class T> void Grid4d<T>::mult(const Grid4d<T>& a) {
	Grid4dMult<T,T> (*this, a);
}

template<class T> void Grid4d<T>::clamp(Real min, Real max) {
	kn4dClamp<T> (*this, T(min), T(max) );
}

template<> Real Grid4d<Real>::getMax() {
	return kn4dMaxReal (*this);
}
template<> Real Grid4d<Real>::getMin() {
	return kn4dMinReal (*this);
}
template<> Real Grid4d<Real>::getMaxAbs() {
	Real amin = kn4dMinReal (*this);
	Real amax = kn4dMaxReal (*this);
	return max( fabs(amin), fabs(amax));
}
template<> Real Grid4d<Vec4>::getMax() {
	return sqrt(kn4dMaxVec<Vec4> (*this));
}
template<> Real Grid4d<Vec4>::getMin() { 
	return sqrt(kn4dMinVec<Vec4> (*this));
}
template<> Real Grid4d<Vec4>::getMaxAbs() {
	return sqrt(kn4dMaxVec<Vec4> (*this));
}
template<> Real Grid4d<int>::getMax() {
	return (Real) kn4dMaxInt (*this);
}
template<> Real Grid4d<int>::getMin() {
	return (Real) kn4dMinInt (*this);
}
template<> Real Grid4d<int>::getMaxAbs() {
	int amin = kn4dMinInt (*this);
	int amax = kn4dMaxInt (*this);
	return max( fabs((Real)amin), fabs((Real)amax));
}
template<> Real Grid4d<Vec3>::getMax() {
	return sqrt(kn4dMaxVec<Vec3> (*this));
}
template<> Real Grid4d<Vec3>::getMin() { 
	return sqrt(kn4dMinVec<Vec3> (*this));
}
template<> Real Grid4d<Vec3>::getMaxAbs() {
	return sqrt(kn4dMaxVec<Vec3> (*this));
}


template<class T> void Grid4d<T>::printGrid(int zSlice, int tSlice, bool printIndex, int bnd) {
	std::ostringstream out;
	out << std::endl;
	FOR_IJKT_BND(*this,bnd) {
		IndexInt idx = (*this).index(i,j,k,t);
		if (  ( (zSlice>=0 && k==zSlice) || (zSlice<0) ) &&
		  	  ( (tSlice>=0 && t==tSlice) || (tSlice<0) ) ) {
			out << " ";
			if(printIndex) out << "  "<<i<<","<<j<<","<<k<<","<<t <<":";
			out << (*this)[idx]; 
			if(i==(*this).getSizeX()-1 -bnd) {
				out << std::endl; 
				if(j==(*this).getSizeY()-1 -bnd) {
					out << std::endl; 
					if(k==(*this).getSizeZ()-1 -bnd) { out << std::endl; }
			} }
		}
	}
	out << endl; debMsg("Printing '" << this->getName() <<"' "<< out.str().c_str()<<" " , 1);
}


// helper to set/get components of vec4 Grids
 struct knGetComp4d : public KernelBase { knGetComp4d(const Grid4d<Vec4>& src, Grid4d<Real>& dst, int c) :  KernelBase(&src,0) ,src(src),dst(dst),c(c)   { runMessage(); run(); }   inline void op(IndexInt idx, const Grid4d<Vec4>& src, Grid4d<Real>& dst, int c )  { dst[idx]    = src[idx][c]; }    inline const Grid4d<Vec4>& getArg0() { return src; } typedef Grid4d<Vec4> type0;inline Grid4d<Real>& getArg1() { return dst; } typedef Grid4d<Real> type1;inline int& getArg2() { return c; } typedef int type2; void runMessage() { debMsg("Executing kernel knGetComp4d ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,src,dst,c);  }   } const Grid4d<Vec4>& src; Grid4d<Real>& dst; int c;   };
#line 291 "grid4d.cpp"

;
 struct knSetComp4d : public KernelBase { knSetComp4d(const Grid4d<Real>& src, Grid4d<Vec4>& dst, int c) :  KernelBase(&src,0) ,src(src),dst(dst),c(c)   { runMessage(); run(); }   inline void op(IndexInt idx, const Grid4d<Real>& src, Grid4d<Vec4>& dst, int c )  { dst[idx][c] = src[idx];    }    inline const Grid4d<Real>& getArg0() { return src; } typedef Grid4d<Real> type0;inline Grid4d<Vec4>& getArg1() { return dst; } typedef Grid4d<Vec4> type1;inline int& getArg2() { return c; } typedef int type2; void runMessage() { debMsg("Executing kernel knSetComp4d ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,src,dst,c);  }   } const Grid4d<Real>& src; Grid4d<Vec4>& dst; int c;   };
#line 292 "grid4d.cpp"

;
void getComp4d(const Grid4d<Vec4>& src, Grid4d<Real>& dst, int c) { knGetComp4d(src,dst,c); };
void setComp4d(const Grid4d<Real>& src, Grid4d<Vec4>& dst, int c) { knSetComp4d(src,dst,c); };


template <class T>  struct knSetBnd4d : public KernelBase { knSetBnd4d(Grid4d<T>& grid, T value, int w) :  KernelBase(&grid,0) ,grid(grid),value(value),w(w)   { runMessage(); run(); }   inline void op(int i, int j, int k, int t, Grid4d<T>& grid, T value, int w )  { 
	bool bnd = 
		(i<=w || i>=grid.getSizeX()-1-w || 
		 j<=w || j>=grid.getSizeY()-1-w || 
		 k<=w || k>=grid.getSizeZ()-1-w ||
		 t<=w || t>=grid.getSizeT()-1-w );
	if (bnd) 
		grid(i,j,k,t) = value;
}    inline Grid4d<T>& getArg0() { return grid; } typedef Grid4d<T> type0;inline T& getArg1() { return value; } typedef T type1;inline int& getArg2() { return w; } typedef int type2; void runMessage() { debMsg("Executing kernel knSetBnd4d ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   " t "<< minT<<" - "<< maxT  , 4); }; void run() {   const int _maxX = maxX; const int _maxY = maxY; if (maxT > 1) { const int _maxZ = maxZ; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int t=0; t < maxT; t++) for (int k=0; k < _maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,value,w);  } } else if (maxZ > 1) { const int t=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,value,w);  } } else { const int t=0; const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,value,w);  } }   } Grid4d<T>& grid; T value; int w;   };
#line 297 "grid4d.cpp"



template<class T> void Grid4d<T>::setBound(T value, int boundaryWidth) {
	knSetBnd4d<T>( *this, value, boundaryWidth );
}

template <class T>  struct knSetBnd4dNeumann : public KernelBase { knSetBnd4dNeumann(Grid4d<T>& grid, int w) :  KernelBase(&grid,0) ,grid(grid),w(w)   { runMessage(); run(); }   inline void op(int i, int j, int k, int t, Grid4d<T>& grid, int w )  { 
	bool set = false;
	int  si=i, sj=j, sk=k, st=t;
	if( i<=w) {
		si = w+1; set=true;
	}
	if( i>=grid.getSizeX()-1-w){
		si = grid.getSizeX()-1-w-1; set=true;
	}
	if( j<=w){
		sj = w+1; set=true;
	}
	if( j>=grid.getSizeY()-1-w){
		sj = grid.getSizeY()-1-w-1; set=true;
	}
	if( k<=w ) {
		sk = w+1; set=true;
	}
	if( k>=grid.getSizeZ()-1-w ) {
		sk = grid.getSizeZ()-1-w-1; set=true;
	}
	if( t<=w ) {
		st = w+1; set=true;
	}
	if( t>=grid.getSizeT()-1-w ) {
		st = grid.getSizeT()-1-w-1; set=true;
	}
	if(set)
		grid(i,j,k,t) = grid(si, sj, sk, st);
}    inline Grid4d<T>& getArg0() { return grid; } typedef Grid4d<T> type0;inline int& getArg1() { return w; } typedef int type1; void runMessage() { debMsg("Executing kernel knSetBnd4dNeumann ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   " t "<< minT<<" - "<< maxT  , 4); }; void run() {   const int _maxX = maxX; const int _maxY = maxY; if (maxT > 1) { const int _maxZ = maxZ; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int t=0; t < maxT; t++) for (int k=0; k < _maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,w);  } } else if (maxZ > 1) { const int t=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,w);  } } else { const int t=0; const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,grid,w);  } }   } Grid4d<T>& grid; int w;   };
#line 311 "grid4d.cpp"



template<class T> void Grid4d<T>::setBoundNeumann(int boundaryWidth) {
	knSetBnd4dNeumann<T>( *this, boundaryWidth );
}

//******************************************************************************
// testing helpers

//! compute maximal diference of two cells in the grid, needed for testing system

Real grid4dMaxDiff(Grid4d<Real>& g1, Grid4d<Real>& g2 ) {
	double maxVal = 0.;
	FOR_IJKT_BND(g1,0) {
		maxVal = std::max(maxVal, (double)fabs( g1(i,j,k,t)-g2(i,j,k,t) ));
	}
	return maxVal; 
}

Real grid4dMaxDiffInt(Grid4d<int>& g1, Grid4d<int>& g2 ) {
	double maxVal = 0.;
	FOR_IJKT_BND(g1,0) {
		maxVal = std::max(maxVal, (double)fabs( (double)g1(i,j,k,t)-g2(i,j,k,t) ));
	}
	return maxVal; 
}

Real grid4dMaxDiffVec3(Grid4d<Vec3>& g1, Grid4d<Vec3>& g2 ) {
	double maxVal = 0.;
	FOR_IJKT_BND(g1,0) {
		double d = 0.;
		for(int c=0; c<3; ++c) { 
			d += fabs( (double)g1(i,j,k,t)[c] - (double)g2(i,j,k,t)[c] );
		}
		maxVal = std::max(maxVal, d );
	}
	return maxVal; 
}

Real grid4dMaxDiffVec4(Grid4d<Vec4>& g1, Grid4d<Vec4>& g2 ) {
	double maxVal = 0.;
	FOR_IJKT_BND(g1,0) {
		double d = 0.;
		for(int c=0; c<4; ++c) { 
			d += fabs( (double)g1(i,j,k,t)[c] - (double)g2(i,j,k,t)[c] );
		}
		maxVal = std::max(maxVal, d );
	}
	return maxVal; 
}

// set a region to some value


template <class S>  struct knSetRegion4d : public KernelBase { knSetRegion4d(Grid4d<S>& dst, Vec4 start, Vec4 end, S value ) :  KernelBase(&dst,0) ,dst(dst),start(start),end(end),value(value)   { runMessage(); run(); }   inline void op(int i, int j, int k, int t, Grid4d<S>& dst, Vec4 start, Vec4 end, S value  )  {
	Vec4 p(i,j,k,t);
	for(int c=0; c<4; ++c) if(p[c]<start[c] || p[c]>end[c]) return;
	dst(i,j,k,t) = value;
}    inline Grid4d<S>& getArg0() { return dst; } typedef Grid4d<S> type0;inline Vec4& getArg1() { return start; } typedef Vec4 type1;inline Vec4& getArg2() { return end; } typedef Vec4 type2;inline S& getArg3() { return value; } typedef S type3; void runMessage() { debMsg("Executing kernel knSetRegion4d ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   " t "<< minT<<" - "<< maxT  , 4); }; void run() {   const int _maxX = maxX; const int _maxY = maxY; if (maxT > 1) { const int _maxZ = maxZ; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int t=0; t < maxT; t++) for (int k=0; k < _maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,dst,start,end,value);  } } else if (maxZ > 1) { const int t=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,dst,start,end,value);  } } else { const int t=0; const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,dst,start,end,value);  } }   } Grid4d<S>& dst; Vec4 start; Vec4 end; S value;   };
#line 394 "grid4d.cpp"


//! simple init functions in 4d
void setRegion4d(Grid4d<Real>& dst, Vec4 start, Vec4 end, Real value) { knSetRegion4d<Real>(dst,start,end,value); }
//! simple init functions in 4d, vec4
void setRegion4dVec4(Grid4d<Vec4>& dst, Vec4 start, Vec4 end, Vec4 value) { knSetRegion4d<Vec4>(dst,start,end,value); }

//! slow helper to visualize tests, get a 3d slice of a 4d grid
void getSliceFrom4d(Grid4d<Real>& src, int srct, Grid<Real>& dst) { 
	const int bnd = 0;
	if(! src.isInBounds(Vec4i(bnd,bnd,bnd,srct)) ) return;

	for(int k=bnd; k<src.getSizeZ()-bnd; k++) 
	for(int j=bnd; j<src.getSizeY()-bnd; j++) 
	for(int i=bnd; i<src.getSizeX()-bnd; i++)
	{
		if(!dst.isInBounds(Vec3i(i,j,k))) continue;
		dst(i,j,k) = src(i,j,k,srct);
	}
}
//! slow helper to visualize tests, get a 3d slice of a 4d vec4 grid
void getSliceFrom4dVec(Grid4d<Vec4>& src, int srct, Grid<Vec3>& dst, Grid<Real>* dstt=NULL) { 
	const int bnd = 0;
	if(! src.isInBounds(Vec4i(bnd,bnd,bnd,srct)) ) return;

	for(int k=bnd; k<src.getSizeZ()-bnd; k++) 
	for(int j=bnd; j<src.getSizeY()-bnd; j++) 
	for(int i=bnd; i<src.getSizeX()-bnd; i++)
	{
		if(!dst.isInBounds(Vec3i(i,j,k))) continue;
		for(int c=0; c<3; ++c) 
			dst(i,j,k)[c] = src(i,j,k,srct)[c];
		if(dstt) (*dstt)(i,j,k) = src(i,j,k,srct)[3];
	}
}


//******************************************************************************
// interpolation

//! same as in grid.h , but takes an additional optional "desired" size
static inline void gridFactor4d(Vec4 s1, Vec4 s2, Vec4 optSize, Vec4 scale, Vec4& srcFac, Vec4& retOff ) {
	for(int c=0; c<4; c++) { if(optSize[c] > 0.){ s2[c] = optSize[c]; } }
	srcFac = calcGridSizeFactor4d( s1, s2) / scale;
	retOff       = -retOff * srcFac + srcFac*0.5;
}

//! interpolate 4d grid from one size to another size
// real valued offsets & scale


template <class S>  struct knInterpol4d : public KernelBase { knInterpol4d(Grid4d<S>& target, Grid4d<S>& source, const Vec4& srcFac, const Vec4& offset) :  KernelBase(&target,0) ,target(target),source(source),srcFac(srcFac),offset(offset)   { runMessage(); run(); }   inline void op(int i, int j, int k, int t, Grid4d<S>& target, Grid4d<S>& source, const Vec4& srcFac, const Vec4& offset )  {
	Vec4 pos = Vec4(i,j,k,t) * srcFac + offset;
	target(i,j,k,t) = source.getInterpolated(pos);
}    inline Grid4d<S>& getArg0() { return target; } typedef Grid4d<S> type0;inline Grid4d<S>& getArg1() { return source; } typedef Grid4d<S> type1;inline const Vec4& getArg2() { return srcFac; } typedef Vec4 type2;inline const Vec4& getArg3() { return offset; } typedef Vec4 type3; void runMessage() { debMsg("Executing kernel knInterpol4d ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   " t "<< minT<<" - "<< maxT  , 4); }; void run() {   const int _maxX = maxX; const int _maxY = maxY; if (maxT > 1) { const int _maxZ = maxZ; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int t=0; t < maxT; t++) for (int k=0; k < _maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,srcFac,offset);  } } else if (maxZ > 1) { const int t=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,srcFac,offset);  } } else { const int t=0; const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,t,target,source,srcFac,offset);  } }   } Grid4d<S>& target; Grid4d<S>& source; const Vec4& srcFac; const Vec4& offset;   };
#line 448 "grid4d.cpp"

 
//! linearly interpolate data of a 4d grid

void interpolateGrid4d( Grid4d<Real>& target, Grid4d<Real>& source , Vec4 offset=Vec4(0.), Vec4 scale=Vec4(1.), Vec4 size=Vec4(-1.) ) {
	Vec4 srcFac(1.), off2 = offset;
	gridFactor4d( toVec4(source.getSize()), toVec4(target.getSize()), size,scale,   srcFac,off2   );
	knInterpol4d<Real> (target, source, srcFac, off2 );
}
//! linearly interpolate vec4 data of a 4d grid

void interpolateGrid4dVec( Grid4d<Vec4>& target, Grid4d<Vec4>& source, Vec4 offset=Vec4(0.), Vec4 scale=Vec4(1.), Vec4 size=Vec4(-1.) ) {
	Vec4 srcFac(1.), off2 = offset;
	gridFactor4d( toVec4(source.getSize()), toVec4(target.getSize()), size,scale,   srcFac,off2   );
	knInterpol4d<Vec4> (target, source, srcFac, off2 );
}


// explicit instantiation
template class Grid4d<int>;
template class Grid4d<Real>;
template class Grid4d<Vec3>;
template class Grid4d<Vec4>;

} //namespace


