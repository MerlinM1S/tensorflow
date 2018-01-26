




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/shapes.h"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * shapes classes
 *
 ******************************************************************************/

#ifndef _SHAPES_H
#define _SHAPES_H

#include "manta.h"
#include "vectorbase.h"
#include "levelset.h"

namespace Manta {

// forward declaration
class Mesh;
	
//! Base class for all shapes

class Shape : public PbClass {public:
	enum GridType { TypeNone = 0, TypeBox = 1, TypeSphere = 2, TypeCylinder = 3, TypeSlope = 4 };
	
	Shape(FluidSolver* parent);
	
	//! Get the type of grid
	inline GridType getType() const { return mType; }
	
	//! Apply shape to flag grid, set inside cells to <value>
	void applyToGrid(GridBase* grid, FlagGrid* respectFlags=0);
	void applyToGridSmooth(GridBase* grid, Real sigma=1.0, Real shift=0, FlagGrid* respectFlags=0);
	LevelsetGrid computeLevelset();
	void collideMesh(Mesh& mesh);
	virtual Vec3 getCenter() const { return Vec3::Zero; }
	virtual void setCenter(const Vec3& center) {}
	virtual Vec3 getExtent() const { return Vec3::Zero; }
	
	//! Inside test of the shape
	virtual bool isInside(const Vec3& pos) const;
	inline bool isInsideGrid(int i, int j, int k) const { return isInside(Vec3(i+0.5,j+0.5,k+0.5)); };
	
	virtual void generateMesh(Mesh* mesh) {} ;    
	virtual void generateLevelset(Grid<Real>& phi) {};    
	
protected: 	GridType mType; }
;

//! Dummy shape

class NullShape : public Shape {   
public:
	NullShape(FluidSolver* parent) :Shape(parent){}
	
	virtual bool isInside(const Vec3& pos) const { return false; }
	virtual void generateMesh(Mesh* mesh) {}
	
protected: 	virtual void generateLevelset(Grid<Real>& phi) { gridSetConst<Real>( phi , 1000.0f ); } }
;

//! Box shape

class Box : public Shape {   
public:
	Box(FluidSolver* parent, Vec3 center = Vec3::Invalid, Vec3 p0 = Vec3::Invalid, Vec3 p1 = Vec3::Invalid, Vec3 size = Vec3::Invalid);
	
	inline Vec3 getSize() const { return mP1-mP0; }
	inline Vec3 getP0() const { return mP0; }
	inline Vec3 getP1() const { return mP1; }
	virtual void setCenter(const Vec3& center) { Vec3 dh=0.5*(mP1-mP0); mP0 = center-dh; mP1 = center+dh;}
	virtual Vec3 getCenter() const { return 0.5*(mP1+mP0); }
	virtual Vec3 getExtent() const { return getSize(); }
	virtual bool isInside(const Vec3& pos) const;
	virtual void generateMesh(Mesh* mesh);
	virtual void generateLevelset(Grid<Real>& phi);
	
protected: 	Vec3 mP0, mP1; }
;

//! Spherical shape

class Sphere : public Shape {   
public:
	Sphere(FluidSolver* parent, Vec3 center, Real radius, Vec3 scale=Vec3(1,1,1));
	
	virtual void setCenter(const Vec3& center) { mCenter = center; }
	virtual Vec3 getCenter() const { return mCenter; }
	inline Real getRadius() const { return mRadius; }
	virtual Vec3 getExtent() const { return Vec3(2.0*mRadius); }    
	virtual bool isInside(const Vec3& pos) const;
	virtual void generateMesh(Mesh* mesh);
	virtual void generateLevelset(Grid<Real>& phi);
	
protected:
	Vec3 mCenter, mScale; 	Real mRadius; }
;

//! Cylindrical shape

class Cylinder : public Shape {   
public:
	Cylinder(FluidSolver* parent, Vec3 center, Real radius, Vec3 z);
	
	void setRadius(Real r) { mRadius = r; }
	void setZ(Vec3 z) { mZDir=z; mZ=normalize(mZDir); }
	
	virtual void setCenter(const Vec3& center) { mCenter=center; }
	virtual Vec3 getCenter() const { return mCenter; }
	inline Real getRadius() const { return mRadius; }
	inline Vec3 getZ() const { return mZ*mZDir; }
	virtual Vec3 getExtent() const { return Vec3(2.0*sqrt(square(mZ)+square(mRadius))); }    
	virtual bool isInside(const Vec3& pos) const;
	virtual void generateMesh(Mesh* mesh);
	virtual void generateLevelset(Grid<Real>& phi);

protected:
	Vec3 mCenter, mZDir; 	Real mRadius, mZ; }
;

//! Slope shape
// generates a levelset based on a plane
// plane is specified by two angles and an offset on the y axis in (offset vector would be ( 0, offset, 0) )
// the two angles are specified in degrees, between: y-axis and x-axis 
//                                                   y-axis and z-axis

class Slope : public Shape {public:
	Slope(FluidSolver* parent, Real anglexy, Real angleyz, Real origin, Vec3 gs);

	virtual void setOrigin (const Real& origin)  { mOrigin=origin; }
	virtual void setAnglexy(const Real& anglexy) { mAnglexy=anglexy; }
	virtual void setAngleyz(const Real& angleyz) { mAnglexy=angleyz; }

	inline Real getOrigin()   const { return mOrigin; }
	inline Real getmAnglexy() const { return mAnglexy; }
	inline Real getmAngleyz() const { return mAngleyz; }
	virtual bool isInside(const Vec3& pos) const;
	virtual void generateMesh(Mesh* mesh);
	virtual void generateLevelset(Grid<Real>& phi);

protected:
	Real mAnglexy, mAngleyz;
	Real mOrigin; 	Vec3 mGs; }
;

} //namespace
#endif


