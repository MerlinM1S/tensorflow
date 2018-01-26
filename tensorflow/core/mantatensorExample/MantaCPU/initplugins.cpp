




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/plugin/initplugins.cpp"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Tools to setup fields and inflows
 *
 ******************************************************************************/

#include "vectorbase.h"
#include "shapes.h"
#include "commonkernels.h"
#include "particle.h"
#include "noisefield.h"
#include "simpleimage.h"
#include "mesh.h"

using namespace std;

namespace Manta {
	
//! Apply noise to grid


 struct KnApplyNoiseInfl : public KernelBase { KnApplyNoiseInfl(const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, const Grid<Real>& sdf, Real scale, Real sigma) :  KernelBase(&flags,0) ,flags(flags),density(density),noise(noise),sdf(sdf),scale(scale),sigma(sigma)   { runMessage(); run(); }  inline void op(int i, int j, int k, const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, const Grid<Real>& sdf, Real scale, Real sigma )  {
	if (!flags.isFluid(i,j,k) || sdf(i,j,k) > sigma) return;
	Real factor = clamp(1.0-0.5/sigma * (sdf(i,j,k)+sigma), 0.0, 1.0);
	
	Real target = noise.evaluate(Vec3(i,j,k)) * scale * factor;
	if (density(i,j,k) < target)
		density(i,j,k) = target;
}   inline const FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline Grid<Real>& getArg1() { return density; } typedef Grid<Real> type1;inline WaveletNoiseField& getArg2() { return noise; } typedef WaveletNoiseField type2;inline const Grid<Real>& getArg3() { return sdf; } typedef Grid<Real> type3;inline Real& getArg4() { return scale; } typedef Real type4;inline Real& getArg5() { return sigma; } typedef Real type5; void runMessage() { debMsg("Executing kernel KnApplyNoiseInfl ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,noise,sdf,scale,sigma);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,noise,sdf,scale,sigma);  } }  } const FlagGrid& flags; Grid<Real>& density; WaveletNoiseField& noise; const Grid<Real>& sdf; Real scale; Real sigma;   };
#line 29 "plugin/initplugins.cpp"



//! Init noise-modulated density inside shape

void densityInflow(const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, Shape* shape, Real scale=1.0, Real sigma=0) {
	Grid<Real> sdf = shape->computeLevelset();
	KnApplyNoiseInfl(flags, density, noise, sdf, scale, sigma);
}
//! Apply noise to real grid based on an SDF
 struct KnAddNoise : public KernelBase { KnAddNoise(const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, const Grid<Real>* sdf, Real scale) :  KernelBase(&flags,0) ,flags(flags),density(density),noise(noise),sdf(sdf),scale(scale)   { runMessage(); run(); }  inline void op(int i, int j, int k, const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, const Grid<Real>* sdf, Real scale )  {
	if (!flags.isFluid(i,j,k) || (sdf && (*sdf)(i,j,k) > 0.) ) return;
	density(i,j,k) += noise.evaluate(Vec3(i,j,k)) * scale;
}   inline const FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline Grid<Real>& getArg1() { return density; } typedef Grid<Real> type1;inline WaveletNoiseField& getArg2() { return noise; } typedef WaveletNoiseField type2;inline const Grid<Real>* getArg3() { return sdf; } typedef Grid<Real> type3;inline Real& getArg4() { return scale; } typedef Real type4; void runMessage() { debMsg("Executing kernel KnAddNoise ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,noise,sdf,scale);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,noise,sdf,scale);  } }  } const FlagGrid& flags; Grid<Real>& density; WaveletNoiseField& noise; const Grid<Real>* sdf; Real scale;   };
#line 45 "plugin/initplugins.cpp"


void addNoise(const FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, const Grid<Real>* sdf=NULL, Real scale=1.0 ) {
	KnAddNoise(flags, density, noise, sdf, scale );
}

//! sample noise field and set pdata with its values (for convenience, scale the noise values)

template <class T>  struct knSetPdataNoise : public KernelBase { knSetPdataNoise(BasicParticleSystem& parts, ParticleDataImpl<T>& pdata, WaveletNoiseField& noise, Real scale) :  KernelBase(parts.size()) ,parts(parts),pdata(pdata),noise(noise),scale(scale)   { runMessage(); run(); }   inline void op(IndexInt idx, BasicParticleSystem& parts, ParticleDataImpl<T>& pdata, WaveletNoiseField& noise, Real scale )  {
	pdata[idx] = noise.evaluate( parts.getPos(idx) ) * scale;
}    inline BasicParticleSystem& getArg0() { return parts; } typedef BasicParticleSystem type0;inline ParticleDataImpl<T>& getArg1() { return pdata; } typedef ParticleDataImpl<T> type1;inline WaveletNoiseField& getArg2() { return noise; } typedef WaveletNoiseField type2;inline Real& getArg3() { return scale; } typedef Real type3; void runMessage() { debMsg("Executing kernel knSetPdataNoise ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,parts,pdata,noise,scale);  }   } BasicParticleSystem& parts; ParticleDataImpl<T>& pdata; WaveletNoiseField& noise; Real scale;   };
#line 55 "plugin/initplugins.cpp"



template <class T>  struct knSetPdataNoiseVec : public KernelBase { knSetPdataNoiseVec(BasicParticleSystem& parts, ParticleDataImpl<T>& pdata, WaveletNoiseField& noise, Real scale) :  KernelBase(parts.size()) ,parts(parts),pdata(pdata),noise(noise),scale(scale)   { runMessage(); run(); }   inline void op(IndexInt idx, BasicParticleSystem& parts, ParticleDataImpl<T>& pdata, WaveletNoiseField& noise, Real scale )  {
	pdata[idx] = noise.evaluateVec( parts.getPos(idx) ) * scale;
}    inline BasicParticleSystem& getArg0() { return parts; } typedef BasicParticleSystem type0;inline ParticleDataImpl<T>& getArg1() { return pdata; } typedef ParticleDataImpl<T> type1;inline WaveletNoiseField& getArg2() { return noise; } typedef WaveletNoiseField type2;inline Real& getArg3() { return scale; } typedef Real type3; void runMessage() { debMsg("Executing kernel knSetPdataNoiseVec ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,parts,pdata,noise,scale);  }   } BasicParticleSystem& parts; ParticleDataImpl<T>& pdata; WaveletNoiseField& noise; Real scale;   };
#line 59 "plugin/initplugins.cpp"


void setNoisePdata(BasicParticleSystem& parts, ParticleDataImpl<Real>& pd, WaveletNoiseField& noise, Real scale=1.) { knSetPdataNoise<Real>(parts, pd,noise,scale); }
void setNoisePdataVec3(BasicParticleSystem& parts, ParticleDataImpl<Vec3>& pd, WaveletNoiseField& noise, Real scale=1.) { knSetPdataNoiseVec<Vec3>(parts, pd,noise,scale); }
void setNoisePdataInt(BasicParticleSystem& parts, ParticleDataImpl<int >& pd, WaveletNoiseField& noise, Real scale=1.) { knSetPdataNoise<int> (parts, pd,noise,scale); }

//! SDF gradient from obstacle flags, for turbulence.py
//  FIXME, slow, without kernel...
Grid<Vec3> obstacleGradient(const FlagGrid& flags) {
	LevelsetGrid levelset(flags.getParent(),false);
	Grid<Vec3> gradient(flags.getParent());
	
	// rebuild obstacle levelset
	FOR_IDX(levelset) {
		levelset[idx] = flags.isObstacle(idx) ? -0.5 : 0.5;
	}
	levelset.reinitMarching(flags, 6.0, 0, true, false, FlagGrid::TypeReserved);
	
	// build levelset gradient
	GradientOp(gradient, levelset);
	
	FOR_IDX(levelset) {
		Vec3 grad = gradient[idx];
		Real s = normalize(grad);
		if (s <= 0.1 || levelset[idx] >= 0) 
			grad=Vec3(0.);        
		gradient[idx] = grad * levelset[idx];
	}
	
	return gradient;
}

//! SDF from obstacle flags, for turbulence.py
LevelsetGrid obstacleLevelset(const FlagGrid& flags) {
	LevelsetGrid levelset(flags.getParent(),false);

	// rebuild obstacle levelset
	FOR_IDX(levelset) {
		levelset[idx] = flags.isObstacle(idx) ? -0.5 : 0.5;
	}
	levelset.reinitMarching(flags, 6.0, 0, true, false, FlagGrid::TypeReserved);

	return levelset;
}    


//*****************************************************************************
// blender init functions 



 struct KnApplyEmission : public KernelBase { KnApplyEmission(FlagGrid& flags, Grid<Real>& density, Grid<Real>& emission, bool isAbsolute) :  KernelBase(&flags,0) ,flags(flags),density(density),emission(emission),isAbsolute(isAbsolute)   { runMessage(); run(); }  inline void op(int i, int j, int k, FlagGrid& flags, Grid<Real>& density, Grid<Real>& emission, bool isAbsolute )  {
	if (!flags.isFluid(i,j,k) || emission(i,j,k) == 0.) return;
	if (isAbsolute)
		density(i,j,k) = emission(i,j,k);
	else
		density(i,j,k) += emission(i,j,k);
}   inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline Grid<Real>& getArg1() { return density; } typedef Grid<Real> type1;inline Grid<Real>& getArg2() { return emission; } typedef Grid<Real> type2;inline bool& getArg3() { return isAbsolute; } typedef bool type3; void runMessage() { debMsg("Executing kernel KnApplyEmission ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,emission,isAbsolute);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,emission,isAbsolute);  } }  } FlagGrid& flags; Grid<Real>& density; Grid<Real>& emission; bool isAbsolute;   };
#line 111 "plugin/initplugins.cpp"



//! Add emission values
//isAbsolute: whether to add emission values to existing, or replace
void applyEmission(FlagGrid& flags, Grid<Real>& density, Grid<Real>& emission, bool isAbsolute) {
	KnApplyEmission(flags, density, emission, isAbsolute);
}

// blender init functions for meshes



 struct KnApplyDensity : public KernelBase { KnApplyDensity(FlagGrid& flags, Grid<Real>& density, Grid<Real>& sdf, Real value, Real sigma) :  KernelBase(&flags,0) ,flags(flags),density(density),sdf(sdf),value(value),sigma(sigma)   { runMessage(); run(); }  inline void op(int i, int j, int k, FlagGrid& flags, Grid<Real>& density, Grid<Real>& sdf, Real value, Real sigma )  {
	if (!flags.isFluid(i,j,k) || sdf(i,j,k) > sigma) return;
	density(i,j,k) = value;
}   inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline Grid<Real>& getArg1() { return density; } typedef Grid<Real> type1;inline Grid<Real>& getArg2() { return sdf; } typedef Grid<Real> type2;inline Real& getArg3() { return value; } typedef Real type3;inline Real& getArg4() { return sigma; } typedef Real type4; void runMessage() { debMsg("Executing kernel KnApplyDensity ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,sdf,value,sigma);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,density,sdf,value,sigma);  } }  } FlagGrid& flags; Grid<Real>& density; Grid<Real>& sdf; Real value; Real sigma;   };
#line 129 "plugin/initplugins.cpp"


//! Init noise-modulated density inside mesh

void densityInflowMeshNoise(FlagGrid& flags, Grid<Real>& density, WaveletNoiseField& noise, Mesh* mesh, Real scale=1.0, Real sigma=0) {
	LevelsetGrid sdf(density.getParent(), false);
	mesh->computeLevelset(sdf, 1.);
	KnApplyNoiseInfl(flags, density, noise, sdf, scale, sigma);
}

//! Init constant density inside mesh

void densityInflowMesh(FlagGrid& flags, Grid<Real>& density, Mesh* mesh, Real value=1., Real cutoff = 7, Real sigma=0) {
	LevelsetGrid sdf(density.getParent(), false);
	mesh->computeLevelset(sdf, 2., cutoff);
	KnApplyDensity(flags, density, sdf, value, sigma);
}


//*****************************************************************************

//! check for symmetry , optionally enfore by copying

void checkSymmetry( Grid<Real>& a, Grid<Real>* err=NULL, bool symmetrize=false, int axis=0, int bound=0) {
	const int c  = axis; 
	const int s = a.getSize()[c];
	FOR_IJK(a) { 
		Vec3i idx(i,j,k), mdx(i,j,k);
		mdx[c] = s-1-idx[c];
		if( bound>0 && ((!a.isInBounds(idx,bound)) || (!a.isInBounds(mdx,bound))) ) continue;

		if(err) (*err)(idx) = fabs( (double)(a(idx) - a(mdx) ) ); 
		if(symmetrize && (idx[c]<s/2)) {
			a(idx) = a(mdx);
		}
	}
}
//! check for symmetry , mac grid version


void checkSymmetryVec3( Grid<Vec3>& a, Grid<Real>* err=NULL, bool symmetrize=false , int axis=0, int bound=0, int disable=0) {
	if(err) err->setConst(0.);

	// each dimension is measured separately for flexibility (could be combined)
	const int c  = axis;
	const int o1 = (c+1)%3;
	const int o2 = (c+2)%3;

	// x
	if(! (disable&1) ) {
		const int s = a.getSize()[c]+1; 
		FOR_IJK(a) { 
			Vec3i idx(i,j,k), mdx(i,j,k);
			mdx[c] = s-1-idx[c]; 
			if(mdx[c] >= a.getSize()[c]) continue; 
			if( bound>0 && ((!a.isInBounds(idx,bound)) || (!a.isInBounds(mdx,bound))) ) continue;

			// special case: center "line" of values , should be zero!
			if(mdx[c] == idx[c] ) {
				if(err) (*err)(idx) += fabs( (double)( a(idx)[c] ) ); 
				if(symmetrize) a(idx)[c] = 0.;
				continue; 
			}

			// note - the a(mdx) component needs to be inverted here!
			if(err) (*err)(idx) += fabs( (double)( a(idx)[c]- (a(mdx)[c]*-1.) ) ); 
			if(symmetrize && (idx[c]<s/2)) {
				a(idx)[c] = a(mdx)[c] * -1.;
			}
		}
	}

	// y
	if(! (disable&2) ) {
		const int s = a.getSize()[c];
		FOR_IJK(a) { 
			Vec3i idx(i,j,k), mdx(i,j,k);
			mdx[c] = s-1-idx[c]; 
			if( bound>0 && ((!a.isInBounds(idx,bound)) || (!a.isInBounds(mdx,bound))) ) continue;

			if(err) (*err)(idx) += fabs( (double)( a(idx)[o1]-a(mdx)[o1] ) ); 
			if(symmetrize && (idx[c]<s/2)) {
				a(idx)[o1] = a(mdx)[o1];
			}
		}
	} 

	// z
	if(! (disable&4) ) {
		const int s = a.getSize()[c];
		FOR_IJK(a) { 
			Vec3i idx(i,j,k), mdx(i,j,k);
			mdx[c] = s-1-idx[c]; 
			if( bound>0 && ((!a.isInBounds(idx,bound)) || (!a.isInBounds(mdx,bound))) ) continue;

			if(err) (*err)(idx) += fabs( (double)( a(idx)[o2]-a(mdx)[o2] ) ); 
			if(symmetrize && (idx[c]<s/2)) {
				a(idx)[o2] = a(mdx)[o2];
			}
		}
	} 

}


// from simpleimage.cpp
void projectImg( SimpleImage& img, Grid<Real>& val, int shadeMode=0, Real scale=1.);

//! output shaded (all 3 axes at once for 3D)
//! shading modes: 0 smoke, 1 surfaces

void projectPpmFull( Grid<Real>& val, string name, int shadeMode=0, Real scale=1.) {
	SimpleImage img;
	projectImg( img, val, shadeMode, scale );
	img.writePpm( name );
}

// helper functions for pdata operator tests

//! init some test particles at the origin

void addTestParts( BasicParticleSystem& parts, int num) {
	for(int i=0; i<num; ++i)
		parts.addBuffered( Vec3(0,0,0) );

	parts.doCompress();
	parts.insertBufferedParticles();
}

//! calculate the difference between two pdata fields (note - slow!, not parallelized)

Real pdataMaxDiff( ParticleDataBase* a, ParticleDataBase* b ) {    
	double maxVal = 0.;
	//debMsg(" PD "<< a->getType()<<"  as"<<a->getSizeSlow()<<"  bs"<<b->getSizeSlow() , 1);
	assertMsg(a->getType()     == b->getType()    , "pdataMaxDiff problem - different pdata types!");
	assertMsg(a->getSizeSlow() == b->getSizeSlow(), "pdataMaxDiff problem - different pdata sizes!");
	
	if (a->getType() & ParticleDataBase::TypeReal) 
	{
		ParticleDataImpl<Real>& av = *dynamic_cast<ParticleDataImpl<Real>*>(a);
		ParticleDataImpl<Real>& bv = *dynamic_cast<ParticleDataImpl<Real>*>(b);
		FOR_PARTS(av) {
			maxVal = std::max(maxVal, (double)fabs( av[idx]-bv[idx] ));
		}
	} else if (a->getType() & ParticleDataBase::TypeInt) 
	{
		ParticleDataImpl<int>& av = *dynamic_cast<ParticleDataImpl<int>*>(a);
		ParticleDataImpl<int>& bv = *dynamic_cast<ParticleDataImpl<int>*>(b);
		FOR_PARTS(av) {
			maxVal = std::max(maxVal, (double)fabs( (double)av[idx]-bv[idx] ));
		}
	} else if (a->getType() & ParticleDataBase::TypeVec3) {
		ParticleDataImpl<Vec3>& av = *dynamic_cast<ParticleDataImpl<Vec3>*>(a);
		ParticleDataImpl<Vec3>& bv = *dynamic_cast<ParticleDataImpl<Vec3>*>(b);
		FOR_PARTS(av) {
			double d = 0.;
			for(int c=0; c<3; ++c) { 
				d += fabs( (double)av[idx][c] - (double)bv[idx][c] );
			}
			maxVal = std::max(maxVal, d );
		}
	} else {
		errMsg("pdataMaxDiff: Grid Type is not supported (only Real, Vec3, int)");    
	}

	return maxVal;
}


//! calculate center of mass given density grid, for re-centering

Vec3 calcCenterOfMass(Grid<Real>& density) {
	Vec3 p(0.0f);
	Real w = 0.0f;
	FOR_IJK(density){
		p += density(i, j, k) * Vec3(i + 0.5f, j + 0.5f, k + 0.5f);
		w += density(i, j, k);
	}
	if (w > 1e-6f)
		p /= w;
	return p;
}


//*****************************************************************************
// helper functions for volume fractions (which are needed for second order obstacle boundaries)



inline static Real calcFraction(Real phi1, Real phi2)
{
	if(phi1>0. && phi2>0.) return 1.;
	if(phi1<0. && phi2<0.) return 0.;

	// make sure phi1 < phi2
	if (phi2<phi1) { Real t = phi1; phi1= phi2; phi2 = t; }
	Real denom = phi1-phi2;
	if (denom > -1e-04) return 0.5; 

	Real frac = 1. - phi1/denom;
	if(frac<0.01) frac = 0.; // stomp small values , dont mark as fluid
	return std::min(Real(1), frac );
}


 struct KnUpdateFractions : public KernelBase { KnUpdateFractions(const FlagGrid& flags, const Grid<Real>& phiObs, MACGrid& fractions, const int &boundaryWidth) :  KernelBase(&flags,1) ,flags(flags),phiObs(phiObs),fractions(fractions),boundaryWidth(boundaryWidth)   { runMessage(); run(); }  inline void op(int i, int j, int k, const FlagGrid& flags, const Grid<Real>& phiObs, MACGrid& fractions, const int &boundaryWidth )  {

	// walls at domain bounds and inner objects
	fractions(i,j,k).x = calcFraction( phiObs(i,j,k) , phiObs(i-1,j,k));
	fractions(i,j,k).y = calcFraction( phiObs(i,j,k) , phiObs(i,j-1,k));
    if(phiObs.is3D()) {
	fractions(i,j,k).z = calcFraction( phiObs(i,j,k) , phiObs(i,j,k-1));
	}

	// remaining BCs at the domain boundaries 
	const int w = boundaryWidth;
	// only set if not in obstacle
 	if(phiObs(i,j,k)<0.) return;

	// x-direction boundaries
	if(i <= w+1) {                     //min x
		if( (flags.isInflow(i-1,j,k)) ||
			(flags.isOutflow(i-1,j,k)) ||
			(flags.isOpen(i-1,j,k)) ) {
				fractions(i,j,k).x = fractions(i,j,k).y = 1.; if(flags.is3D()) fractions(i,j,k).z = 1.;
		}
	}
	if(i >= flags.getSizeX()-w-2) {    //max x
		if(	(flags.isInflow(i+1,j,k)) ||
			(flags.isOutflow(i+1,j,k)) ||
			(flags.isOpen(i+1,j,k)) ) {
			fractions(i+1,j,k).x = fractions(i+1,j,k).y = 1.; if(flags.is3D()) fractions(i+1,j,k).z = 1.;
		}
	}
	// y-direction boundaries
 	if(j <= w+1) {                     //min y
		if(	(flags.isInflow(i,j-1,k)) ||
			(flags.isOutflow(i,j-1,k)) ||
			(flags.isOpen(i,j-1,k)) ) {
			fractions(i,j,k).x = fractions(i,j,k).y = 1.; if(flags.is3D()) fractions(i,j,k).z = 1.;
		}
 	}
 	if(j >= flags.getSizeY()-w-2) {      //max y
		if(	(flags.isInflow(i,j+1,k)) ||
			(flags.isOutflow(i,j+1,k)) ||
			(flags.isOpen(i,j+1,k)) ) {
			fractions(i,j+1,k).x = fractions(i,j+1,k).y = 1.; if(flags.is3D()) fractions(i,j+1,k).z = 1.;
		}
 	}
	// z-direction boundaries
	if(flags.is3D()) {
	if(k <= w+1) {                 //min z
		if(	(flags.isInflow(i,j,k-1)) ||
			(flags.isOutflow(i,j,k-1)) ||
			(flags.isOpen(i,j,k-1)) ) {
			fractions(i,j,k).x = fractions(i,j,k).y = 1.; if(flags.is3D()) fractions(i,j,k).z = 1.;
		}
	}
	if(j >= flags.getSizeZ()-w-2) { //max z
		if(	(flags.isInflow(i,j,k+1)) ||
			(flags.isOutflow(i,j,k+1)) ||
			(flags.isOpen(i,j,k+1)) ) {
			fractions(i,j,k+1).x = fractions(i,j,k+1).y = 1.; if(flags.is3D()) fractions(i,j,k+1).z = 1.;
		}
	}
	}

}   inline const FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline const Grid<Real>& getArg1() { return phiObs; } typedef Grid<Real> type1;inline MACGrid& getArg2() { return fractions; } typedef MACGrid type2;inline const int& getArg3() { return boundaryWidth; } typedef int type3; void runMessage() { debMsg("Executing kernel KnUpdateFractions ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,flags,phiObs,fractions,boundaryWidth);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,flags,phiObs,fractions,boundaryWidth);  } }  } const FlagGrid& flags; const Grid<Real>& phiObs; MACGrid& fractions; const int& boundaryWidth;   };
#line 336 "plugin/initplugins.cpp"



//! update fill fraction values
void updateFractions(const FlagGrid& flags, const Grid<Real>& phiObs, MACGrid& fractions, const int &boundaryWidth=0) {
	fractions.setConst( Vec3(0.) );
	KnUpdateFractions(flags, phiObs, fractions, boundaryWidth);
}


 struct KnUpdateFlagsObs : public KernelBase { KnUpdateFlagsObs(FlagGrid& flags, const MACGrid* fractions, const Grid<Real>& phiObs, const Grid<Real>* phiOut ) :  KernelBase(&flags,1) ,flags(flags),fractions(fractions),phiObs(phiObs),phiOut(phiOut)   { runMessage(); run(); }  inline void op(int i, int j, int k, FlagGrid& flags, const MACGrid* fractions, const Grid<Real>& phiObs, const Grid<Real>* phiOut  )  {

	bool isObs = false;
	if(fractions) {
		Real f = 0.;
		f += fractions->get(i  ,j,k).x;
		f += fractions->get(i+1,j,k).x;
		f += fractions->get(i,j  ,k).y;
		f += fractions->get(i,j+1,k).y;
		if (flags.is3D()) {
		f += fractions->get(i,j,k  ).z;
		f += fractions->get(i,j,k+1).z; }
		if(f==0.) isObs = true;
	} else {
		if(phiObs(i,j,k) < 0.) isObs = true;
	}

	bool isOutflow = false;
 	if (phiOut && (*phiOut)(i,j,k) < 0.) isOutflow = true;

 	if (isObs)          flags(i,j,k) = FlagGrid::TypeObstacle;
 	else if (isOutflow) flags(i,j,k) = (FlagGrid::TypeEmpty | FlagGrid::TypeOutflow);
  	else                flags(i,j,k) = FlagGrid::TypeEmpty;
}   inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline const MACGrid* getArg1() { return fractions; } typedef MACGrid type1;inline const Grid<Real>& getArg2() { return phiObs; } typedef Grid<Real> type2;inline const Grid<Real>* getArg3() { return phiOut; } typedef Grid<Real> type3; void runMessage() { debMsg("Executing kernel KnUpdateFlagsObs ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,flags,fractions,phiObs,phiOut);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,flags,fractions,phiObs,phiOut);  } }  } FlagGrid& flags; const MACGrid* fractions; const Grid<Real>& phiObs; const Grid<Real>* phiOut;   };
#line 407 "plugin/initplugins.cpp"



//! update obstacle and outflow flags from levelsets
//! optionally uses fill fractions for obstacle
void setObstacleFlags(FlagGrid& flags, const Grid<Real>& phiObs, const MACGrid* fractions=NULL, const Grid<Real>* phiOut=NULL ) {
	KnUpdateFlagsObs(flags, fractions, phiObs, phiOut );
}


//! small helper for test case test_1040_secOrderBnd.py
 struct kninitVortexVelocity : public KernelBase { kninitVortexVelocity(const Grid<Real> &phiObs, MACGrid& vel, const Vec3 &center, const Real &radius) :  KernelBase(&phiObs,0) ,phiObs(phiObs),vel(vel),center(center),radius(radius)   { runMessage(); run(); }  inline void op(int i, int j, int k, const Grid<Real> &phiObs, MACGrid& vel, const Vec3 &center, const Real &radius )  {
	
	if(phiObs(i,j,k) >= -1.) {

		Real dx = i - center.x; if(dx>=0) dx -= .5; else dx += .5;
		Real dy = j - center.y;
		Real r = std::sqrt(dx*dx+dy*dy);
		Real alpha = atan2(dy,dx);

		vel(i,j,k).x = -std::sin(alpha)*(r/radius);

		dx = i - center.x;
		dy = j - center.y; if(dy>=0) dy -= .5; else dy += .5;
		r = std::sqrt(dx*dx+dy*dy);
		alpha = atan2(dy,dx);

		vel(i,j,k).y = std::cos(alpha)*(r/radius);

	}

}   inline const Grid<Real> & getArg0() { return phiObs; } typedef Grid<Real>  type0;inline MACGrid& getArg1() { return vel; } typedef MACGrid type1;inline const Vec3& getArg2() { return center; } typedef Vec3 type2;inline const Real& getArg3() { return radius; } typedef Real type3; void runMessage() { debMsg("Executing kernel kninitVortexVelocity ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,phiObs,vel,center,radius);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,phiObs,vel,center,radius);  } }  } const Grid<Real> & phiObs; MACGrid& vel; const Vec3& center; const Real& radius;   };
#line 440 "plugin/initplugins.cpp"



void initVortexVelocity(Grid<Real> &phiObs, MACGrid& vel, const Vec3 &center, const Real &radius) {
	kninitVortexVelocity(phiObs,  vel, center, radius);
}


} // namespace



