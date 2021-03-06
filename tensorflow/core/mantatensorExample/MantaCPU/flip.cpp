




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/plugin/flip.cpp"
/******************************************************************************
 *
 * MantaFlow fluid solver framework 
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * FLIP (fluid implicit particles)
 * for use with particle data fields
 *
 ******************************************************************************/

#include "particle.h"
#include "grid.h"
#include "commonkernels.h"
#include "randomstream.h"
#include "levelset.h"
#include "shapes.h"

using namespace std;
namespace Manta {



// init

//! note - this is a simplified version , sampleLevelsetWithParticles has more functionality


void sampleFlagsWithParticles(const FlagGrid& flags, BasicParticleSystem& parts, const int discretization, const Real randomness) {
	const bool is3D = flags.is3D();
	const Real jlen = randomness / discretization;
	const Vec3 disp (1.0 / discretization, 1.0 / discretization, 1.0/discretization);
	RandomStream mRand(9832);
 
	FOR_IJK_BND(flags, 0) {
		if ( flags.isObstacle(i,j,k) ) continue;
		if ( flags.isFluid(i,j,k) ) {
			const Vec3 pos (i,j,k);
			for (int dk=0; dk<(is3D ? discretization : 1); dk++)
			for (int dj=0; dj<discretization; dj++)
			for (int di=0; di<discretization; di++) {
				Vec3 subpos = pos + disp * Vec3(0.5+di, 0.5+dj, 0.5+dk);
				subpos += jlen * (Vec3(1,1,1) - 2.0 * mRand.getVec3());
				if(!is3D) subpos[2] = 0.5; 
				parts.addBuffered(subpos);
			}
		}
	}
	parts.insertBufferedParticles();
}

//! sample a level set with particles, use reset to clear the particle buffer,
//! and skipEmpty for a continuous inflow (in the latter case, only empty cells will
//! be re-filled once they empty when calling sampleLevelsetWithParticles during 
//! the main loop).


void sampleLevelsetWithParticles(const LevelsetGrid& phi, const FlagGrid& flags, BasicParticleSystem& parts, const int discretization, const Real randomness, const bool reset=false, const bool refillEmpty=false) {
	const bool is3D = phi.is3D();
	const Real jlen = randomness / discretization;
	const Vec3 disp (1.0 / discretization, 1.0 / discretization, 1.0/discretization);
	RandomStream mRand(9832);
 
	if(reset) {
		parts.clear(); 
		parts.doCompress();
	}

	FOR_IJK_BND(phi, 0) {
		if ( flags.isObstacle(i,j,k) ) continue;
		if ( refillEmpty && flags.isFluid(i,j,k) ) continue;
		if ( phi(i,j,k) < 1.733 ) {
			const Vec3 pos (i,j,k);
			for (int dk=0; dk<(is3D ? discretization : 1); dk++)
			for (int dj=0; dj<discretization; dj++)
			for (int di=0; di<discretization; di++) {
				Vec3 subpos = pos + disp * Vec3(0.5+di, 0.5+dj, 0.5+dk);
				subpos += jlen * (Vec3(1,1,1) - 2.0 * mRand.getVec3());
				if(!is3D) subpos[2] = 0.5; 
				if( phi.getInterpolated(subpos) > 0. ) continue; 
				parts.addBuffered(subpos);
			}
		}
	}

	parts.insertBufferedParticles();
}

//! sample a shape with particles, use reset to clear the particle buffer,
//! and skipEmpty for a continuous inflow (in the latter case, only empty cells will
//! be re-filled once they empty when calling sampleShapeWithParticles during
//! the main loop).



void sampleShapeWithParticles(const Shape& shape, const FlagGrid& flags, BasicParticleSystem& parts, const int discretization, const Real randomness, const bool reset=false, const bool refillEmpty=false, const LevelsetGrid *exclude=NULL) {
	const bool is3D = flags.is3D();
	const Real jlen = randomness / discretization;
	const Vec3 disp (1.0 / discretization, 1.0 / discretization, 1.0/discretization);
	RandomStream mRand(9832);

	if(reset) {
		parts.clear();
		parts.doCompress();
	}

	FOR_IJK_BND(flags, 0) {
		if ( flags.isObstacle(i,j,k) ) continue;
		if ( refillEmpty && flags.isFluid(i,j,k) ) continue;
		const Vec3 pos (i,j,k);
		for (int dk=0; dk<(is3D ? discretization : 1); dk++)
		for (int dj=0; dj<discretization; dj++)
		for (int di=0; di<discretization; di++) {
			Vec3 subpos = pos + disp * Vec3(0.5+di, 0.5+dj, 0.5+dk);
			subpos += jlen * (Vec3(1,1,1) - 2.0 * mRand.getVec3());
			if(!is3D) subpos[2] = 0.5;
			if(exclude && exclude->getInterpolated(subpos) <= 0.) continue;
			if(!shape.isInside(subpos)) continue;
			parts.addBuffered(subpos);
		}
	}

	parts.insertBufferedParticles();
}

//! mark fluid cells and helpers
 struct knClearFluidFlags : public KernelBase { knClearFluidFlags(FlagGrid& flags, int dummy=0) :  KernelBase(&flags,0) ,flags(flags),dummy(dummy)   { runMessage(); run(); }  inline void op(int i, int j, int k, FlagGrid& flags, int dummy=0 )  {
	if (flags.isFluid(i,j,k)) {
		flags(i,j,k) = (flags(i,j,k) | FlagGrid::TypeEmpty) & ~FlagGrid::TypeFluid;
	}
}   inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline int& getArg1() { return dummy; } typedef int type1; void runMessage() { debMsg("Executing kernel knClearFluidFlags ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,dummy);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,flags,dummy);  } }  } FlagGrid& flags; int dummy;   };
#line 130 "plugin/flip.cpp"



 struct knSetNbObstacle : public KernelBase { knSetNbObstacle(FlagGrid& nflags, const FlagGrid& flags, const Grid<Real>& phiObs) :  KernelBase(&nflags,1) ,nflags(nflags),flags(flags),phiObs(phiObs)   { runMessage(); run(); }  inline void op(int i, int j, int k, FlagGrid& nflags, const FlagGrid& flags, const Grid<Real>& phiObs )  {
	if ( phiObs(i,j,k)>0. ) return;
	if (flags.isEmpty(i,j,k)) {
		bool set=false;
		if( (flags.isFluid(i-1,j,k)) && (phiObs(i+1,j,k)<=0.) ) set=true;
		if( (flags.isFluid(i+1,j,k)) && (phiObs(i-1,j,k)<=0.) ) set=true;
		if( (flags.isFluid(i,j-1,k)) && (phiObs(i,j+1,k)<=0.) ) set=true;
		if( (flags.isFluid(i,j+1,k)) && (phiObs(i,j-1,k)<=0.) ) set=true;
		if(flags.is3D()) {
		if( (flags.isFluid(i,j,k-1)) && (phiObs(i,j,k+1)<=0.) ) set=true;
		if( (flags.isFluid(i,j,k+1)) && (phiObs(i,j,k-1)<=0.) ) set=true;
		}
		if(set) nflags(i,j,k) = (flags(i,j,k) | FlagGrid::TypeFluid) & ~FlagGrid::TypeEmpty;
	}
}   inline FlagGrid& getArg0() { return nflags; } typedef FlagGrid type0;inline const FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline const Grid<Real>& getArg2() { return phiObs; } typedef Grid<Real> type2; void runMessage() { debMsg("Executing kernel knSetNbObstacle ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,nflags,flags,phiObs);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,nflags,flags,phiObs);  } }  } FlagGrid& nflags; const FlagGrid& flags; const Grid<Real>& phiObs;   };
#line 136 "plugin/flip.cpp"


void markFluidCells(const BasicParticleSystem& parts, FlagGrid& flags, const Grid<Real>* phiObs=NULL, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	// remove all fluid cells
	knClearFluidFlags(flags, 0);
	
	// mark all particles in flaggrid as fluid
	for(IndexInt idx=0; idx<parts.size(); idx++) {
		if (!parts.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) continue;
		Vec3i p = toVec3i( parts.getPos(idx) );
		if (flags.isInBounds(p) && flags.isEmpty(p))
			flags(p) = (flags(p) | FlagGrid::TypeFluid) & ~FlagGrid::TypeEmpty;
	}

	// special for second order obstacle BCs, check empty cells in boundary region
	if(phiObs) {
		FlagGrid tmp(flags);
		knSetNbObstacle(tmp, flags, *phiObs);
		flags.swap(tmp);
	}
}

// for testing purposes only...
void testInitGridWithPos(Grid<Real>& grid) {
	FOR_IJK(grid) { grid(i,j,k) = norm( Vec3(i,j,k) ); }
}



//! helper to calculate particle radius factor to cover the diagonal of a cell in 2d/3d
inline Real calculateRadiusFactor(Grid<Real>& grid, Real factor) {
	return (grid.is3D() ? sqrt(3.) : sqrt(2.) ) * (factor+.01); // note, a 1% safety factor is added here
} 

//! re-sample particles based on an input levelset 
// optionally skip seeding new particles in "exclude" SDF



void adjustNumber( BasicParticleSystem& parts, MACGrid& vel, FlagGrid& flags, int minParticles, int maxParticles, LevelsetGrid& phi, Real radiusFactor=1. , Real narrowBand=-1. , Grid<Real>* exclude=NULL ) {
	// which levelset to use as threshold
	const Real SURFACE_LS = -1.0 * calculateRadiusFactor(phi, radiusFactor);
	Grid<int> tmp( vel.getParent() );
	std::ostringstream out;

	// count particles in cells, and delete excess particles
	for (IndexInt idx=0; idx<(int)parts.size(); idx++) {
		if (parts.isActive(idx)) {
			Vec3i p = toVec3i( parts.getPos(idx) );
			if (!tmp.isInBounds(p) ) {
				parts.kill(idx); // out of domain, remove
				continue;
			}

			Real phiv = phi.getInterpolated( parts.getPos(idx) );
			if( phiv > 0 ) { parts.kill(idx); continue; }
			if( narrowBand>0. && phiv < -narrowBand) { parts.kill(idx); continue; }

			bool atSurface = false;
			if (phiv > SURFACE_LS) atSurface = true;
			int num = tmp(p);
			
			// dont delete particles in non fluid cells here, the particles are "always right"
			if ( num > maxParticles && (!atSurface) ) {
				parts.kill(idx);
			} else {
				tmp(p) = num+1;
			}
		}
	}

	// seed new particles
	RandomStream mRand(9832);
	FOR_IJK(tmp) {
		int cnt = tmp(i,j,k);
		
		// skip cells near surface
		if (phi(i,j,k) > SURFACE_LS) continue;
		if( narrowBand>0. && phi(i,j,k) < -narrowBand ) { continue; }
		if( exclude && ( (*exclude)(i,j,k) < 0.) ) { continue; }

		if (flags.isFluid(i,j,k) && cnt < minParticles) {
			for (int m=cnt; m < minParticles; m++) { 
				Vec3 pos = Vec3(i,j,k) + mRand.getVec3();
				//Vec3 pos (i + 0.5, j + 0.5, k + 0.5); // cell center
				parts.addBuffered( pos ); 
			}
		}
	}

	parts.doCompress();
	parts.insertBufferedParticles();
}

// simple and slow helper conversion to show contents of int grids like a real grid in the ui
// (use eg to quickly display contents of the particle-index grid)

void debugIntToReal( Grid<int>& source, Grid<Real>& dest, Real factor=1. ) {
	FOR_IJK( source ) { dest(i,j,k) = (Real)source(i,j,k) * factor; }
}

// build a grid that contains indices for a particle system
// the particles in a cell i,j,k are particles[index(i,j,k)] to particles[index(i+1,j,k)-1]
// (ie,  particles[index(i+1,j,k)] already belongs to cell i+1,j,k)


void gridParticleIndex( BasicParticleSystem& parts, ParticleIndexSystem& indexSys, FlagGrid& flags, Grid<int>& index, Grid<int>* counter=NULL ) {
	bool delCounter = false;
	if(!counter) { counter = new Grid<int>(  flags.getParent() ); delCounter=true; }
	else         { counter->clear(); }
	
	// count particles in cells, and delete excess particles
	index.clear();
	int inactive = 0;
	for (IndexInt idx=0; idx<(IndexInt)parts.size(); idx++) {
		if (parts.isActive(idx)) {
			// check index for validity...
			Vec3i p = toVec3i( parts.getPos(idx) );
			if (! index.isInBounds(p)) { inactive++; continue; }

			index(p)++;
		} else {
			inactive++;
		}
	}

	// note - this one might be smaller...
	indexSys.resize( parts.size()-inactive );

	// convert per cell number to continuous index
	IndexInt idx=0;
	FOR_IJK( index ) {
		int num = index(i,j,k);
		index(i,j,k) = idx;
		idx += num;
	}

	// add particles to indexed array, we still need a per cell particle counter
	for (IndexInt idx=0; idx<(IndexInt)parts.size(); idx++) {
		if (!parts.isActive(idx)) continue;
		Vec3i p = toVec3i( parts.getPos(idx) );
		if (! index.isInBounds(p)) { continue; }

		// initialize position and index into original array
		//indexSys[ index(p)+(*counter)(p) ].pos        = parts[idx].pos;
		indexSys[ index(p)+(*counter)(p) ].sourceIndex = idx;
		(*counter)(p)++;
	}

	if(delCounter) delete counter;
}





 struct ComputeUnionLevelsetPindex : public KernelBase { ComputeUnionLevelsetPindex(const Grid<int>& index, const BasicParticleSystem& parts, const ParticleIndexSystem& indexSys, LevelsetGrid& phi, const Real radius, const ParticleDataImpl<int> *ptype, const int exclude) :  KernelBase(&index,0) ,index(index),parts(parts),indexSys(indexSys),phi(phi),radius(radius),ptype(ptype),exclude(exclude)   { runMessage(); run(); }  inline void op(int i, int j, int k, const Grid<int>& index, const BasicParticleSystem& parts, const ParticleIndexSystem& indexSys, LevelsetGrid& phi, const Real radius, const ParticleDataImpl<int> *ptype, const int exclude )  {
	const Vec3 gridPos = Vec3(i,j,k) + Vec3(0.5); // shifted by half cell
	Real phiv = radius * 1.0;  // outside

	int r  = int(radius) + 1;
	int rZ = phi.is3D() ? r : 0;
	for(int zj=k-rZ; zj<=k+rZ; zj++) 
	for(int yj=j-r ; yj<=j+r ; yj++) 
	for(int xj=i-r ; xj<=i+r ; xj++) {
		if (!phi.isInBounds(Vec3i(xj,yj,zj))) continue;

		// note, for the particle indices in indexSys the access is periodic (ie, dont skip for eg inBounds(sx,10,10)
		IndexInt isysIdxS = index.index(xj,yj,zj);
		IndexInt pStart = index(isysIdxS), pEnd=0;
		if(phi.isInBounds(isysIdxS+1)) pEnd = index(isysIdxS+1);
		else                           pEnd = indexSys.size();

		// now loop over particles in cell
		for(IndexInt p=pStart; p<pEnd; ++p) {
			const int psrc = indexSys[p].sourceIndex;
			if(ptype && ((*ptype)[psrc] & exclude)) continue;
			const Vec3 pos = parts[psrc].pos;
			phiv = std::min( phiv , fabs( norm(gridPos-pos) )-radius );
		}
	}
	phi(i,j,k) = phiv;
}   inline const Grid<int>& getArg0() { return index; } typedef Grid<int> type0;inline const BasicParticleSystem& getArg1() { return parts; } typedef BasicParticleSystem type1;inline const ParticleIndexSystem& getArg2() { return indexSys; } typedef ParticleIndexSystem type2;inline LevelsetGrid& getArg3() { return phi; } typedef LevelsetGrid type3;inline const Real& getArg4() { return radius; } typedef Real type4;inline const ParticleDataImpl<int> * getArg5() { return ptype; } typedef ParticleDataImpl<int>  type5;inline const int& getArg6() { return exclude; } typedef int type6; void runMessage() { debMsg("Executing kernel ComputeUnionLevelsetPindex ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,index,parts,indexSys,phi,radius,ptype,exclude);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,index,parts,indexSys,phi,radius,ptype,exclude);  } }  } const Grid<int>& index; const BasicParticleSystem& parts; const ParticleIndexSystem& indexSys; LevelsetGrid& phi; const Real radius; const ParticleDataImpl<int> * ptype; const int exclude;   };
#line 305 "plugin/flip.cpp"


 



void unionParticleLevelset(const BasicParticleSystem& parts, const ParticleIndexSystem& indexSys, const FlagGrid& flags, const Grid<int>& index, LevelsetGrid& phi, const Real radiusFactor=1., const ParticleDataImpl<int> *ptype=NULL, const int exclude=0) {
	// use half a cell diagonal as base radius
	const Real radius = 0.5 * calculateRadiusFactor(phi, radiusFactor);
	// no reset of phi necessary here 
	ComputeUnionLevelsetPindex(index, parts, indexSys, phi, radius, ptype, exclude);

	phi.setBound(0.5, 0);
}







 struct ComputeAveragedLevelsetWeight : public KernelBase { ComputeAveragedLevelsetWeight(const BasicParticleSystem& parts, const Grid<int>& index, const ParticleIndexSystem& indexSys, LevelsetGrid& phi, const Real radius, const ParticleDataImpl<int>* ptype, const int exclude) :  KernelBase(&index,0) ,parts(parts),index(index),indexSys(indexSys),phi(phi),radius(radius),ptype(ptype),exclude(exclude)   { runMessage(); run(); }  inline void op(int i, int j, int k, const BasicParticleSystem& parts, const Grid<int>& index, const ParticleIndexSystem& indexSys, LevelsetGrid& phi, const Real radius, const ParticleDataImpl<int>* ptype, const int exclude )  {
	const Vec3 gridPos = Vec3(i,j,k) + Vec3(0.5); // shifted by half cell
	Real phiv = radius * 1.0; // outside 

	// loop over neighborhood, similar to ComputeUnionLevelsetPindex
	const Real sradiusInv = 1. / (4. * radius * radius) ;
	int   r = int(1. * radius) + 1;
	int   rZ = phi.is3D() ? r : 0;
	// accumulators
	Real  wacc = 0.;
	Vec3  pacc = Vec3(0.);
	Real  racc = 0.;

	for(int zj=k-rZ; zj<=k+rZ; zj++) 
	for(int yj=j-r ; yj<=j+r ; yj++) 
	for(int xj=i-r ; xj<=i+r ; xj++) {
		if (! phi.isInBounds(Vec3i(xj,yj,zj)) ) continue;

		IndexInt isysIdxS = index.index(xj,yj,zj);
		IndexInt pStart = index(isysIdxS), pEnd=0;
		if(phi.isInBounds(isysIdxS+1)) pEnd = index(isysIdxS+1);
		else                           pEnd = indexSys.size();
		for(IndexInt p=pStart; p<pEnd; ++p) {
			IndexInt   psrc = indexSys[p].sourceIndex;
			if(ptype && ((*ptype)[psrc] & exclude)) continue;

			Vec3  pos  = parts[psrc].pos; 
			Real  s    = normSquare(gridPos-pos) * sradiusInv;
			//Real  w = std::max(0., cubed(1.-s) );
			Real  w = std::max(0., (1.-s)); // a bit smoother
			wacc += w;
			racc += radius * w;
			pacc += pos    * w;
		} 
	}

	if(wacc > VECTOR_EPSILON) {
		racc /= wacc;
		pacc /= wacc;
		phiv = fabs( norm(gridPos-pacc) )-racc;
	}
	phi(i,j,k) = phiv;
}   inline const BasicParticleSystem& getArg0() { return parts; } typedef BasicParticleSystem type0;inline const Grid<int>& getArg1() { return index; } typedef Grid<int> type1;inline const ParticleIndexSystem& getArg2() { return indexSys; } typedef ParticleIndexSystem type2;inline LevelsetGrid& getArg3() { return phi; } typedef LevelsetGrid type3;inline const Real& getArg4() { return radius; } typedef Real type4;inline const ParticleDataImpl<int>* getArg5() { return ptype; } typedef ParticleDataImpl<int> type5;inline const int& getArg6() { return exclude; } typedef int type6; void runMessage() { debMsg("Executing kernel ComputeAveragedLevelsetWeight ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,parts,index,indexSys,phi,radius,ptype,exclude);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,parts,index,indexSys,phi,radius,ptype,exclude);  } }  } const BasicParticleSystem& parts; const Grid<int>& index; const ParticleIndexSystem& indexSys; LevelsetGrid& phi; const Real radius; const ParticleDataImpl<int>* ptype; const int exclude;   };
#line 351 "plugin/flip.cpp"



template<class T> T smoothingValue(Grid<T> val, int i, int j, int k, T center) {
	return val(i,j,k);
}

// smoothing, and  

template <class T>  struct knSmoothGrid : public KernelBase { knSmoothGrid(Grid<T>& me, Grid<T>& tmp, Real factor) :  KernelBase(&me,1) ,me(me),tmp(tmp),factor(factor)   { runMessage(); run(); }  inline void op(int i, int j, int k, Grid<T>& me, Grid<T>& tmp, Real factor )  {
	T val = me(i,j,k) + 
			me(i+1,j,k) + me(i-1,j,k) + 
			me(i,j+1,k) + me(i,j-1,k) ;
	if(me.is3D()) {
		val += me(i,j,k+1) + me(i,j,k-1);
	}
	tmp(i,j,k) = val * factor;
}   inline Grid<T>& getArg0() { return me; } typedef Grid<T> type0;inline Grid<T>& getArg1() { return tmp; } typedef Grid<T> type1;inline Real& getArg2() { return factor; } typedef Real type2; void runMessage() { debMsg("Executing kernel knSmoothGrid ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,me,tmp,factor);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,me,tmp,factor);  } }  } Grid<T>& me; Grid<T>& tmp; Real factor;   };
#line 401 "plugin/flip.cpp"




template <class T>  struct knSmoothGridNeg : public KernelBase { knSmoothGridNeg(Grid<T>& me, Grid<T>& tmp, Real factor) :  KernelBase(&me,1) ,me(me),tmp(tmp),factor(factor)   { runMessage(); run(); }  inline void op(int i, int j, int k, Grid<T>& me, Grid<T>& tmp, Real factor )  {
	T val = me(i,j,k) + 
			me(i+1,j,k) + me(i-1,j,k) + 
			me(i,j+1,k) + me(i,j-1,k) ;
	if(me.is3D()) {
		val += me(i,j,k+1) + me(i,j,k-1);
	}
	val *= factor;
	if(val<tmp(i,j,k)) tmp(i,j,k) = val;
	else               tmp(i,j,k) = me(i,j,k);
}   inline Grid<T>& getArg0() { return me; } typedef Grid<T> type0;inline Grid<T>& getArg1() { return tmp; } typedef Grid<T> type1;inline Real& getArg2() { return factor; } typedef Real type2; void runMessage() { debMsg("Executing kernel knSmoothGridNeg ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,me,tmp,factor);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,me,tmp,factor);  } }  } Grid<T>& me; Grid<T>& tmp; Real factor;   };
#line 412 "plugin/flip.cpp"



 




void averagedParticleLevelset(const BasicParticleSystem& parts, const ParticleIndexSystem& indexSys, const FlagGrid& flags, const Grid<int>& index, LevelsetGrid& phi, const Real radiusFactor=1., const int smoothen=1, const int smoothenNeg=1, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	// use half a cell diagonal as base radius
	const Real radius = 0.5 * calculateRadiusFactor(phi, radiusFactor); 
	ComputeAveragedLevelsetWeight(parts, index, indexSys, phi, radius, ptype, exclude);

	// post-process level-set
	for(int i=0; i<std::max(smoothen,smoothenNeg); ++i) {
		LevelsetGrid tmp(flags.getParent());
		if(i<smoothen) { 
			knSmoothGrid    <Real> (phi,tmp, 1./(phi.is3D() ? 7. : 5.) );
			phi.swap(tmp);
		}
		if(i<smoothenNeg) { 
			knSmoothGridNeg <Real> (phi,tmp, 1./(phi.is3D() ? 7. : 5.) );
			phi.swap(tmp);
		}
	} 
	phi.setBound(0.5, 0);
}



 struct knPushOutofObs : public KernelBase { knPushOutofObs(BasicParticleSystem& parts, const FlagGrid& flags, const Grid<Real>& phiObs, const Real shift, const Real thresh, const ParticleDataImpl<int>* ptype, const int exclude) :  KernelBase(parts.size()) ,parts(parts),flags(flags),phiObs(phiObs),shift(shift),thresh(thresh),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, BasicParticleSystem& parts, const FlagGrid& flags, const Grid<Real>& phiObs, const Real shift, const Real thresh, const ParticleDataImpl<int>* ptype, const int exclude )  {
	if (!parts.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	Vec3i p = toVec3i( parts.getPos(idx) );

	if (!flags.isInBounds(p)) return;
	Real v = phiObs.getInterpolated(parts.getPos(idx));
	if(v < thresh) {
		Vec3 grad = getGradient( phiObs, p.x,p.y,p.z );
		if( normalize(grad) < VECTOR_EPSILON ) return;
		parts.setPos(idx, parts.getPos(idx) + grad*(thresh - v + shift));
	}
}    inline BasicParticleSystem& getArg0() { return parts; } typedef BasicParticleSystem type0;inline const FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline const Grid<Real>& getArg2() { return phiObs; } typedef Grid<Real> type2;inline const Real& getArg3() { return shift; } typedef Real type3;inline const Real& getArg4() { return thresh; } typedef Real type4;inline const ParticleDataImpl<int>* getArg5() { return ptype; } typedef ParticleDataImpl<int> type5;inline const int& getArg6() { return exclude; } typedef int type6; void runMessage() { debMsg("Executing kernel knPushOutofObs ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,parts,flags,phiObs,shift,thresh,ptype,exclude);  }   } BasicParticleSystem& parts; const FlagGrid& flags; const Grid<Real>& phiObs; const Real shift; const Real thresh; const ParticleDataImpl<int>* ptype; const int exclude;   };
#line 451 "plugin/flip.cpp"


//! push particles out of obstacle levelset

void pushOutofObs(BasicParticleSystem& parts, const FlagGrid& flags, const Grid<Real>& phiObs, const Real shift=0, const Real thresh=0, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	knPushOutofObs(parts, flags, phiObs, shift, thresh, ptype, exclude);
}

//******************************************************************************
// grid interpolation functions


template <class T>  struct knSafeDivReal : public KernelBase { knSafeDivReal(Grid<T>& me, const Grid<Real>& other, Real cutoff=VECTOR_EPSILON) :  KernelBase(&me,0) ,me(me),other(other),cutoff(cutoff)   { runMessage(); run(); }   inline void op(IndexInt idx, Grid<T>& me, const Grid<Real>& other, Real cutoff=VECTOR_EPSILON )  { 
	if(other[idx]<cutoff) {
		me[idx] = 0.;
	} else {
		T div( other[idx] );
		me[idx] = safeDivide(me[idx], div ); 
	}
}    inline Grid<T>& getArg0() { return me; } typedef Grid<T> type0;inline const Grid<Real>& getArg1() { return other; } typedef Grid<Real> type1;inline Real& getArg2() { return cutoff; } typedef Real type2; void runMessage() { debMsg("Executing kernel knSafeDivReal ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,me,other,cutoff);  }   } Grid<T>& me; const Grid<Real>& other; Real cutoff;   };
#line 473 "plugin/flip.cpp"



// Set velocities on the grid from the particle system




 struct knMapLinearVec3ToMACGrid : public KernelBase { knMapLinearVec3ToMACGrid(const BasicParticleSystem& p, const FlagGrid& flags, MACGrid& vel, Grid<Vec3>& tmp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<int>* ptype, const int exclude) :  KernelBase(p.size()) ,p(p),flags(flags),vel(vel),tmp(tmp),pvel(pvel),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, const BasicParticleSystem& p, const FlagGrid& flags, MACGrid& vel, Grid<Vec3>& tmp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<int>* ptype, const int exclude )  {
	unusedParameter(flags);
	if (!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	vel.setInterpolated( p[idx].pos, pvel[idx], &tmp[0] );
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline const FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline MACGrid& getArg2() { return vel; } typedef MACGrid type2;inline Grid<Vec3>& getArg3() { return tmp; } typedef Grid<Vec3> type3;inline const ParticleDataImpl<Vec3>& getArg4() { return pvel; } typedef ParticleDataImpl<Vec3> type4;inline const ParticleDataImpl<int>* getArg5() { return ptype; } typedef ParticleDataImpl<int> type5;inline const int& getArg6() { return exclude; } typedef int type6; void runMessage() { debMsg("Executing kernel knMapLinearVec3ToMACGrid ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; for (IndexInt i = 0; i < _sz; i++) op(i, p,flags,vel,tmp,pvel,ptype,exclude);   } const BasicParticleSystem& p; const FlagGrid& flags; MACGrid& vel; Grid<Vec3>& tmp; const ParticleDataImpl<Vec3>& pvel; const ParticleDataImpl<int>* ptype; const int exclude;   };

// optionally , this function can use an existing vec3 grid to store the weights
// this is useful in combination with the simple extrapolation function



void mapPartsToMAC(const FlagGrid& flags, MACGrid& vel, MACGrid& velOld, const BasicParticleSystem& parts, const ParticleDataImpl<Vec3>& partVel, Grid<Vec3>* weight=NULL, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	// interpol -> grid. tmpgrid for particle contribution weights
	bool freeTmp = false;
	if(!weight) {
		weight = new Grid<Vec3>(flags.getParent());
		freeTmp = true;
	} else {
		weight->clear(); // make sure we start with a zero grid!
	}
	vel.clear();
	knMapLinearVec3ToMACGrid( parts, flags, vel, *weight, partVel, ptype, exclude );

	// stomp small values in weight to zero to prevent roundoff errors
	weight->stomp(Vec3(VECTOR_EPSILON));
	vel.safeDivide(*weight);
	
	// store original state
	velOld.copyFrom( vel );
	if(freeTmp) delete weight;
}




template <class T>  struct knMapLinear : public KernelBase { knMapLinear( BasicParticleSystem& p, FlagGrid& flags, Grid<T>& target, Grid<Real>& gtmp, ParticleDataImpl<T>& psource ) :  KernelBase(p.size()) ,p(p),flags(flags),target(target),gtmp(gtmp),psource(psource)   { runMessage(); run(); }   inline void op(IndexInt idx,  BasicParticleSystem& p, FlagGrid& flags, Grid<T>& target, Grid<Real>& gtmp, ParticleDataImpl<T>& psource  )  {
	unusedParameter(flags);
	if (!p.isActive(idx)) return;
	target.setInterpolated( p[idx].pos, psource[idx], gtmp );
}    inline BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline Grid<T>& getArg2() { return target; } typedef Grid<T> type2;inline Grid<Real>& getArg3() { return gtmp; } typedef Grid<Real> type3;inline ParticleDataImpl<T>& getArg4() { return psource; } typedef ParticleDataImpl<T> type4; void runMessage() { debMsg("Executing kernel knMapLinear ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; for (IndexInt i = 0; i < _sz; i++) op(i, p,flags,target,gtmp,psource);   } BasicParticleSystem& p; FlagGrid& flags; Grid<T>& target; Grid<Real>& gtmp; ParticleDataImpl<T>& psource;   }; 
template<class T>
void mapLinearRealHelper( FlagGrid& flags, Grid<T>& target , 
		BasicParticleSystem& parts , ParticleDataImpl<T>& source ) 
{
	Grid<Real> tmp(flags.getParent());
	target.clear();
	knMapLinear<T>( parts, flags, target, tmp, source ); 
	knSafeDivReal<T>( target, tmp );
}

void mapPartsToGrid( FlagGrid& flags, Grid<Real>& target , BasicParticleSystem& parts , ParticleDataImpl<Real>& source ) {
	mapLinearRealHelper<Real>(flags,target,parts,source);
}
void mapPartsToGridVec3( FlagGrid& flags, Grid<Vec3>& target , BasicParticleSystem& parts , ParticleDataImpl<Vec3>& source ) {
	mapLinearRealHelper<Vec3>(flags,target,parts,source);
}
// integers need "max" mode, not yet implemented
//PYTHON() void mapPartsToGridInt ( FlagGrid& flags, Grid<int >& target , BasicParticleSystem& parts , ParticleDataImpl<int >& source ) {
//	mapLinearRealHelper<int >(flags,target,parts,source);
//}



template <class T>  struct knMapFromGrid : public KernelBase { knMapFromGrid( BasicParticleSystem& p, Grid<T>& gsrc, ParticleDataImpl<T>& target ) :  KernelBase(p.size()) ,p(p),gsrc(gsrc),target(target)   { runMessage(); run(); }   inline void op(IndexInt idx,  BasicParticleSystem& p, Grid<T>& gsrc, ParticleDataImpl<T>& target  )  {
	if (!p.isActive(idx)) return;
	target[idx] = gsrc.getInterpolated( p[idx].pos );
}    inline BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline Grid<T>& getArg1() { return gsrc; } typedef Grid<T> type1;inline ParticleDataImpl<T>& getArg2() { return target; } typedef ParticleDataImpl<T> type2; void runMessage() { debMsg("Executing kernel knMapFromGrid ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,gsrc,target);  }   } BasicParticleSystem& p; Grid<T>& gsrc; ParticleDataImpl<T>& target;   };
#line 550 "plugin/flip.cpp"

 
void mapGridToParts( Grid<Real>& source , BasicParticleSystem& parts , ParticleDataImpl<Real>& target ) {
	knMapFromGrid<Real>(parts, source, target);
}
void mapGridToPartsVec3( Grid<Vec3>& source , BasicParticleSystem& parts , ParticleDataImpl<Vec3>& target ) {
	knMapFromGrid<Vec3>(parts, source, target);
}


// Get velocities from grid




 struct knMapLinearMACGridToVec3_PIC : public KernelBase { knMapLinearMACGridToVec3_PIC(BasicParticleSystem& p, FlagGrid& flags, MACGrid& vel, ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<int>* ptype, const int exclude) :  KernelBase(p.size()) ,p(p),flags(flags),vel(vel),pvel(pvel),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, BasicParticleSystem& p, FlagGrid& flags, MACGrid& vel, ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<int>* ptype, const int exclude )  {
	if (!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	// pure PIC
	pvel[idx] = vel.getInterpolated( p[idx].pos );
}    inline BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline MACGrid& getArg2() { return vel; } typedef MACGrid type2;inline ParticleDataImpl<Vec3>& getArg3() { return pvel; } typedef ParticleDataImpl<Vec3> type3;inline const ParticleDataImpl<int>* getArg4() { return ptype; } typedef ParticleDataImpl<int> type4;inline const int& getArg5() { return exclude; } typedef int type5; void runMessage() { debMsg("Executing kernel knMapLinearMACGridToVec3_PIC ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,flags,vel,pvel,ptype,exclude);  }   } BasicParticleSystem& p; FlagGrid& flags; MACGrid& vel; ParticleDataImpl<Vec3>& pvel; const ParticleDataImpl<int>* ptype; const int exclude;   };
#line 567 "plugin/flip.cpp"




void mapMACToParts(FlagGrid& flags, MACGrid& vel , BasicParticleSystem& parts , ParticleDataImpl<Vec3>& partVel, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	knMapLinearMACGridToVec3_PIC( parts, flags, vel, partVel, ptype, exclude );
}

// with flip delta interpolation 




 struct knMapLinearMACGridToVec3_FLIP : public KernelBase { knMapLinearMACGridToVec3_FLIP(const BasicParticleSystem& p, const FlagGrid& flags, const MACGrid& vel, const MACGrid& oldVel, ParticleDataImpl<Vec3>& pvel, const Real flipRatio, const ParticleDataImpl<int>* ptype, const int exclude) :  KernelBase(p.size()) ,p(p),flags(flags),vel(vel),oldVel(oldVel),pvel(pvel),flipRatio(flipRatio),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, const BasicParticleSystem& p, const FlagGrid& flags, const MACGrid& vel, const MACGrid& oldVel, ParticleDataImpl<Vec3>& pvel, const Real flipRatio, const ParticleDataImpl<int>* ptype, const int exclude )  {
	if (!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	Vec3 v     =        vel.getInterpolated(p[idx].pos);
	Vec3 delta = v - oldVel.getInterpolated(p[idx].pos); 
	pvel[idx] = flipRatio * (pvel[idx] + delta) + (1.0 - flipRatio) * v;    
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline const FlagGrid& getArg1() { return flags; } typedef FlagGrid type1;inline const MACGrid& getArg2() { return vel; } typedef MACGrid type2;inline const MACGrid& getArg3() { return oldVel; } typedef MACGrid type3;inline ParticleDataImpl<Vec3>& getArg4() { return pvel; } typedef ParticleDataImpl<Vec3> type4;inline const Real& getArg5() { return flipRatio; } typedef Real type5;inline const ParticleDataImpl<int>* getArg6() { return ptype; } typedef ParticleDataImpl<int> type6;inline const int& getArg7() { return exclude; } typedef int type7; void runMessage() { debMsg("Executing kernel knMapLinearMACGridToVec3_FLIP ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,flags,vel,oldVel,pvel,flipRatio,ptype,exclude);  }   } const BasicParticleSystem& p; const FlagGrid& flags; const MACGrid& vel; const MACGrid& oldVel; ParticleDataImpl<Vec3>& pvel; const Real flipRatio; const ParticleDataImpl<int>* ptype; const int exclude;   };
#line 583 "plugin/flip.cpp"





void flipVelocityUpdate(const FlagGrid& flags, const MACGrid& vel, const MACGrid& velOld, const BasicParticleSystem& parts, ParticleDataImpl<Vec3>& partVel, const Real flipRatio, const ParticleDataImpl<int>* ptype=NULL, const int exclude=0) {
	knMapLinearMACGridToVec3_FLIP( parts, flags, vel, velOld, partVel, flipRatio, ptype, exclude );
}


//******************************************************************************
// narrow band 


 struct knCombineVels : public KernelBase { knCombineVels(MACGrid& vel, Grid<Vec3>& w, MACGrid& combineVel, LevelsetGrid* phi, Real narrowBand, Real thresh ) :  KernelBase(&vel,0) ,vel(vel),w(w),combineVel(combineVel),phi(phi),narrowBand(narrowBand),thresh(thresh)   { runMessage(); run(); }  inline void op(int i, int j, int k, MACGrid& vel, Grid<Vec3>& w, MACGrid& combineVel, LevelsetGrid* phi, Real narrowBand, Real thresh  )  {
	int idx = vel.index(i,j,k);

	for(int c=0; c<3; ++c)
	{
			// Correct narrow-band FLIP
			Vec3 pos(i,j,k);
			pos[(c+1)%3] += Real(0.5);
			pos[(c+2)%3] += Real(0.5);
			Real p = phi->getInterpolated(pos);

			if (p < -narrowBand) { vel[idx][c] = 0; continue; }

			if (w[idx][c] > thresh) {
				combineVel[idx][c] = vel[idx][c];
				vel[idx][c] = -1;
			}
			else
			{
				vel[idx][c] = 0;
			}
	}
}   inline MACGrid& getArg0() { return vel; } typedef MACGrid type0;inline Grid<Vec3>& getArg1() { return w; } typedef Grid<Vec3> type1;inline MACGrid& getArg2() { return combineVel; } typedef MACGrid type2;inline LevelsetGrid* getArg3() { return phi; } typedef LevelsetGrid type3;inline Real& getArg4() { return narrowBand; } typedef Real type4;inline Real& getArg5() { return thresh; } typedef Real type5; void runMessage() { debMsg("Executing kernel knCombineVels ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,vel,w,combineVel,phi,narrowBand,thresh);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,vel,w,combineVel,phi,narrowBand,thresh);  } }  } MACGrid& vel; Grid<Vec3>& w; MACGrid& combineVel; LevelsetGrid* phi; Real narrowBand; Real thresh;   };
#line 601 "plugin/flip.cpp"



//! narrow band velocity combination

void combineGridVel( MACGrid& vel, Grid<Vec3>& weight, MACGrid& combineVel, LevelsetGrid* phi=NULL, Real narrowBand=0.0, Real thresh=0.0) {
	knCombineVels(vel, weight, combineVel, phi, narrowBand, thresh);
}

//! surface tension helper
void getLaplacian(Grid<Real> &laplacian, const Grid<Real> &grid) {
	LaplaceOp(laplacian, grid);
}


} // namespace



