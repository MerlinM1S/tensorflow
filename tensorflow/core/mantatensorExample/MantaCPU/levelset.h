




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/levelset.h"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Levelset
 *
 ******************************************************************************/

#ifndef _LEVELSET_H_
#define _LEVELSET_H_

#include "grid.h"

namespace Manta {
class Mesh;

//! Special function for levelsets

class LevelsetGrid : public Grid<Real> {public:
	LevelsetGrid(FluidSolver* parent, bool show = true);
	
	//! reconstruct the levelset using fast marching
	

void reinitMarching(const FlagGrid& flags, Real maxTime=4.0, MACGrid* velTransport=NULL, bool ignoreWalls=false, bool correctOuterLayer=true, int obstacleType = FlagGrid::TypeObstacle );

	//! create a triangle mesh from the levelset isosurface
	void createMesh(Mesh& mesh);
	
	//! union with another levelset
	void join(const LevelsetGrid& o);
	void subtract(const LevelsetGrid& o);
	
	//! initialize levelset from flags (+/- 0.5 heaviside)
	void initFromFlags(const FlagGrid& flags, bool ignoreWalls=false);
	 	static Real invalidTimeValue(); }
;

} //namespace
#endif


