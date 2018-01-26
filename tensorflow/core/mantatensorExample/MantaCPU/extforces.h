
#ifndef _EXT_FORCES_H
#define _EXT_FORCES_H

#include "grid.h"

namespace Manta {
    
//! add Buoyancy force based on fctor (e.g. smoke density)
void addBuoyancy(const FlagGrid& flags, const Grid<Real>& density, MACGrid& vel, Vec3 gravity, Real coefficient=1.);

}

#endif // -- _EXT_FORCES_H