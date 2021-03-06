




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/plugin/fire.cpp"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2016 Sebastian Barschkis, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * Fire modeling plugin
 *
 ******************************************************************************/

#include "general.h"
#include "grid.h"
#include "vectorbase.h"

using namespace std;

namespace Manta {






 struct KnProcessBurn : public KernelBase { KnProcessBurn(Grid<Real>& fuel, Grid<Real>& density, Grid<Real>& react, Grid<Real>* red, Grid<Real>* green, Grid<Real>* blue, Grid<Real>* heat, Real burningRate, Real flameSmoke, Real ignitionTemp, Real maxTemp, Real dt, Vec3 flameSmokeColor) :  KernelBase(&fuel,1) ,fuel(fuel),density(density),react(react),red(red),green(green),blue(blue),heat(heat),burningRate(burningRate),flameSmoke(flameSmoke),ignitionTemp(ignitionTemp),maxTemp(maxTemp),dt(dt),flameSmokeColor(flameSmokeColor)   { runMessage(); run(); }  inline void op(int i, int j, int k, Grid<Real>& fuel, Grid<Real>& density, Grid<Real>& react, Grid<Real>* red, Grid<Real>* green, Grid<Real>* blue, Grid<Real>* heat, Real burningRate, Real flameSmoke, Real ignitionTemp, Real maxTemp, Real dt, Vec3 flameSmokeColor )  {
	// Save initial values
	Real origFuel = fuel(i,j,k);
	Real origSmoke = density(i,j,k);
	Real smokeEmit = 0.0f;
	Real flame = 0.0f;
	
	// Process fuel
	fuel(i,j,k) -= burningRate * dt;
	if (fuel(i,j,k) < 0.0f)
		fuel(i,j,k) = 0.0f;
	
	// Process reaction coordinate
	if (origFuel > VECTOR_EPSILON) {
		react(i,j,k) *= fuel(i,j,k) / origFuel;
		flame = pow(react(i,j,k), 0.5f);
	} else {
		react(i,j,k) = 0.0f;
	}
	
	// Set fluid temperature based on fuel burn rate and "flameSmoke" factor
	smokeEmit = (origFuel < 1.0f) ? (1.0 - origFuel) * 0.5f : 0.0f;
	smokeEmit = (smokeEmit + 0.5f) * (origFuel - fuel(i,j,k)) * 0.1f * flameSmoke;
	density(i,j,k) += smokeEmit;
	clamp( density(i,j,k), (Real)0.0f, (Real)1.0f);
	
	// Set fluid temperature from the flame temperature profile
	if (heat && flame)
		(*heat)(i,j,k) = (1.0f - flame) * ignitionTemp + flame * maxTemp;
	
	// Mix new color
	if (smokeEmit > VECTOR_EPSILON) {
		float smokeFactor = density(i,j,k) / (origSmoke + smokeEmit);
		if(red)   (*red)(i,j,k)   = ((*red)(i,j,k)   + flameSmokeColor.x * smokeEmit) * smokeFactor;
		if(green) (*green)(i,j,k) = ((*green)(i,j,k) + flameSmokeColor.y * smokeEmit) * smokeFactor;
		if(blue)  (*blue)(i,j,k)  = ((*blue)(i,j,k)  + flameSmokeColor.z * smokeEmit) * smokeFactor;
	}
}   inline Grid<Real>& getArg0() { return fuel; } typedef Grid<Real> type0;inline Grid<Real>& getArg1() { return density; } typedef Grid<Real> type1;inline Grid<Real>& getArg2() { return react; } typedef Grid<Real> type2;inline Grid<Real>* getArg3() { return red; } typedef Grid<Real> type3;inline Grid<Real>* getArg4() { return green; } typedef Grid<Real> type4;inline Grid<Real>* getArg5() { return blue; } typedef Grid<Real> type5;inline Grid<Real>* getArg6() { return heat; } typedef Grid<Real> type6;inline Real& getArg7() { return burningRate; } typedef Real type7;inline Real& getArg8() { return flameSmoke; } typedef Real type8;inline Real& getArg9() { return ignitionTemp; } typedef Real type9;inline Real& getArg10() { return maxTemp; } typedef Real type10;inline Real& getArg11() { return dt; } typedef Real type11;inline Vec3& getArg12() { return flameSmokeColor; } typedef Vec3 type12; void runMessage() { debMsg("Executing kernel KnProcessBurn ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,fuel,density,react,red,green,blue,heat,burningRate,flameSmoke,ignitionTemp,maxTemp,dt,flameSmokeColor);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,fuel,density,react,red,green,blue,heat,burningRate,flameSmoke,ignitionTemp,maxTemp,dt,flameSmokeColor);  } }  } Grid<Real>& fuel; Grid<Real>& density; Grid<Real>& react; Grid<Real>* red; Grid<Real>* green; Grid<Real>* blue; Grid<Real>* heat; Real burningRate; Real flameSmoke; Real ignitionTemp; Real maxTemp; Real dt; Vec3 flameSmokeColor;   };
#line 27 "plugin/fire.cpp"









void processBurn(Grid<Real>& fuel, Grid<Real>& density, Grid<Real>& react, Grid<Real>* red = NULL, Grid<Real>* green = NULL, Grid<Real>* blue = NULL, Grid<Real>* heat = NULL, Real burningRate = 0.75f, Real flameSmoke = 1.0f, Real ignitionTemp = 1.25f, Real maxTemp = 1.75f, Real dt = 0.1f, Vec3 flameSmokeColor = Vec3(0.7f, 0.7f, 0.7f)) {
	KnProcessBurn(fuel, density, react, red, green, blue, heat, burningRate,
				  flameSmoke, ignitionTemp, maxTemp, dt, flameSmokeColor);
}




 struct KnUpdateFlame : public KernelBase { KnUpdateFlame(Grid<Real>& react, Grid<Real>& flame) :  KernelBase(&react,1) ,react(react),flame(flame)   { runMessage(); run(); }  inline void op(int i, int j, int k, Grid<Real>& react, Grid<Real>& flame )  {
	if (react(i,j,k) > 0.0f)
		flame(i,j,k) = pow(react(i,j,k), 0.5f);
	else
		flame(i,j,k) = 0.0f;
}   inline Grid<Real>& getArg0() { return react; } typedef Grid<Real> type0;inline Grid<Real>& getArg1() { return flame; } typedef Grid<Real> type1; void runMessage() { debMsg("Executing kernel KnUpdateFlame ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,react,flame);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=1; j < _maxY; j++) for (int i=1; i < _maxX; i++) op(i,j,k,react,flame);  } }  } Grid<Real>& react; Grid<Real>& flame;   };
#line 80 "plugin/fire.cpp"




void updateFlame(Grid<Real>& react, Grid<Real>& flame) {
	KnUpdateFlame(react, flame);
}

} // namespace


