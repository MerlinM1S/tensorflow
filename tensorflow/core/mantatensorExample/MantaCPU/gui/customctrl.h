




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ansorge/workspace_master/manta/source/gui/customctrl.h"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * GUI extension from python
 *
 ******************************************************************************/

#ifndef _CUSTOMCTRL_H__
#define _CUSTOMCTRL_H__

#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QBoxLayout>
#include "manta.h"

namespace Manta {

// fwd decl.
class Mesh;
class GuiThread;
class MainThread;
	
//! Interface for python declared controls

class CustomControl : public PbClass {public:
	CustomControl();
	
	virtual void init(QBoxLayout* layout) {};
 protected: }
;

//! Checkbox with attached text display
class TextCheckbox : public QCheckBox {
	Q_OBJECT
public:
	TextCheckbox(const std::string& name, bool val);
	void attach(QBoxLayout* layout);
	void set(bool v);
	bool get();
	
public slots:
	void update(int v);
		
protected:
	bool mVal;
	QLabel* mLabel;    
	QString mSName;    
};

//! Slider with attached text display
class TextSlider : public QSlider {
	Q_OBJECT
public:
	TextSlider(const std::string& name, float val, float min, float max);
	void attach(QBoxLayout* layout);
	void set(float v);
	float get();
	
public slots:
	void update(int v);
		
protected:
	float mMin, mMax, mScale;
	QLabel* mLabel;    
	QString mSName;    
};
	
//! Links a slider control


class CustomSlider : public CustomControl {public:
	CustomSlider(std::string text, float val, float min, float max);
	virtual void init(QBoxLayout* layout);
	
	float get();
	void set(float v);
	
protected:
	float mMin, mMax, mVal;
	std::string mSName; 	TextSlider* mSlider; }
;

//! Links a checkbox control


class CustomCheckbox : public CustomControl {public:
	CustomCheckbox(std::string text, bool val);
	virtual void init(QBoxLayout* layout);
	
	bool get();
	void set(bool v);
	
protected:
	bool mVal;
	std::string mSName; 	TextCheckbox* mCheckbox; }
;
	

//! GUI adapter class to call from Python

class Gui : public PbClass {public:
	Gui();
	
	void setBackgroundMesh(Mesh* m);
	void show(bool twoD=false);
	void update();
	void pause();
	PbClass* addControl(PbType t);
	void screenshot(std::string filename);

	// control display upon startup
	void nextRealGrid();
	void nextVec3Grid();
	void nextParts();
	void nextPdata();
	void nextMesh();
	void nextVec3Display();
	void nextMeshDisplay();
	void nextPartDisplay(); 
	void toggleHideGrids();
	void setCamPos(float x, float y, float z);
	void setCamRot(float x, float y, float z);  
	void windowSize(int w, int h);
	
protected:
	GuiThread* mGuiPtr; 	MainThread* mMainPtr; }
;
	
} // namespace

#endif



