/**************************************************************************
* Program Name: iVAS
*
* Filename: define.h
*
* Description:
*  
*
*  Define system parameters and media format.
*   
*  
*
* Copyright (C) 2011-2012.
* All Rights Reserved.
**************************************************************************/

#ifndef DEFINE_H_INCLUDED
#define DEFINE_H_INCLUDED

#define	FACE_DATUM_FILENAME "../../image/faces.bin"

#define NUM_BIN       256 //number of bins in histogram
#define UNIFORM_LBP   0  //use 58 uniform LBP patterns to be statistically efficient 
#define THRESHOLD	  0  //use threshold to reduce local noise, if set to 0 means no threshold
#define KAI_DISTANCE  0	 //use normalized distance for histogram
#define WEIGHTED_MATCH  0 //weighted matching faces within each class

#include "global.h"


void init(FACE3D_Type * gf, int width, int height);
void initFaceFeature(FACE3D_Type * gf, int width, int height);
void face3DAnalysis(unsigned char * imageData, int widthStep, FACE3D_Type * gf);

void videoAnalysis(unsigned char * imageData, int widthStep, FACE3D_Type * gf);
void extractFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf);
void extractLBPFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf, bool isFlip);
void extractGBPFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf);
void extractGaborFeatures(FACE3D_Type* gf);

void loadFaceData( FACE3D_Type * gf );
int	 matchFace(FACE3D_Type * gf );

#endif //DEFINE_H_INCLUDED
