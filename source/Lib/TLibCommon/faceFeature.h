/**************************************************************************
* Program Name: faceRecognition
*
* Script File: faceFeature.h
*
* Description:
*  
*
*  Features initialization and processing
*   
*  
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
**************************************************************************/

#ifndef _FACE_FEATURE_H_
#define _FACE_FEATURE_H_

#include "global.h"


void initGlobalStruct(gFaceReco* gf);
void freeGlobalStruct(gFaceReco* gf);
void initOneFeature(featStruct* fst, gFaceReco* gf);
void freeOneFeature(featStruct* fst);

inline void resetLBPHist(UInt* hist, int n);
void extractLBPFeatures(gFaceReco* gf);
void dumpFeatures(gFaceReco* gf, bool bAdd);
void loadFeatures(gFaceReco* gf);


#endif //_FACE_FEATURE_H_
