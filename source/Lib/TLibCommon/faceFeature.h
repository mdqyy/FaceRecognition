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
#include "get_config.h"

void config(gFaceReco* gf, char* configFile);				//configuration
void initGlobalStruct(gFaceReco* gf);
void freeGlobalStruct(gFaceReco* gf);
void initOneFeature(featStruct* fst, gFaceReco* gf);
void freeOneFeature(featStruct* fst);

inline void resetHist(UInt* hist, int n);
void extractLBPFeatures(gFaceReco* gf);
void dumpFeatures(gFaceReco* gf, FILE* pFaceFeatBin);
void loadFeatures(gFaceReco* gf);
void convolution2D(UChar *src, float *dst, double *kernel, int size, int height, int width);
void extractGaborFeatures(gFaceReco* gf);
void extractIntensityFeatures(gFaceReco* gf);

int matchFaceID(gFaceReco* gf);
float matchFeatureHistDist(float* feature1, float* feature2, int length);

void copyOneFeatureToBuffer(gFaceReco* gf, int idx);
void extractReferDistFeatures(gFaceReco* gf);


#endif //_FACE_FEATURE_H_
