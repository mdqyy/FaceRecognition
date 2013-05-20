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
#include "TLibRankSVM/train.h"

void config(gFaceReco* gf, char* configFile);				//configuration
void initGlobalStruct(gFaceReco* gf);
void freeGlobalStruct(gFaceReco* gf);
void initOneFeature(featStruct* fst, gFaceReco* gf);
void freeOneFeature(featStruct* fst);

inline void resetHist(UInt* hist, int n);
void shuffle(int *list, int n);

void extractLBPFeatures(gFaceReco* gf);
void dumpFeatures(gFaceReco* gf, FILE* pFaceFeatBin);
void loadFeatures(gFaceReco* gf);
void convolution2D(UChar *src, float *dst, double *kernel, int size, int height, int width);
void extractGaborFeatures(gFaceReco* gf);
void extractIntensityFeatures(gFaceReco* gf);

int matchFaceID(gFaceReco* gf);
float matchFeatureDist(float* feature1, float* feature2, int length);
float matchFeatureHistDist(float* feature1, float* feature2, int length);
int	matchFaceWhiteList(gFaceReco* gf);
int matchFaceIDVerification(gFaceReco* gf, FILE* pDebug);

void copyOneFeatureToBuffer(gFaceReco* gf, int idx);
void extractReferDistFeatures(gFaceReco* gf, FILE* pFaceFeatBin); // for train only
void extractReferDistFeaturesInMatch(gFaceReco* gf); // for match only
void extractAbsDist(gFaceReco* gf, featStruct* feature1, featStruct* feature2, float* dist);

void svmTraining(float ** features, int nSample, int featureSize, int * sampleLabel, char * modelFileName, float c);
void trainmodel(char*docfile,char* modelfile );
void test(char *docfile,char*modelfile);	//svm test

bool isInList(int* list, int listLength, int queryID);
int	getVotePool(int* pool, int min, int max, int size);
void trainOneToRestModels(gFaceReco* gf, int id, int* whiteList, int sizeList);
float matchOneInList(gFaceReco* gf, int id);
void svmTest(float ** features, int nSample, int featureSize, int * sampleLabel, char * modelFileName);


#endif //_FACE_FEATURE_H_
