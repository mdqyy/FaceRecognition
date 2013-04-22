/**************************************************************************
* Program Name: faceRecognition
*
* Script File: cvFaceFeature.h
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

#ifndef _CV_FACE_FEATURE_H_
#define _CV_FACE_FEATURE_H_




#include "global.h"
#include "globalCV.h"

void initGlobalCVStruct(gFaceRecoCV* gcv, gFaceReco* gf);		//init
void freeGlobalCVStruct(gFaceRecoCV* gcv, gFaceReco* gf);		//free

bool runFaceAndEyesDetect(IplImage* pFrame, gFaceReco* gf, gFaceRecoCV* gcv);	//return 1 if face detected, 0 otherwise
void faceAlign(IplImage* src, IplImage* dst, gFaceReco* gf);	//face alignment

void cameraCapture(gFaceReco* gf, gFaceRecoCV* gcv);
void processTrainInput(gFaceReco* gf, gFaceRecoCV* gcv);		//train input 

void train(gFaceReco* gf, gFaceRecoCV* gcv);



#endif //_CV_FACE_FEATURE_H_
