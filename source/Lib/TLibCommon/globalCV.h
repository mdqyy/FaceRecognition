/***
* Program Name: faceRecognition
*
* Script File: globalCV.h
*
* Description:
*  
*  Define system parameters and global structures using CV based class
*   
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/

#ifndef __GLOBALCV_H__
#define __GLOBALCV_H__


#include "TLibCommon\faceDetector.h"
#include "TLibCommon\EyesDetector.h"


typedef struct globalCVStruct
{
	faceDetector*	faceDet;
	eyesDetector*	eyeDet;
	IplImage*		gray_face_CNN;
	CvPoint			pointPos[6];
	IplImage*		warpedImg;
	IplImage**		faceTags;
}gFaceRecoCV;


#endif // _CV_GLOBALCV_H_