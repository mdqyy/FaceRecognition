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

#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#define USE_LBP				1
#define USE_GBP				0
#define USE_GABOR			0
#define NUM_NEAREST_NBOR	5

#define FACE_FEATURE_LEN	5120
#define TOTAL_FEATURE_LEN   5120

#define LBP_STEP	10
#define LBP_WINDOW	10



#define MAX_FACE_ID			100

#define DEBUG_OUTPUT_ALIGNED 0
#define DEBUG_MODE			0
#define FLIP_MATCH			0
#define HISTOGRAM_EQUALIZATION 1
#define ROTATE_INVARIANT_LBP 0


typedef struct face3DTag
{
	int FRAME_WIDTH;
	int FRAME_HEIGHT;

	unsigned char * mask;

	int gaborWSize;					//size of the Gabor filter
	int nGabors;					//num of Gabor filters
	double ** gaborCoefficients;

	int gwStep;						//stepSize of the shifting Gabor windows
	int LBP_H_Step;
	int LBP_W_Step;
	int LBP_H_Window;
	int LBP_W_Window;
	int * LBPHist;

	int RX0;						//the region of interest for face
	int RY0;
	int RX1;
	int RY1;

	int tWidth;						//normalized ROI dimemsions
	int tHeight;

	int * fImage0;					//original face ROI		256*192	
	int * fImage1;					//down-sampled by 2;	128*96
	int * fImage2;					//down-sampled by 4		64*48
#if FLIP_MATCH
	int *fImage0flip;
	int *fImage1flip;
	int *fImage2flip;
	float *faceFeaturesFlip;
#endif

	float * faceFeatures;
	int featurePtr;
	int numIDtag;                   //number of tagged IDs 2013.2.20

	int				featureLength;	// Face matching via Nearest Neighbors Vote. 2013.01.21
	int				bufFaceDataLen;
	unsigned char	*bufferFaceData;
	float			featDistance[NUM_NEAREST_NBOR];
	int				usedDistFlag[NUM_NEAREST_NBOR];
	int				bestDistID[NUM_NEAREST_NBOR];
	int				voteCntFaceID[MAX_FACE_ID];

#if DEBUG_MODE
	char		bestDistImageName[NUM_NEAREST_NBOR+1][200];
#endif

#if ROTATE_INVARIANT_LBP
	int		lookupTable[256];
#endif
	

}FACE3D_Type;

typedef struct unitFaceFeatClass
{
	int		id;
	float	feature[TOTAL_FEATURE_LEN];
#if DEBUG_MODE
	char	imagename[200];
#endif



}unitFaceFeatClass;

#endif //DEFINE_H_INCLUDED


////////////// faceCNN
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <stdio.h>
#include "cv.h"

using namespace std;

typedef struct eye_Feature
{
	CvPoint leftCorner;
	CvPoint rightCorner;
	CvPoint ball;
} eyeFeature;

typedef struct eyes
{
	eyeFeature LE;
	eyeFeature RE;
	int     Length;
} eyesInfo;



//#define eyeTopPad 25
//#define eyeBottomPad 75
//
//#define eyeSidePad 29
//#define NORMALEYELENGTH 42
//
//
//#define FACECLIPWIDTH 31
//#define FACECLIPHEIGHT 38
//#define FACECLIPSIZE   1178

//05/09 5:30pm
//#define eyeTopPad 25
//#define eyeBottomPad 75
//
//#define eyeSidePad 23
//#define NORMALEYELENGTH 54


#define eyeTopPad 25
#define eyeBottomPad 75

#define eyeSidePad 26
#define NORMALEYELENGTH 48

//#define eyeSidePad 22
//#define NORMALEYELENGTH 56


//#define eyeSidePad 30
//#define NORMALEYELENGTH 40


//#define FACECLIPWIDTH 132
//#define FACECLIPHEIGHT 132
//#define FACECLIPSIZE   17424

#define FACECLIPWIDTH 120
#define FACECLIPHEIGHT 120
#define FACECLIPSIZE   14400


#define LDAFACECLIPWIDTH 60
#define LDAFACECLIPHEIGHT 60
#define LDAFACECLIPSIZE   3600


#define CNNFACECLIPWIDTH  33
#define CNNFACECLIPHEIGHT 33
#define CNNFACECLIPSIZE   1089



#define LEARNINGRATE 0.034f
//#define MAX_ITER  2000
#define MUL_ITER   60
#define NNODE 12

#define CONNECTIONNODE 120

extern int    NSAMPLES;
extern int    MAX_ITER;

typedef vector<string> file_lists;

#ifdef UCHAR
#undef UCHAR
#endif
#define UCHAR unsigned char

// Camera parameters
typedef struct Camera_Para
{
	int Width;
	int Height;

	int Width_blk;
	int Height_blk;

	int Captured;
} CameraPara;


#endif


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

#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED


typedef struct face3DTag
{
	int FRAME_WIDTH;
	int FRAME_HEIGHT;

	unsigned char * mask;

	int gaborWSize;					//size of the Gabor filter
	int nGabors;					//num of Gabor filters
	double ** gaborCoefficients;

	int gwStep;						//stepSize of the shifting Gabor windows
	int LBP_H_Step;
	int LBP_W_Step;
	int * LBPHist;

	int RX0;						//the region of interest for face
	int RY0;
	int RX1;
	int RY1;

	int tWidth;						//normalized ROI dimemsions
	int tHeight;

	int * fImage0;					//original face ROI		256*192	
	int * fImage1;					//down-sampled by 2;	128*96
	int * fImage2;					//down-sampled by 4		64*48

	float * faceFeatures;
	int featurePtr;


}FACE3D_Type;


#endif //DEFINE_H_INCLUDED