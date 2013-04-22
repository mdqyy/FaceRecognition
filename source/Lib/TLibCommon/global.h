/***
* Program Name: faceRecognition
*
* Script File: global.h
*
* Description:
*  
*  Define system parameters and global structures
*   
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/

#ifndef __GLOBAL_H__
#define __GLOBAL_H__


#define MAX_FLOAT_NUMBER        3.402823466e+38F //< maximum single float number
#define PI						3.1415926535898


//typedef
typedef unsigned int	UInt;
typedef	unsigned char	UChar;



typedef	struct unitFeatStructure
{
	int		id;				//face id

	float*	featLBP;		//LBP features
	float*	featGabor;		//Gabor features
	float*	featIntensity;	//Intensity features
}featStruct;

typedef	struct imagePathStructure
{
	int		id;
	char	path[260];		//reletive path
}pathStruct;


//global structure for Face Recogniton
typedef struct globalStructure
{
	//global switchs
	bool	bUseLBP;		//use LBP features
	bool	bUseGabor;		//use Gabor features
	bool	bUseIntensity;	//use Intensity features
	bool	bUseHOG;		//use HOG features
	bool	bUseCA;			//use correlation angles

	bool	bUseWeight;		//use weight control
	bool	bFlipMatch;		//use flip match
	bool	bHistEqu;		//use histogram equalization
	bool	bUniformLBP;	//use uniform LBP
	bool	bChiDist;		//use chi-square distance

	bool	bOverWriteBin;	//overwrite feature binary file

	//feature parameters
	int		featLenTotal;	//overall feature length
	int		featLenLBP;		//LBP feature length
	int		featLenGabor;	
	int		featLenIntensity;
	int		featLenHOG;
	int		featLenCA;

	//Face alignment
	int		faceWidth;		//Crop face width
	int		faceHeight;
	int		faceWidth1;		//downsampled face width
	int		faceHeight1;
	int		faceWidth2;
	int		faceHeight2;
	int		faceRegion0X;	//face ROI top left x
	int		faceRegion0Y;	//face ROI top left y
	int		faceRegion1X;
	int		faceRegion1Y;
	int		leftEyeX;		//assigned left eye x coordinate
	int		leftEyeY;		//assigned left eye y coordinate
	int		rightEyeX;		//assigned right eye x coordinate
	int		rightEyeY;		//assigned right eye y coordinate
	int		actLeftEyeX;	//actual left eye x
	int		actLeftEyeY;
	int		actRightEyeX;
	int		actRightEyeY;
	int		faceChannel;	//number of face image channels: 1 - gray, 3 - color


	//LBP
	int		numBinsLBP;		//histogram bins of LBP
	int		numHistsLBP;	//num of histograms of LBP
	int*	uniTableLBP;	//look up table for uniformLBP
	int		LBPStepW;		//LBP width step
	int		LBPStepH;		//LBP height step
	int		LBPWindowW;		//LBP window width
	int		LBPWindowH;
	UInt*	LBPHist;		//LBP histogram
	int		LBPNeighBorThreshold;	//LBP neighboor threshold

	//data
	UChar*	face;			//original face data
	UChar*	face1;			//downsampled face data
	UChar*	face2;

	featStruct	features;	//feature struct
	featStruct*	loadedFeatures;	//loaded features from trained binary file
	pathStruct*	imageList;	//image list
	int		maxNumImages;	//max number of input images
	int		numImageInList;	//number of images in the input list
	int		numValidFaces;	//number of detected face images
	int		numLoadedFaces;	//number of loaded faces from binary file

	float*	weight;			//features weights

	

	//paths
	char	trainImageDir[260];		//train image directory
	char	faceBinPath[260];		//face.bin 
	char	imageTagDir[260];		//image tags dir
	char	weightBinPath[260];		//weight bin
	char	svmListDir[260];

	char	matchImageDir[260];		//match image dir
	char	resultTxtPath[260];		//result text file

	char	gaborBinPath[260];		//gabor kernel bin
	
	char	cameraCaptureDir[260];	//camera capture


	//limitations
	int		maxFaceTags;			//max face tags
	int		trainStartID;			//train start ID
	int		trainEndID;
	







}gFaceReco;


#endif //__GLOBAL_H__


////////////// faceCNN
#ifndef _CV_GLOBAL_H_
#define _CV_GLOBAL_H_


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


#endif // _CV_GLOBAL_H_