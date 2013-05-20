/**
* Program Name: faceRecognition
*
* Script File: matchMain.cpp
*
* Face Recognition testing phase
*
**/


#include <stdio.h>
#include "TLibCommon/global.h"
#include "TLibCommon/cvFaceFeature.h"
#include "TLibCommon/faceFeature.h"

int    NSAMPLES = 1;
int    MAX_ITER = 1;
int    NTESTSAMPLES = 1;

using namespace std;



void main()
{
	//initilization
	gFaceReco		gf;
	gFaceRecoCV		gcv;
	gf.bIsTraining = 0;
	gf.bIsMatching = 1;
	config(&gf, "../../image/config.cfg");
	initGlobalStruct(&gf);
	initGlobalCVStruct(&gcv, &gf);



	//--------------------------------------------------//
	//To do
	if (gf.bUseLFW)
		testLFWVerification(&gf, &gcv);
	else
		match(&gf, &gcv);


	//--------------------------------------------------//





	//clean-ups
	freeGlobalStruct(&gf);
	freeGlobalCVStruct(&gcv, &gf);


	system("pause");

}