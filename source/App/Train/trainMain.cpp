/**
* Program Name: faceRecognition
*
* Script File: trainMain.cpp
*
* Face Recognition Training main function and config
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
	config(&gf, "../../image/config.cfg");
	initGlobalStruct(&gf);
	initGlobalCVStruct(&gcv, &gf);



	//--------------------------------------------------//
	//To do
	train(&gf, &gcv);


	//--------------------------------------------------//





	//clean-ups
	freeGlobalStruct(&gf);
	freeGlobalCVStruct(&gcv, &gf);


	system("pause");

}