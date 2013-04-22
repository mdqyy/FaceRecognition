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
#include "TLibCommon/get_config.h"

int    NSAMPLES = 1;
int    MAX_ITER = 1;
int    NTESTSAMPLES = 1;


using namespace std;


void config(gFaceReco* gf, char* configFile)
{

	map<string, string> m; 
	if ( ReadConfig(configFile, m))   // Read parameters from config file
	{
		//load parameters from config file
		LoadParmBool(m, gf->bUseLBP, "UseLBP");
		LoadParmBool(m, gf->bUseGabor, "UseGabor");
		LoadParmBool(m, gf->bUseIntensity, "UseIntensity");
		LoadParmBool(m, gf->bUseHOG, "UseHOG");
		LoadParmBool(m, gf->bUseCA, "UseCA");
		LoadParmBool(m, gf->bUseWeight, "UseWeight");
		LoadParmBool(m, gf->bFlipMatch, "FlipMatch");
		LoadParmBool(m, gf->bHistEqu, "HistEqu");
		LoadParmBool(m, gf->bUniformLBP, "UniformLBP");
		LoadParmBool(m, gf->bChiDist, "ChiDist");
		LoadParmBool(m, gf->bOverWriteBin, "OverWriteBin");

		//limits
		LoadParm(m, gf->maxFaceTags, "maxFaceTags");
		LoadParm(m, gf->maxNumImages, "maxNumImages");
		LoadParm(m, gf->trainStartID, "trainStartID");
		LoadParm(m, gf->trainEndID, "trainEndID");


		//face
		LoadParm(m, gf->faceWidth, "faceWidth");
		LoadParm(m, gf->faceHeight, "faceHeight");
		LoadParm(m, gf->leftEyeX, "leftEyeX");\
		LoadParm(m, gf->leftEyeY, "leftEyeY");
		LoadParm(m, gf->rightEyeX, "rightEyeX");
		LoadParm(m, gf->rightEyeY, "rightEyeY");
		LoadParm(m, gf->faceChannel, "faceChannel");
		//LBP
		
		LoadParm(m, gf->numHistsLBP, "numHistsLBP");
		LoadParm(m, gf->LBPStepW, "LBPStepW");
		LoadParm(m, gf->LBPStepH, "LBPStepH");
		LoadParm(m, gf->LBPWindowW, "LBPWindowW");
		LoadParm(m, gf->LBPWindowH, "LBPWindowH");
		LoadParm(m, gf->LBPNeighBorThreshold, "LBPNeighBorThreshold");
		

	}
	else
	{
		printf("Error loading config, will use default parameters.../n");
		gf->bUseLBP = 1;
		gf->bUseGabor = 0;
		gf->bUseIntensity = 0;
		gf->bUseHOG = 0;
		gf->bUseCA = 0;
		gf->bUseWeight = 0;
		gf->bFlipMatch = 0;
		gf->bHistEqu = 1;
		gf->bUniformLBP = 1;
		gf->bChiDist = 1;
		gf->bOverWriteBin = 1;

		//limits
		gf->maxFaceTags = 300;
		gf->maxNumImages = 1500;
		gf->trainStartID = 1;
		gf->trainEndID = 60;


		//face
		gf->faceWidth = 80;
		gf->faceHeight = 80;
		gf->leftEyeX = 24;
		gf->leftEyeY = 25;
		gf->rightEyeX = 56;
		gf->rightEyeY = 25;
		gf->faceChannel = 3;
		//LBP
		gf->numHistsLBP = 20;
		gf->LBPStepW = 10;
		gf->LBPStepH = 10;
		gf->LBPWindowW = 10;
		gf->LBPWindowH = 10;
		gf->LBPNeighBorThreshold = 0;





	}

	//rest 
	sprintf(gf->trainImageDir,"%s", "../../image/train/");
	sprintf(gf->faceBinPath,"%s", "../../image/faces.bin");
	sprintf(gf->imageTagDir,"%s", "../../image/ImgTag/");
	sprintf(gf->weightBinPath,"%s", "../../image/weight.bin");
	sprintf(gf->svmListDir,"%s", "../../image/svm/");
	sprintf(gf->matchImageDir,"%s", "../../image/match/");
	sprintf(gf->resultTxtPath,"%s", "../../image/matchResult.txt");
	sprintf(gf->gaborBinPath,"%s", "../../image/gabor.bin");
	sprintf(gf->cameraCaptureDir, "%s", "../../image/cameraCapture/");

	gf->faceWidth1 = gf->faceWidth / 2;
	gf->faceWidth2 = gf->faceWidth / 4;
	gf->faceHeight1 = gf->faceHeight / 2;
	gf->faceHeight2 = gf->faceHeight / 4;

	if ( gf->bUniformLBP)
	{
		gf->numBinsLBP = 59;
	}
	else
	{
		gf->numBinsLBP = 256;
	}

	//feature length
	gf->featLenTotal = 0;
	if ( gf->bUseLBP)
	{
		gf->featLenLBP = gf->numBinsLBP * gf->numHistsLBP;
		gf->featLenTotal += gf->featLenLBP;
	}





	




}

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