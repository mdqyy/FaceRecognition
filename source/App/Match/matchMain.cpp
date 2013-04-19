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

		LoadParm(m, gf->featLenTotal, "featLenTotal");
		LoadParm(m, gf->featLenLBP, "featLenLBP");

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

		gf->featLenTotal = 1180;
		gf->featLenLBP = 1180;



	}



	




}

void main()
{
	gFaceReco		gf;
	config(&gf, "../../image/config.cfg");

}