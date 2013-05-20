/**************************************************************************
* Program Name: faceRecognition
*
* Script File: faceFeature.cpp
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


#include "faceFeature.h"
#include "kmean.h"
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

/* global configuration */
void config(gFaceReco* gf, char* configFile)
{

	map<string, string> m; 
	if ( ReadConfig(configFile, m))   // Read parameters from config file
	{
		//load parameters from config file
		LoadParmBool(m, gf->bVerification, "bVerification");
		LoadParmBool(m, gf->bUseReferDist, "UseReferDist");
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
		LoadParmBool(m, gf->bUseAllSamples, "bUseAllSamples");
		LoadParmBool(m, gf->bWhiteList, "bWhiteList");
		LoadParmBool(m, gf->bUseBlockDist, "UseBlockDist");
		LoadParmBool(m, gf->bUseBlockCS, "UseBlockCS");
		LoadParmBool(m, gf->bUseLFW, "UseLFW");

		//limits
		LoadParm(m, gf->maxFaceTags, "maxFaceTags");
		LoadParm(m, gf->maxNumImages, "maxNumImages");
		LoadParm(m, gf->trainStartID, "trainStartID");
		LoadParm(m, gf->trainEndID, "trainEndID");


		//face
		LoadParm(m, gf->faceWidth, "faceWidth");
		LoadParm(m, gf->faceHeight, "faceHeight");
		LoadParm(m, gf->leftEyeX, "leftEyeX");
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

		//Gabor
		LoadParm(m, gf->numBinsGabor, "numBinsGabor");
		LoadParm(m, gf->numHistsGabor, "numHistsGabor");
		LoadParm(m, gf->gaborStepW, "gaborStepW");
		LoadParm(m, gf->gaborStepH, "gaborStepH");
		LoadParm(m, gf->gaborWindowW, "gaborWindowW");
		LoadParm(m, gf->gaborWindowH, "gaborWindowH");
		LoadParm(m, gf->gaborNeighBorThreshold, "gaborNeighBorThreshold");

		//Intensity
		LoadParm(m, gf->numBinsIntensity, "numBinsIntensity");
		LoadParm(m, gf->numHistsIntensity, "numHistsIntensity");
		LoadParm(m, gf->IntensityStepW, "IntensityStepW");
		LoadParm(m, gf->IntensityStepH, "IntensityStepH");
		LoadParm(m, gf->IntensityWindowW, "IntensityWindowW");
		LoadParm(m, gf->IntensityWindowH, "IntensityWindowH");

		//ReferDist
		LoadParm(m, gf->featLenReferDist, "featLenReferDist");

		//SVM
		LoadParm(m, gf->svmNumClasses, "svmNumClasses");
		LoadParm(m, gf->svmNumSamples, "svmNumSamples");
		LoadParm(m, gf->svmInterIntraRatio, "svmInterIntraRatio");
		LoadParm(m, gf->magicNumber, "magicNumber");
		LoadParm(m, gf->bias, "bias");



		

	}
	else
	{
		printf("Error loading config, will use default parameters.../n");
		gf->bUseReferDist = 1;
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
		gf->bWhiteList = 0;
		gf->bUseBlockDist = 0;
		gf->bUseBlockCS = 0;
		gf->bUseLFW = 0;

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

		//gabor
		gf->numBinsGabor = 16;
		gf->numHistsGabor = 320;
		gf->gaborStepW = 10;
		gf->gaborStepH = 10;
		gf->gaborWindowW = 10;
		gf->gaborWindowH = 10;
		gf->gaborNeighBorThreshold = 0;

		//Intensity
		gf->numBinsIntensity = 16;
		gf->numHistsIntensity = 20;
		gf->IntensityStepW = 20;
		gf->IntensityStepH = 20;
		gf->IntensityWindowW = 20;
		gf->IntensityWindowH = 20;

		//ReferDist
		gf->featLenReferDist = 200;

		//SVM
		gf->svmNumClasses = 2;
		gf->svmNumSamples = 3000;
		gf->svmInterIntraRatio = 5;
		gf->bUseAllSamples = 0;
		gf->magicNumber = 1;
		gf->bias = -1;






	}

	//path 
	sprintf(gf->trainImageDir,"%s", "../../image/train/");
	sprintf(gf->faceBinPath,"%s", "../../image/faces.bin");
	sprintf(gf->imageTagDir,"%s", "../../image/ImgTag/");
	sprintf(gf->weightBinPath,"%s", "../../image/weight.bin");
	sprintf(gf->svmListDir,"%s", "../../image/svm/");
	sprintf(gf->matchImageDir,"%s", "../../image/match/");
	sprintf(gf->resultTxtPath,"%s", "../../image/matchResult.txt");
	sprintf(gf->gaborBinPath,"%s", "../../image/gabor.bin");
	sprintf(gf->cameraCaptureDir, "%s", "../../image/cameraCapture/");
	sprintf(gf->referCentersPath, "%s", "../../image/referCenters.bin");
	sprintf(gf->svmModelPath, "%s", "../../image/svmModel.model");
	sprintf(gf->svmModelDir, "%s", "../../image/models/");
	sprintf(gf->lfwDir, "%s", "../../image/lfw/");
	sprintf(gf->lfwPairsTrain, "%s", "../../image/lfw/pairsDevTrain.txt");
	sprintf(gf->lfwPairsTest, "%s", "../../image/lfw/pairsDevTest.txt");

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
	gf->featLenLBP	 = 0;
	gf->featLenIntensity = 0;
	gf->featLenGabor = 0;
	gf->featLenHOG   = 0;
	if ( gf->bUseLBP)
	{
		gf->featLenLBP = gf->numBinsLBP * gf->numHistsLBP;
		gf->featLenTotal += gf->featLenLBP;
	}

	if ( gf->bUseGabor)
	{
		gf->featLenGabor = gf->numBinsGabor * gf->numHistsGabor;
		gf->featLenTotal += gf->featLenGabor;
	}

	if ( gf->bUseIntensity)
	{
		gf->featLenIntensity = gf->numBinsIntensity * gf->numHistsIntensity;
		gf->featLenTotal += gf->featLenIntensity;
	}

	if ( gf->bUseReferDist)
	{
		gf->featLenTotal = gf->featLenReferDist;	//intermediate feature replaces the exist ones
	}



	if ( gf->bWhiteList)
	{
		gf->bUseAllSamples = 1;
	}

	if ( gf->bUseBlockDist || gf->bUseBlockCS)
	{
		gf->svmFeatureLen = 0;
		if ( gf->bUseLBP)
			gf->svmFeatureLen += gf->numHistsLBP;
		if ( gf->bUseGabor)
			gf->svmFeatureLen += gf->numHistsGabor;
		if ( gf->bUseIntensity)
			gf->svmFeatureLen += gf->numHistsIntensity;
	}
	else
	{
		gf->svmFeatureLen = gf->featLenTotal;
	}

	




}//end config


/* initialize global struct */
void initGlobalStruct(gFaceReco* gf)
{
	gf->face = (UChar*)malloc(sizeof(int) * gf->faceWidth * gf->faceHeight);
	gf->face1 = (UChar*)malloc(sizeof(int) * gf->faceWidth1 * gf->faceHeight1);
	gf->face2 = (UChar*)malloc(sizeof(int) * gf->faceWidth2 * gf->faceHeight2);

	initOneFeature(&(gf->features), gf);

	if ( gf->bUseLBP)
	{
		gf->LBPHist = (UInt*)malloc(sizeof(UInt) * gf->numBinsLBP);
	}
	else
	{
		gf->LBPHist = NULL;
	}

	if ( gf->bUniformLBP)
	{
		gf->uniTableLBP = (int*)malloc(sizeof(int) * 256);
		int		base[8] = { 1, 2, 4, 8, 16, 32, 64, 128};
		int		i, j, k, pos;
		int		tmp;
		int		projBin;
		int*	tab = gf->uniTableLBP;

		//init
		for ( i = 0; i < 256; i++)
		{
			tab[i] = 58;
		}
		tab[0] = 0;
		tab[255] = 1;

		projBin = 2;
		for ( i = 1; i < 8; i++)
		{
			//num of 1s from 1 to 7
			
			for ( pos = 0; pos < 8; pos++)
			{
				k = pos;
				tmp = 0;
				for ( j = 0; j < i; j++)
				{
					k = k % 8;
					tmp += base[k];
					k++;
				}
				tab[tmp] = projBin++;
			}
		}
	}
	else
	{
		gf->uniTableLBP = NULL;
	}

	if ( gf->bUseWeight)
	{
		gf->weight = (float*)malloc(sizeof(float) * gf->featLenTotal);
	}
	else
	{
		gf->weight = NULL;
	}

	gf->loadedFeatures = NULL;

	gf->imageList = (pathStruct*)malloc(sizeof(pathStruct) * gf->maxNumImages);

	
	if ( gf->bUseGabor)
	{
		FILE*		pGaborBin;
		errno_t		err;
		int			k,m;
		double*		ptr;

		err = fopen_s(&pGaborBin, gf->gaborBinPath, "rb");
		if (err != 0)
		{
			printf("Can't open gabor binary file to read!\n");
			system("pause");
			exit(-1);
		}
		fread(&(gf->nGabors),sizeof(int),1,pGaborBin);
		fread(&(gf->gaborSize), sizeof(int),1,pGaborBin);
		gf->gaborCoefs = (double**)malloc( gf->nGabors * 2 * sizeof(double*));

		//read real and imaginary parts
		for ( k = 0; k < 2 * gf->nGabors; k++)
		{
			gf->gaborCoefs[k] = (double*)malloc(gf->gaborSize * gf->gaborSize * sizeof(double));
			ptr = gf->gaborCoefs[k];
			for ( m = 0; m < gf->gaborSize * gf->gaborSize; m++)
			{
				fread(&ptr[m], sizeof(double), 1, pGaborBin);
			}
		}
		fclose(pGaborBin);

		gf->GaborHist = (UInt*)malloc(sizeof(UInt) * gf->numBinsGabor);
		gf->gaborResponse = (float*)malloc(sizeof(float)* gf->faceWidth1 * gf->faceHeight1);
		gf->gaborResponseReal = (float*)malloc(sizeof(float)* gf->faceWidth1 * gf->faceHeight1);
		gf->gaborResponseImg = (float*)malloc(sizeof(float)* gf->faceWidth1 * gf->faceHeight1);

	}
	else
	{
		gf->gaborCoefs = NULL;
		gf->GaborHist = NULL;
		gf->gaborResponse = NULL;
		gf->gaborResponseReal = NULL;
		gf->gaborResponseImg = NULL;
	}

	if ( gf->bUseIntensity)
	{
		gf->IntensityHist = (UInt*)malloc(sizeof(UInt) * gf->numBinsIntensity);
	}
	else
	{
		gf->IntensityHist = NULL;
	}

	if ( ((gf->bUseReferDist) || (gf->bVerification)) && (gf->bIsTraining))
	{
		int i;
		gf->bufferFeatures = (featStruct*)malloc(sizeof(featStruct) * gf->maxNumImages);
		for ( i = 0; i < gf->maxNumImages; i++)
		{
			initOneFeature(&(gf->bufferFeatures[i]), gf);
		}
	}
	else
	{
		gf->bufferFeatures = NULL;
	}

	gf->referCenters = NULL;

	if ( (gf->bVerification) && (gf->bIsTraining) && (!gf->bUseAllSamples))
	{
		int	i;
		//init svm buffer
		gf->svmTrainFeatures = (float**)malloc(sizeof(float*) * gf->svmNumSamples);
		for ( i = 0; i < gf->svmNumSamples; i++)
		{
			gf->svmTrainFeatures[i] = (float*)malloc(sizeof(float) * gf->featLenTotal);
		}
		gf->svmSampleLabels = (int*)malloc(sizeof(int) * gf->svmNumSamples);
	}
	else
	{
		gf->svmTrainFeatures = NULL;
		gf->svmSampleLabels = NULL;
	}

	if (gf->bVerification)
	{
		gf->svmModel = (float*)malloc(sizeof(int) * gf->featLenTotal);
		gf->svmModelTmp = (float*)malloc(sizeof(int) * gf->featLenTotal);
	}
	else
	{
		gf->svmModel = NULL;
		gf->svmModelTmp = NULL;
	}

	if ( gf->bVerification)
	{
		gf->svmTmpFeature = (float*)malloc(sizeof(int) * gf->featLenTotal);
	}
	else
	{
		gf->svmTmpFeature = NULL;
	}

	gf->whiteList = NULL;




}//end initGlobalStruct

/* release memory */
void freeGlobalStruct(gFaceReco* gf)
{
	int i;

	freeOneFeature(&(gf->features));

	if ( gf->LBPHist != NULL)
	{
		free(gf->LBPHist);
		gf->LBPHist = NULL;
	}

	if ( gf->uniTableLBP != NULL)
	{
		free(gf->uniTableLBP);
		gf->uniTableLBP = NULL;
	}

	if ( gf->weight != NULL)
	{
		free(gf->weight);
		gf->weight = NULL;
	}

	if ( gf->loadedFeatures != NULL)
	{
		for ( i = 0; i < gf->numLoadedFaces; i++)
		{
			freeOneFeature(&(gf->loadedFeatures[i]));
		}
		gf->loadedFeatures = NULL;
	}

	if ( gf->bufferFeatures != NULL)
	{
		for ( i = 0; i < gf->maxNumImages; i++)
		{
			freeOneFeature(&(gf->bufferFeatures[i]));
		}
		gf->bufferFeatures = NULL;
	}

	if ( gf->imageList != NULL)
	{
		free(gf->imageList);
		gf->imageList = NULL;
	}

	if ( gf->gaborCoefs != NULL)
	{
		for ( i = 0; i < 2 * gf->nGabors; i++)
		{
			free(gf->gaborCoefs[i]);
			gf->gaborCoefs[i] = NULL;
		}
		free(gf->gaborCoefs);
		gf->gaborCoefs = NULL;
	}

	if ( gf->GaborHist != NULL)
	{
		free(gf->GaborHist);
		gf->GaborHist = NULL;
	}

	if (gf->gaborResponse != NULL)
	{
		free(gf->gaborResponse);
		gf->gaborResponse = NULL;
	}

	if ( gf->gaborResponseReal != NULL)
	{
		free(gf->gaborResponseReal);
		gf->gaborResponseReal = NULL;
	}

	if ( gf->gaborResponseImg != NULL)
	{
		free(gf->gaborResponseImg);
		gf->gaborResponseImg = NULL;
	}

	if ( gf->IntensityHist != NULL)
	{
		free(gf->IntensityHist);
		gf->IntensityHist = NULL;
	}

	if ( gf->referCenters != NULL)
	{
		for ( i = 0; i < gf->featLenReferDist; i++)
		{
			if ( gf->referCenters[i] != NULL)
			{
				free(gf->referCenters[i]);
				gf->referCenters[i] = NULL;
			}
		}
		gf->referCenters = NULL;
	}

	if ( gf->svmTrainFeatures != NULL)
	{
		for ( i = 0; i < gf->svmNumSamples; i++)
		{
			if ( gf->svmTrainFeatures[i] != NULL)
			{
				free(gf->svmTrainFeatures[i]);
				gf->svmTrainFeatures[i] = NULL;
			}
		}
		free(gf->svmTrainFeatures);
		gf->svmTrainFeatures = NULL;
	}

	if ( gf->svmSampleLabels != NULL)
	{
		free(gf->svmSampleLabels);
		gf->svmSampleLabels = NULL;
	}

	if ( gf->svmTmpFeature != NULL)
	{
		free( gf->svmTmpFeature);
		gf->svmTmpFeature = NULL;
	}

	if ( gf->whiteList != NULL)
	{
		free(gf->whiteList);
		gf->whiteList = NULL;
	}



}//end freeGlobalStruct

/* allocate one feature structure */
void initOneFeature(featStruct* fst, gFaceReco* gf)
{
	if (gf->bUseLBP)
	{
		fst->featLBP = (float*)malloc(sizeof(float) * gf->featLenLBP);
	}
	else
	{
		fst->featLBP = NULL;
	}

	if ( gf->bUseGabor)
	{
		fst->featGabor = (float*)malloc(sizeof(float) * gf->featLenGabor);
	}
	else
	{
		fst->featGabor = NULL;
	}

	if ( gf->bUseIntensity)
	{
		fst->featIntensity = (float*)malloc(sizeof(float) * gf->featLenIntensity);
	}
	else
	{
		fst->featIntensity = NULL;
	}

	if ( gf->bUseReferDist)
	{
		fst->featReferDist = (float*)malloc(sizeof(float) * gf->featLenReferDist);
	}
	else
	{
		fst->featReferDist = NULL;
	}

}//end initOneFeature

/* release one feature structure */
void freeOneFeature(featStruct* fst)
{
	if ( fst != NULL)
	{
		if ( fst->featLBP != NULL)
		{
			free(fst->featLBP);
			fst->featLBP = NULL;
		}
		if ( fst->featGabor != NULL)
		{
			free(fst->featGabor);
			fst->featGabor = NULL;
		}
		if ( fst->featIntensity != NULL)
		{
			free(fst->featIntensity);
			fst->featIntensity = NULL;
		}
		if ( fst->featReferDist != NULL)
		{
			free(fst->featReferDist);
			fst->featReferDist = NULL;
		}
	}
}//end freeOneFeature



/* reset histogram */
inline void resetHist(UInt* hist, int n)
{
	//int		i;
	//for ( i = 0; i < n; i++)
	//{
	//	hist[i] = 0;
	//}
	memset(hist, 0, sizeof(UInt) * n);
}

/* shuffle array to randomly select group members */
void shuffle(int *list, int n) 
{    
    srand((unsigned) time(NULL)); 
	int t, j;


    if (n > 1) 
	{
        int i;
        for (i = n - 1; i > 0; i--) 
		{
            j = i + rand() / (RAND_MAX / (n - i) + 1);
            t = list[j];
            list[j] = list[i];
            list[i] = t;
        }
    }
}

/*  Extract LBP Features  */
void extractLBPFeatures(gFaceReco* gf)
{
	int		i, j, k, gr, gc;
	int		numBins;
	int		ptrFeat;
	UChar	currVal, *ptrImg;
	int		LBPVal;
	int		height, width;
	float*	feature;
	int		stepW, stepH, winW, winH;
	UInt*	hist;
	int*	tab;
	UChar*	face1;
	UChar*  face2;
	int		threshold;

	feature = gf->features.featLBP;
	stepW	= gf->LBPStepW;
	stepH	= gf->LBPStepH;
	winW	= gf->LBPWindowW;
	winH	= gf->LBPWindowH;
	numBins = gf->numBinsLBP;
	hist	= gf->LBPHist;
	tab		= gf->uniTableLBP;
	face1	= gf->face1;
	face2	= gf->face2;
	threshold = gf->LBPNeighBorThreshold;

	//init
	ptrFeat = 0;

	//------------------ extraction at face1(1/2 face0)----------------------//
	height = gf->faceHeight1;
	width  = gf->faceWidth1;

	for(gr=0; gr<=(height-winH); gr+=stepH)
	{
		for(gc=0; gc<=(width-winW); gc+=stepW)
		{
			resetHist(hist, numBins);
			for(i=1; i<(winH-1); i++) 
			{
				for(j=1; j<(winW-1); j++)
				{
					LBPVal = 0;

					ptrImg = face1 + (gr + i) * width + ( gc + j);
					currVal = *ptrImg;


					if(currVal < (*(ptrImg - 1)-threshold)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
					if(currVal < (*(ptrImg + 1)-threshold)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
					if(currVal < (*(ptrImg - width)-threshold)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
					if(currVal < (*(ptrImg + width)-threshold)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

					if(currVal < (*(ptrImg - width + 1)-threshold)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
					if(currVal < (*(ptrImg - width - 1)-threshold)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
					if(currVal < (*(ptrImg + width + 1)-threshold)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
					if(currVal < (*(ptrImg + width - 1)-threshold)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;

					//save into histogram
					if ( gf->bUniformLBP)
					{
						hist[tab[LBPVal]] = hist[tab[LBPVal]] + 1;
					}
					else
					{
						hist[LBPVal] = hist[LBPVal] + 1;
					}
				}
			}

			//save into features
			for ( k = 0; k < numBins; k++)
			{
				feature[ptrFeat] = (float)hist[k];
				ptrFeat++;
			}
		}
	}//end extraction at face1

	//------------------ extraction at face2(1/4 face0)----------------------//

	height = gf->faceHeight2;
	width  = gf->faceWidth2;

	for(gr=0; gr<=(height-winH); gr+=stepH)
	{
		for(gc=0; gc<=(width-winW); gc+=stepW)
		{
			resetHist(hist, numBins);
			for(i=1; i<(winH-1); i++) 
			{
				for(j=1; j<(winW-1); j++)
				{
					LBPVal = 0;

					ptrImg = face2 + (gr + i) * width + ( gc + j);
					currVal = *ptrImg;


					if(currVal < (*(ptrImg - 1)-threshold)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
					if(currVal < (*(ptrImg + 1)-threshold)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
					if(currVal < (*(ptrImg - width)-threshold)) LBPVal = LBPVal + 4;			//LBPFlag[2] = 1;
					if(currVal < (*(ptrImg + width)-threshold)) LBPVal = LBPVal + 8;			//LBPFlag[3] = 1;

					if(currVal < (*(ptrImg - width + 1)-threshold)) LBPVal = LBPVal + 16;		//LBPFlag[4] = 1;
					if(currVal < (*(ptrImg - width - 1)-threshold)) LBPVal = LBPVal + 32;		//LBPFlag[5] = 1;
					if(currVal < (*(ptrImg + width + 1)-threshold)) LBPVal = LBPVal + 64;		//LBPFlag[6] = 1;
					if(currVal < (*(ptrImg + width - 1)-threshold)) LBPVal = LBPVal + 128;		//LBPFlag[7] = 1;

					//save into histogram
					if ( gf->bUniformLBP)
					{
						hist[tab[LBPVal]] = hist[tab[LBPVal]] + 1;
					}
					else
					{
						hist[LBPVal] = hist[LBPVal] + 1;
					}
				}
			}

			//save into features
			for ( k = 0; k < numBins; k++)
			{
				feature[ptrFeat] = hist[k];
				ptrFeat++;
			}
		}
	}//end extraction at face2

	assert(ptrFeat == gf->featLenLBP);



}//end extractLBPFeatures



void dumpFeatures(gFaceReco* gf, FILE* pFaceFeatBin)
{
	
	//write features to binary file
	fwrite(&(gf->features.id), sizeof(int), 1, pFaceFeatBin);	//write id

	
	if ( gf->bUseLBP)
	{
		fwrite(gf->features.featLBP, sizeof(float), gf->featLenLBP, pFaceFeatBin);
	}

	if ( gf->bUseGabor)
	{
		fwrite(gf->features.featGabor, sizeof(float), gf->featLenGabor, pFaceFeatBin);
	}

	if ( gf->bUseIntensity)
	{
		fwrite(gf->features.featIntensity, sizeof(float), gf->featLenIntensity, pFaceFeatBin);
	}
	
	

}//end dumpFeatures



void loadFeatures(gFaceReco* gf)
{
	FILE*		pFaceFeatBin;
	FILE*		pCurrent;
	errno_t		err;
	bool		bUseLBP, bUseGabor, bUseIntensity, bUseReferDist;
	int			oneFeatLen;
	int			binaryLen;
	int			oneFeatLenInByte;
	int			numFaces;
	int			i;
	int			tmpLoadedID;
	int			directFeatLen;

	err = fopen_s(&pFaceFeatBin, gf->faceBinPath, "rb");
	if (err != 0)
	{
		printf("Can't open feature binary file to read!\n");
		system("pause");
		exit(-1);
	}

	//get file size
	fseek( pFaceFeatBin, 0, SEEK_END);
	binaryLen = ftell(pFaceFeatBin);
	rewind(pFaceFeatBin);

	//read switches first
	fread(&(bUseReferDist), sizeof(bool), 1, pFaceFeatBin);
	fread(&(bUseLBP), sizeof(bool), 1, pFaceFeatBin);
	fread(&(bUseGabor), sizeof(bool), 1, pFaceFeatBin);
	fread(&(bUseIntensity), sizeof(bool), 1, pFaceFeatBin);

	//read feat length for each face
	fread(&oneFeatLen, sizeof(int), 1, pFaceFeatBin);

	assert( ( bUseLBP == gf->bUseLBP) && ( bUseGabor == gf->bUseGabor) && (bUseIntensity == gf->bUseIntensity) && (bUseReferDist == gf->bUseReferDist));

	oneFeatLenInByte = oneFeatLen * sizeof(float) + sizeof(int);

	//current ptr position
	pCurrent = pFaceFeatBin;

	//get file size
	numFaces =  binaryLen / oneFeatLenInByte;
	gf->numLoadedFaces = numFaces;

	gf->loadedFeatures = (featStruct*)malloc(sizeof(featStruct) * numFaces);

	//load from binary
	for ( i = 0; i < numFaces; i++)
	{
		initOneFeature(&(gf->loadedFeatures[i]), gf);
		fread(&tmpLoadedID, sizeof(int), 1, pCurrent);
		gf->loadedFeatures[i].id = tmpLoadedID;
		if ( gf->bUseReferDist)
		{
			fread(gf->loadedFeatures[i].featReferDist, sizeof(float), gf->featLenReferDist, pCurrent);
		}
		else
		{
			if ( gf->bUseLBP)
			{
				//gf->loadedFeatures[i].featLBP = (float*)malloc(sizeof(float) * gf->featLenLBP);
				fread(gf->loadedFeatures[i].featLBP, sizeof(float), gf->featLenLBP, pCurrent);
			}
			if ( gf->bUseGabor)
			{
				//gf->loadedFeatures[i].featGabor = (float*)malloc(sizeof(float) * gf->featLenGabor);
				fread(gf->loadedFeatures[i].featGabor, sizeof(float), gf->featLenGabor, pCurrent);
			}
			if ( gf->bUseIntensity)
			{
				//gf->loadedFeatures[i].featIntensity = (float*)malloc(sizeof(float) * gf->featLenIntensity);
				fread(gf->loadedFeatures[i].featIntensity, sizeof(float), gf->featLenIntensity, pCurrent);
			}
		}

	}

	fclose(pFaceFeatBin);
	pCurrent = NULL;

	//load reference centers(if enabled)
	if ( gf->bUseReferDist)
	{
		FILE* pReferCenters;
		err = fopen_s(&pReferCenters, gf->referCentersPath, "rb");
		if (err != 0)
		{
			printf("Can't open referCenters.bin to read!\n");
			system("pause");
			exit(-1);
		}

		directFeatLen = 0;
		if ( gf->bUseLBP)
			directFeatLen += gf->featLenLBP;
		if ( gf->bUseGabor)
			directFeatLen += gf->featLenGabor;
		if ( gf->bUseIntensity)
			directFeatLen += gf->featLenIntensity;

		gf->referCenters = (float**)malloc(sizeof(float*) * gf->featLenReferDist);
		for ( i = 0; i < gf->featLenReferDist; i++)
		{
			gf->referCenters[i] = (float*)malloc(sizeof(float) * directFeatLen);
			fread(gf->referCenters[i], sizeof(float), directFeatLen, pReferCenters);
		}
		fclose(pReferCenters);
	}
		

	//load weight
	if ( gf->bUseWeight)
	{
		FILE* pWeightBin;
		err = fopen_s(&(pWeightBin), gf->weightBinPath, "rb");
		if (err != 0)
		{
			printf("Can't open weight binary file to read!\n");
			system("pause");
			exit(-1);
		}
		fread(gf->weight, sizeof(float), gf->featLenTotal, pWeightBin);

		fclose(pWeightBin);
	}

	//load SVM whitelist model
	if ( (gf->bWhiteList) && (gf->bIsMatching))
	{
		FILE*	pModel;
		char	tmp[1024];
		float	temp;
		int		size;
		sprintf(tmp, "%swhiteList.model", gf->svmModelDir);
		err = fopen_s(&pModel, tmp, "r");
		if (err != 0)
		{
			printf("Can't open white list model to read!\n");
			system("pause");
			exit(-1);
		}

		for ( i = 0; i < 5; i++)	//some header info, throw out
		{
			fgets(tmp, 100, pModel);
		}

		for ( i = 0; i < gf->svmFeatureLen; i++)
		{
			fscanf(pModel, "%lf ", &gf->svmModel[i]);
		}

		assert(fscanf(pModel, "%lf", &temp) == EOF);

		fclose(pModel);

		for ( i = gf->svmFeatureLen; i < gf->featLenTotal; i++)
			gf->svmModel[i] = 0;

		//open white list record
		sprintf(tmp, "%swhiteList.txt", gf->svmModelDir);
		err = fopen_s(&pModel, tmp, "r");
		if (err != 0)
		{
			printf("Can't open white list model to read!\n");
			system("pause");
			exit(-1);
		}
		fscanf(pModel, "%d\n", &size);
		if ( size > 0)
		{
			gf->sizeList = size;
			gf->whiteList = (int*)malloc(sizeof(int) * size);
			for ( i = 0; i < size; i++)
			{
				fscanf(pModel, "%d\n", &gf->whiteList[i]);
			}
		}
		fclose(pModel);

	}


	

	printf("Features loaded!\n");




}//end loadFeatures


/* 2D convolution for gray 8-bit image*/
// src : input image
// dst : output
// kernel: filter kernel
// size : kernel size n(should be an odd number)
// height: image height
// width: image width
void convolution2D(UChar *src, float *dst, double *kernel, int size, int height, int width)
{
	assert((size % 2 == 1) && (size >= 3)); // kernel size should be an odd number
	int cRow, cCol; //center row and column
	int kRow, kCol; //kernel row and column
	int posRow, posCol; //current accessing position
	int pad = (size - 1)/2;
	int doubleHeight = 2 * (height-1);
	int doubleWidth = 2 * (width-1);
	float sum;

	
	
	//scan image
	for (cRow = 0; cRow < height; cRow++ )
	{
		for (cCol = 0; cCol < width; cCol++)
		{
			sum = 0;
			//scan kernel
			for ( kRow = -pad; kRow < pad; kRow++ )
			{
				for ( kCol = -pad; kCol < pad; kCol++)
				{
					posRow = cRow + kRow;
					posCol = cCol + kCol;
					//out of border pixels
					if (posRow < 0) 
					{
						posRow = 0 - posRow;
					}
					else if (posRow >= height) 
					{
						posRow = doubleHeight - posRow;
					}
					if (posCol < 0) 
					{
						posCol = 0 - posCol;
					}
					else if (posCol >= width) 
					{
						posCol = doubleWidth - posCol;
					}

					sum += (float)src[width * posRow + posCol] * (float)kernel[ size * (kRow + pad) + kCol + pad];
				}
			}
			dst[width * cRow + cCol] = sum;
		}
	}


}//end function convulution2D



/* Extract Gabor Features */
void extractGaborFeatures(gFaceReco* gf)
{
	UChar*		face1;
	double**	gaborCoefs;
	double*		kernel;
	int			kernelSize;
	float*		gaborResponse, *gaborResponseReal, *gaborResponseImg;
	float		real, img;
	int			height, width;
	int			i, j, k, m, n, gr, gc;
	int			nGabors;
	float*		feature;
	int			stepW, stepH, winW, winH;
	int			numBins;
	UInt*		hist;
	float*		ptr;
	int			ptrFeat;
	float		currVal;
	int			histVal;
	int			threshold;
	

	hist		= gf->GaborHist;
	stepW		= gf->gaborStepW;
	stepH		= gf->gaborStepH;
	winW		= gf->gaborWindowW;
	winH		= gf->gaborWindowH;
	numBins		= gf->numBinsGabor;
	threshold	= gf->gaborNeighBorThreshold;
	face1		= gf->face1;
	height		= gf->faceHeight1;
	width		= gf->faceWidth1;
	feature		= gf->features.featGabor;
	gaborCoefs	= gf->gaborCoefs;
	nGabors		= gf->nGabors;
	kernelSize	= gf->gaborSize;
	gaborResponse = gf->gaborResponse;
	gaborResponseReal = gf->gaborResponseReal;
	gaborResponseImg = gf->gaborResponseImg;

	//init
	ptrFeat = 0;

	for ( i = 0; i < 2 * nGabors; i += 2)
	{
		kernel = gaborCoefs[i];
		convolution2D(face1, gaborResponseReal, kernel, kernelSize, height, width);
		kernel = gaborCoefs[i+1];
		convolution2D(face1, gaborResponseImg, kernel, kernelSize, height, width);
		//get magnitude
		for ( j = 0; j < height * width; j++)
		{
			real = gaborResponseReal[j];
			img  = gaborResponseImg[j];
			gaborResponse[j] = real*real + img*img;
		}

		//extract HSLGBP
		

		for(gr=0; gr<=(height-winH); gr+=stepH)
		{
			for(gc=0; gc<=(width-winW); gc+=stepW)
			{
				resetHist(hist, numBins);
				for(m=1; m<(winH-1); m++) 
				{
					for(n=1; n<(winW-1); n++)
					{
						histVal = 0;

						ptr = gaborResponse + (gr + m) * width + ( gc + n);
						currVal = *ptr;


						if(currVal < (*(ptr - 1)-threshold)) histVal = histVal + 1;				//LBPFlag[0] = 1;
						if(currVal < (*(ptr + 1)-threshold)) histVal = histVal + 2;				//LBPFlag[1] = 1;
						if(currVal < (*(ptr - width)-threshold)) histVal = histVal + 4;		//LBPFlag[2] = 1;
						if(currVal < (*(ptr + width)-threshold)) histVal = histVal + 8;		//LBPFlag[3] = 1;

						if(currVal < (*(ptr - width + 1)-threshold)) histVal = histVal + 16;	//LBPFlag[4] = 1;
						if(currVal < (*(ptr - width - 1)-threshold)) histVal = histVal + 32;	//LBPFlag[5] = 1;
						if(currVal < (*(ptr + width + 1)-threshold)) histVal = histVal + 64;	//LBPFlag[6] = 1;
						if(currVal < (*(ptr + width - 1)-threshold)) histVal = histVal + 128;	//LBPFlag[7] = 1;

						//save into histogram
						histVal = histVal * numBins / 256;
						hist[histVal] = hist[histVal] + 1;
					}
				}

				//save into features
				for ( k = 0; k < numBins; k++)
				{
					feature[ptrFeat] = (float)hist[k];
					ptrFeat++;
				}
			}
		}//end extraction at face1
		


	}//end gabor kernels loop


	assert(ptrFeat == gf->featLenGabor);




	//clean-ups
	
}//end extractGaborFeatures


/* extract Intensity features as window */
void extractIntensityFeatures(gFaceReco* gf)
{
	int			i, j, gr, gc, k;
	UInt*		hist;
	int			numBins;
	float*		feature;
	UChar		*face0, *face1;
	int			stepW, stepH, winW, winH;
	int			height, width;
	UChar*		ptr;
	UChar		currVal;
	int			featPtr;


	hist		= gf->IntensityHist;
	numBins		= gf->numBinsIntensity;
	feature		= gf->features.featIntensity;
	face0		= gf->face;
	face1		= gf->face1;
	stepW		= gf->IntensityStepW;
	stepH		= gf->IntensityStepH;
	winW		= gf->IntensityWindowW;
	winH		= gf->IntensityWindowH;
	featPtr		= 0;

	//------------------------extract features at original face----------------------------------//
	height	= gf->faceHeight;
	width	= gf->faceWidth;

	for ( gr = 0; gr <= (height - winH); gr += stepH)
	{
		for ( gc = 0; gc <= (width - winW); gc += stepW)
		{
			//reset hist
			resetHist(hist, numBins);
			for ( i = 0; i < winH; i++)
			{
				for ( j = 0; j < winW; j++)
				{
					ptr = face0 + (gr + i) * width + ( gc + j);
					currVal = (int)((*ptr) * numBins / 256);

					hist[currVal] += 1;
				}
			}

			//save to feature vector
			for ( k = 0; k < numBins; k++)
			{
				feature[featPtr] = (float)hist[k];
				featPtr++;
			}
		}
	}
	

	//------------------------extract features at face 1----------------------------------//
	height	= gf->faceHeight1;
	width	= gf->faceWidth1;

	for ( gr = 0; gr <= (height - winH); gr += stepH)
	{
		for ( gc = 0; gc <= (width - winW); gc += stepW)
		{
			//reset hist
			resetHist(hist, numBins);
			for ( i = 0; i < winH; i++)
			{
				for ( j = 0; j < winW; j++)
				{
					ptr = face1 + (gr + i) * width + ( gc + j);
					currVal = (int)((*ptr) * numBins / 256);

					hist[currVal] += 1;
				}
			}

			//save to feature vector
			for ( k = 0; k < numBins; k++)
			{
				feature[featPtr] = (float)hist[k];
				featPtr++;
			}
		}
	}

	//
	assert(featPtr == gf->featLenIntensity);



}//end extractIntensityFeatures



/* match features */
int matchFaceID(gFaceReco* gf)
{
	int				i;
	int				matchedID;
	int				numLoadedFaces;
	float			sumDist;
	float			minDist;
	featStruct*		loadedFeatures;
	featStruct*		currFeatures;


	loadedFeatures = gf->loadedFeatures;
	currFeatures   = &gf->features;
	numLoadedFaces = gf->numLoadedFaces;
	minDist		   = MAX_FLOAT_NUMBER;
	matchedID	   = 0;

	for ( i = 0; i < numLoadedFaces; i++)
	{
		sumDist = 0;
		if ( gf->bUseReferDist)
		{
			sumDist += matchFeatureDist( currFeatures->featReferDist, loadedFeatures[i].featReferDist, gf->featLenReferDist);
		}
		else
		{
			if ( gf->bUseLBP)
			{
				sumDist += matchFeatureHistDist( currFeatures->featLBP, loadedFeatures[i].featLBP, gf->featLenLBP);
			}
			if ( gf->bUseGabor)
			{
				sumDist += matchFeatureHistDist( currFeatures->featGabor, loadedFeatures[i].featGabor, gf->featLenGabor);
			}
			if ( gf->bUseIntensity)
			{
				sumDist += matchFeatureHistDist( currFeatures->featIntensity, loadedFeatures[i].featIntensity, gf->featLenIntensity);
			}
		}

		if ( sumDist < minDist)
		{
			matchedID = loadedFeatures[i].id;
			minDist = sumDist;
		}
		
	}

	//return 
	return matchedID;




}//end matchFaceID

float matchFeatureDist(float* feature1, float* feature2, int length)
{
	float	h1, h2;
	int		i;
	float	dist, normalizedDist, tmp;


	dist = 0;
	for ( i = 0; i < length; i++)
	{
		h1 = feature1[i];
		h2 = feature2[i];
		tmp = h1 - h2;
		dist += tmp * tmp;
	}

	normalizedDist = dist / length;

	return normalizedDist;


}//end matchFeatureDist


/* match histogram distance sumation */
float matchFeatureHistDist(float* feature1, float* feature2, int length)
{
	float	h1, h2;
	int		i;
	float	dist, normalizedDist;
	float	tmp;

	dist = 0;
	for ( i = 0; i < length; i++)
	{
		h1 = feature1[i];
		h2 = feature2[i];
		if ( (h1 > 0) || (h2 > 0))
		{
			tmp = (h1 > h2)? (h1 - h2): (h2 - h1);
			dist += tmp / (h1 + h2);	//chi-square distance
		}
	}

	//normalize
	normalizedDist = dist / length;
	return normalizedDist;

}//end matchFeatureDist


/* match face in white list */
int	matchFaceWhiteList(gFaceReco* gf)
{
	int				i, j;
	int				matchedID;
	int				numLoadedFaces;
	featStruct*		loadedFeatures;
	featStruct*		currFeatures;
	float*			svmTmpFeatures;
	float			score;
	int				voteBin, capBin;

	loadedFeatures	= gf->loadedFeatures;
	currFeatures	= &gf->features;
	numLoadedFaces	= gf->numLoadedFaces;
	svmTmpFeatures	= gf->svmTmpFeature;
	matchedID		= 0;

	capBin = 0;
	voteBin = 0;
	for ( i = 0; i < numLoadedFaces; i++)
	{
		if ( isInList(gf->whiteList, gf->sizeList, loadedFeatures[i].id))
		{
			score = 0;
			extractAbsDist(gf, &loadedFeatures[i], currFeatures, svmTmpFeatures);
			for ( j = 0; j < gf->featLenTotal; j++)
			{
				score += svmTmpFeatures[j] * gf->svmModel[j];
			}
			if ( score > gf->bias)
			{
				voteBin++;
			}
			capBin++;
		}
	}

	if ( (voteBin / capBin * 1.0) > 0.8)
		return 1;
	else
		return 0;


}//end matchFaceWhiteList


/* match face ID using verification method */
int matchFaceIDVerification(gFaceReco* gf, FILE* pDebug)
{
	int				i;
	int				matchedID;
	int				numLoadedFaces;
	int				ptr;
	featStruct*		loadedFeatures;
	featStruct*		currFeatures;
	float*			svmTmpFeatures;
	int				fType = 4;
	float			score;
	int*			voteBin, *capBin;
	float			maxVote, tmpVote;
	//debug only
	int				incorrectMatch = 0;
	int				totalMatch = 0;


	loadedFeatures	= gf->loadedFeatures;
	currFeatures	= &gf->features;
	numLoadedFaces	= gf->numLoadedFaces;
	svmTmpFeatures	= gf->svmTmpFeature;
	matchedID		= 0;
	maxVote			= -1;
	tmpVote			= 0;
	voteBin			= (int*)malloc(sizeof(int) * gf->maxFaceTags);
	capBin			= (int*)malloc(sizeof(int) * gf->maxFaceTags);
	//svm.svm_init_clean(gf->svmModelPath);		//load svm model

	for ( i = 0; i < gf->maxFaceTags; i++)
	{
		voteBin[i] = 0;
		capBin[i] = 0;
	}

	for ( i = 0; i < numLoadedFaces; i++)
	{
		extractAbsDist(gf, currFeatures, &(loadedFeatures[i]), svmTmpFeatures);
		//svm.svm_classifier_clean(&fType, svmTmpFeatures, &score, gf->featLenTotal, 1);
		capBin[loadedFeatures[i].id - 1] += 1;
		if ( score > 0)
		{
			voteBin[loadedFeatures[i].id - 1] += 1;
		}
	}

	//debug
	fprintf(pDebug, "%d	", currFeatures->id);
	for ( i = 0; i < gf->maxFaceTags; i++)
	{
		if ( capBin[i] > 0)
		{
			tmpVote = ((float)voteBin[i]) / capBin[i];
			//debug only
			if ( i == currFeatures->id - 1)
			{
				incorrectMatch += capBin[i] - voteBin[i];
				fprintf(pDebug, "%.2f	", tmpVote * 100);
			}
			else
			{
				incorrectMatch += voteBin[i];
				fprintf(pDebug, "%.2f	", tmpVote*100);
			}
			totalMatch += capBin[i];
			//end debug
			if ( tmpVote > maxVote)
			{
				maxVote = tmpVote;
				matchedID = i + 1;
			}
		}
	}
	
	//debug only
	fprintf(pDebug, "%.2f\n", (1.0 - (float)incorrectMatch/totalMatch)*100);
	
	//
	//clean-up
	free(voteBin);
	voteBin = NULL;
	free(capBin);
	capBin = NULL;
	

	return matchedID;



}//end matchFaceIDVerification


/* copy one feature combination to buffer */
void copyOneFeatureToBuffer(gFaceReco* gf, int idx)
{
	featStruct* currFeat, *bufferFeat;

	currFeat = &gf->features;
	bufferFeat = &gf->bufferFeatures[idx];

	bufferFeat->id = currFeat->id;

	if ( gf->bUseLBP)
	{
		memcpy(bufferFeat->featLBP, currFeat->featLBP, sizeof(float) * gf->featLenLBP);
	}
	if ( gf->bUseGabor)
	{
		memcpy(bufferFeat->featGabor, currFeat->featGabor, sizeof(float) * gf->featLenGabor);
	}
	if ( gf->bUseIntensity)
	{
		memcpy(bufferFeat->featIntensity, currFeat->featIntensity, sizeof(float) * gf->featLenIntensity);
	}

}//end copyOneFeatureToBuffer

/**
* extract Reference Distance features
* Apply kmeans to get reference centers
* Then get the distance features as a new feature
**/
void extractReferDistFeatures(gFaceReco* gf, FILE* pFaceFeatBin)
{
	KMeanType		KM;
	int				vectorSize;
	int				dataSize;
	int				numClusters;
	int				i, j, k, ptr;
	float**			kmInput;
	int*			list;
	featStruct*		bufferFeat;
	float			dist, tmpDist;
	FILE*			pReferCenters;
	errno_t			err;


	//open reference centers binary
	err = fopen_s(&pReferCenters, gf->referCentersPath, "wb");
	if ( err != 0)
	{
		printf("Error opening referCenters.bin to write!\n");
		system("pause");
		exit(-1);
	}


	bufferFeat = gf->bufferFeatures;

	//init k-means
	vectorSize = 0;
	if ( gf->bUseLBP)
		vectorSize += gf->featLenLBP;
	if ( gf->bUseGabor)
		vectorSize += gf->featLenGabor;
	if ( gf->bUseIntensity)
		vectorSize += gf->featLenIntensity;

	dataSize = gf->numValidFaces;
	numClusters = gf->featLenReferDist;
	initKMeanWithParameters(&KM, dataSize, vectorSize, numClusters);
	kmInput = (float**)malloc(sizeof(float*) * dataSize);
	//concatenating features
	for ( i = 0; i < dataSize; i++)
	{
		kmInput[i] = (float*)malloc(sizeof(float) * vectorSize);
		ptr = 0;
		if ( gf->bUseLBP)
		{
			memcpy(&(kmInput[i][ptr]), bufferFeat[i].featLBP, sizeof(float) * gf->featLenLBP);
			ptr += gf->featLenLBP;
		}
		if ( gf->bUseGabor)
		{
			memcpy(&(kmInput[i][ptr]), bufferFeat[i].featGabor, sizeof(float) * gf->featLenGabor);
			ptr += gf->featLenGabor;
		}
		if ( gf->bUseIntensity)
		{
			memcpy(&(kmInput[i][ptr]), bufferFeat[i].featIntensity, sizeof(float) * gf->featLenIntensity);
			ptr += gf->featLenIntensity;
		}
	}//end prepare for k-mean data

	//randomly assign centers
	list = (int*)malloc(sizeof(int) * dataSize);
	for ( i = 0; i < dataSize; i++)
	{
		list[i] = i;
	}
	shuffle(list, dataSize);
	for ( i = 0; i < numClusters; i++)
	{
		memcpy(&KM.kMeanClusterCenters[i][0], &kmInput[list[i]][0], sizeof(float) * vectorSize);
	}

	//apply k-means
	kMeanClustering(kmInput, dataSize, vectorSize, numClusters, &KM, TRUE);


	
	//extract new feature with cluster centers
	for ( i = 0; i < dataSize; i++)
	{
		for ( j = 0; j < numClusters; j++)
		{
			dist = 0;
			for ( k = 0; k < vectorSize; k++)
			{
				tmpDist = KM.kMeanClusterCenters[j][k] - kmInput[i][k];
				dist += tmpDist * tmpDist;
			}
			gf->features.featReferDist[j] = dist;
		}

		memcpy(gf->bufferFeatures[i].featReferDist, gf->features.featReferDist, sizeof(float)*gf->featLenReferDist);
		//write features to binary file
		fwrite(&(bufferFeat[i].id), sizeof(int), 1, pFaceFeatBin);

		fwrite(gf->features.featReferDist, sizeof(float), numClusters, pFaceFeatBin);
	}
	
	//write centers to binary file
	for ( j = 0; j < numClusters; j++)
	{
		fwrite(KM.kMeanClusterCenters[j], sizeof(float), vectorSize, pReferCenters);
	}


	

	//clean-ups
	fclose(pReferCenters);
	for ( i = 0; i < dataSize; i++)
	{
		free(kmInput[i]);
		kmInput[i] = NULL;
	}
	free(kmInput);
	kmInput = NULL;
	free(list);
	list = NULL;

	releaseKMean(&KM, dataSize, vectorSize, numClusters);


}//end extractReferDistFeatures


/* Extract Reference distances in match */
void extractReferDistFeaturesInMatch(gFaceReco* gf)
{
	int				i, j;
	int				ptr;
	float			dist, tmpDist;
	featStruct		features;
	

	features	= gf->features;

	for ( j = 0; j < gf->featLenReferDist; j++)
	{
		ptr = 0;
		dist = 0;
		if ( gf->bUseLBP)
		{
			for ( i = 0; i < gf->featLenLBP; i++)
			{
				tmpDist = features.featLBP[i] - gf->referCenters[j][ptr];
				dist += tmpDist * tmpDist;
				ptr++;
			}
		}
		if ( gf->bUseGabor)
		{
			for ( i = 0; i< gf->featLenGabor; i++)
			{
				tmpDist = features.featGabor[i] - gf->referCenters[j][ptr];
				dist += tmpDist * tmpDist;
				ptr++;
			}
		}

		if ( gf->bUseIntensity)
		{
			for ( i = 0; i < gf->featLenIntensity; i++)
			{
				tmpDist = features.featIntensity[i] - gf->referCenters[j][ptr];
				dist += tmpDist * tmpDist;
				ptr++;
			}
		}

		features.featReferDist[j] = dist;
	}//end one reference distance feature





}//end extractReferDistFeaturesInMatch

/** svm training
* write features into binary file, then call trainModel and test functions
**/
void svmTraining(float ** features, int nSample, int featureSize, int * sampleLabel, char * modelFileName, float c)
{
	int i, j;
	errno_t		err;
	FILE* fp;
	char parameters[500];
	char path[260];

	char *svmTrainFile = "../../image/trainFile.dat";

	err = fopen_s(&fp, svmTrainFile, "w");
	if ( err != 0)
	{
		printf("Error opening svm training file to write!\n");
		system("pause");
		exit(-1);
	}

	
	//write to file
	printf("Prepare to write features, it may take a while depending on disk IO performance...\n");
	printf("# Features: %d, Feature Size: %d\n", nSample, featureSize);
	for (i=0;i<nSample;i++)
	{
		fprintf(fp, "%d qid:1 ", sampleLabel[i]);
		for ( j = 0; j < featureSize; j++)
		{
			if ( features[i][j] != 0)
				fprintf(fp, "%d:%f ", j+1, features[i][j]);
		}
		fprintf(fp,"\n");
	}

	fclose(fp);
	printf("Finished writing features, will start SVM training...\n");

	

	if ( nSample * featureSize < 8000 * 6000 )
	{
		svmMain(c, svmTrainFile, modelFileName);
	}
	else
	{
		sprintf(parameters, "..\\..\\ranksvm64.exe -c %f %s %s", c, svmTrainFile, modelFileName);
		//memory limitation, call outside executives
		system(parameters);
	}

	//test
	sprintf(path, "../../image/predict.txt");
	sprintf(parameters, "..\\..\\predict.exe %s %s %s",svmTrainFile, modelFileName, path);
	system(parameters);

	//unlink(svmTrainFile);
	//unlink(path);

}//end svmTraining

/* computer absolute distance between two features */
void extractAbsDist(gFaceReco* gf, featStruct* feature1, featStruct* feature2, float* dist)
{
	int		i, j, ptr, ptrFeat;
	float	h1, h2, tmp, sum, d1, d2;

	//reset
	memset(dist, 0, sizeof(float) * gf->featLenTotal);

	if ( gf->bUseReferDist)
	{
		for ( i = 0; i < gf->featLenReferDist; i++)
		{
			h1 = feature1->featReferDist[i];
			h2 = feature2->featReferDist[i];
			tmp = h1 - h2;
			if ( tmp < 0)
				tmp = 0 - tmp;
			dist[i] = tmp;
		}
	}
	else if (gf->bUseBlockCS)		//block cosine similarity
	{
		ptr = 0;
		if ( gf->bUseLBP)
		{
			for ( i = 0; i < gf->numHistsLBP; i++)
			{
				for ( j = 0; j < gf->numBinsLBP; j++)
				{
					sum = 0;
					d1 = 0;
					d2 = 0;
					ptrFeat = i * gf->numBinsLBP + j;
					h1 = feature1->featLBP[ptrFeat];
					h2 = feature2->featLBP[ptrFeat];
					sum += h1 * h2;
					d1 += h1 * h1;
					d2 += h2 * h2;
				}
				dist[ptr] = sum/(d1 * d2);
				ptr++;
			}
		}
		if ( gf->bUseGabor)
		{
			for ( i = 0; i < gf->numHistsGabor; i++)
			{
				for ( j = 0; j < gf->numBinsGabor; j++)
				{
					sum = 0;
					d1 = 0;
					d2 = 0;
					ptrFeat = i * gf->numBinsGabor + j;
					h1 = feature1->featGabor[ptrFeat];
					h2 = feature2->featGabor[ptrFeat];
					sum += h1 * h2;
					d1 += h1 * h1;
					d2 += h2 * h2;
				}
				dist[ptr] = sum/(d1 * d2);
				ptr++;
			}
		}
		if ( gf->bUseIntensity)
		{
			for ( i = 0; i < gf->numHistsIntensity; i++)
			{
				for ( j = 0; j < gf->numBinsIntensity; j++)
				{
					sum = 0;
					ptrFeat = i * gf->numBinsIntensity + j;
					h1 = feature1->featIntensity[ptrFeat];
					h2 = feature2->featIntensity[ptrFeat];
					sum += h1 * h2;
					d1 += h1 * h1;
					d2 += h2 * h2;
				}
				dist[ptr] = sum/(d1 * d2);
				ptr++;
			}
		}

	}
	else if (gf->bUseBlockDist)		//block distance
	{
		ptr = 0;
		if ( gf->bUseLBP)
		{
			for ( i = 0; i < gf->numHistsLBP; i++)
			{
				for ( j = 0; j < gf->numBinsLBP; j++)
				{
					sum = 0;
					ptrFeat = i * gf->numBinsLBP + j;
					h1 = feature1->featLBP[ptrFeat];
					h2 = feature2->featLBP[ptrFeat];
					tmp = h1 - h2;
					if ( tmp < 0)
						tmp = 0 - tmp;
					sum += tmp;
				}
				dist[ptr] = sum;
				ptr++;
			}
		}
		if ( gf->bUseGabor)
		{
			for ( i = 0; i < gf->numHistsGabor; i++)
			{
				for ( j = 0; j < gf->numBinsGabor; j++)
				{
					sum = 0;
					ptrFeat = i * gf->numBinsGabor + j;
					h1 = feature1->featGabor[ptrFeat];
					h2 = feature2->featGabor[ptrFeat];
					tmp = h1 - h2;
					if ( tmp < 0)
						tmp = 0 - tmp;
					sum += tmp;
				}
				dist[ptr] = sum;
				ptr++;
			}
		}
		if ( gf->bUseIntensity)
		{
			for ( i = 0; i < gf->numHistsIntensity; i++)
			{
				for ( j = 0; j < gf->numBinsIntensity; j++)
				{
					sum = 0;
					ptrFeat = i * gf->numBinsIntensity + j;
					h1 = feature1->featIntensity[ptrFeat];
					h2 = feature2->featIntensity[ptrFeat];
					tmp = h1 - h2;
					if ( tmp < 0)
						tmp = 0 - tmp;
					sum += tmp;
				}
				dist[ptr] = sum;
				ptr++;
			}
		}
	}
	else	//single distance
	{
		ptr = 0;
		if ( gf->bUseLBP)
		{
			for ( i = 0; i < gf->featLenLBP; i++)
			{
				h1 = feature1->featLBP[i];
				h2 = feature2->featLBP[i];
				tmp = h1 - h2;
				if ( tmp < 0)
					tmp = 0 - tmp;
				dist[ptr] = tmp / ( h1 + h2 + 1);
				ptr++;
			}
		}
		if ( gf->bUseGabor)
		{
			for ( i = 0; i < gf->featLenGabor; i++)
			{
				h1 = feature1->featGabor[i];
				h2 = feature2->featGabor[i];
				tmp = h1 - h2;
				if ( tmp < 0)
					tmp = 0 - tmp;
				dist[ptr] = tmp / ( h1 + h2 + 1);
				ptr++;
			}
		}
		if ( gf->bUseIntensity)
		{
			for ( i = 0; i < gf->featLenIntensity; i++)
			{
				h1 = feature1->featIntensity[i];
				h2 = feature2->featIntensity[i];
				tmp = h1 - h2;
				if ( tmp < 0)
					tmp = 0 - tmp;
				dist[ptr] = tmp / ( h1 + h2 + 1);
				ptr++;
			}
		}
	}


}//end extractAbsDist


/* return if the query ID is in the list */
bool isInList(int* list, int listLength, int queryID)
{
	int		i;
	bool	found;

	found = FALSE;
	for ( i = 0; i < listLength; i++)
	{
		if ( list[i] == queryID)
		{
			found = TRUE;
			break;
		}
	}

	return found;


}//end isInList


/* Train one to rest model in white list */
void trainOneToRestModels(gFaceReco* gf, int id, int* whiteList, int sizeList)
{
	int			i, j, k;
	int			numPairs;
	int			ptr;
	char		path[260];
	bool		b1, b2;

	//calculate # pairs
	numPairs = 0;
	for ( i = 0; i < gf->numLoadedFaces; i++)
	{
		for ( j = i + 1; j < gf->numLoadedFaces; j++)
		{
			b1 = isInList(whiteList, sizeList, gf->loadedFeatures[i].id);
			b2 = isInList(whiteList, sizeList, gf->loadedFeatures[j].id);
			if ( b1 && b2 && ( (gf->loadedFeatures[i].id == id) || (gf->loadedFeatures[j].id == id)))
				numPairs++;
		}
	}

	//if # pairs larger than current one, reallocate the memory
	if (numPairs > gf->svmNumSamples)
	{
		gf->svmTrainFeatures = (float**)realloc(gf->svmTrainFeatures, numPairs);
		for ( k = gf->svmNumSamples; k < numPairs; k++)
		{
			gf->svmTrainFeatures[k] = (float*)malloc(sizeof(float) * gf->featLenTotal);
		}
		gf->svmSampleLabels = (int*)realloc(gf->svmSampleLabels, numPairs);
		gf->svmNumSamples = numPairs;
	}

	//prepare SVM training samples
	ptr = 0;
	for ( i = 0; i < gf->numLoadedFaces; i++)
	{
		for ( j = i+1 ; j < gf->numLoadedFaces; j++)
		{
			b1 = isInList(whiteList, sizeList, gf->loadedFeatures[i].id);
			b2 = isInList(whiteList, sizeList, gf->loadedFeatures[j].id);
			if ( b1 && b2)
			{
				b1 = ( gf->loadedFeatures[i].id == id);
				b2 = ( gf->loadedFeatures[j].id == id);
				if ( b1 && b2)
				{
					//both intra
					extractAbsDist(gf, &gf->loadedFeatures[i], &gf->loadedFeatures[j], gf->svmTmpFeature);
					memcpy(gf->svmTrainFeatures[ptr], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
					gf->svmSampleLabels[ptr] = 2;
					ptr++;
				}
				else if ( b1 ^ b2)
				{
					//one intra one inter
					extractAbsDist(gf, &gf->loadedFeatures[i], &gf->loadedFeatures[j], gf->svmTmpFeature);
					memcpy(gf->svmTrainFeatures[ptr], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
					gf->svmSampleLabels[ptr] = 1;
					ptr++;
				}
			}//end both in list
		}
	}
	assert(ptr == numPairs);
	sprintf(path, "%s%dtoRest.model", gf->svmModelDir, id);
	svmTraining(gf->svmTrainFeatures, numPairs, gf->svmFeatureLen, gf->svmSampleLabels, path, gf->magicNumber);

}//end trainOneToRestModels


/* match one specific model */
float matchOneInList(gFaceReco* gf, int id)
{
	//load model
	FILE*			pModel;
	errno_t			err;
	char			path[260];
	char			tmp[1024];
	float			temp;
	int				i, j, k;
	float			score;
	float			prob;
	featStruct*		loadedFeatures;
	featStruct*		currFeatures;
	float*			svmTmpFeatures;
	int				voteBin, capBin;
	int				numLoadedFaces;

	loadedFeatures	= gf->loadedFeatures;
	currFeatures	= &gf->features;
	numLoadedFaces	= gf->numLoadedFaces;
	svmTmpFeatures	= gf->svmTmpFeature;


	sprintf(path, "%s%dtoRest.model", gf->svmModelDir, id);
	err = fopen_s(&pModel, path, "r");
	if (err != 0)
	{
		printf("Can't open specific model to read!\n");
		system("pause");
		exit(-1);
	}

	for ( k = 0; k < 5; k++)	//some header info, throw out
	{
		fgets(tmp, 100, pModel);
	}

	for ( i = 0; i < gf->featLenTotal; i++)
	{
		fscanf(pModel, "%f \n", &gf->svmModelTmp[i]);
	}

	assert(fscanf(pModel, "%f", &temp) == EOF);

	fclose(pModel);


	capBin = 0;
	voteBin = 0;
	for ( i = 0; i < numLoadedFaces; i++)
	{
		if ( loadedFeatures[i].id == id)
		{
			score = 0;
			extractAbsDist(gf, &loadedFeatures[i], currFeatures, svmTmpFeatures);
			for ( j = 0; j < gf->featLenTotal; j++)
			{
				score += svmTmpFeatures[j] * gf->svmModelTmp[j];
			}
			if ( score > gf->bias)
			{
				voteBin++;
			}
			capBin++;
		}
	}

	prob = 1.0 * voteBin / capBin;

	return prob;



}//end matchOneInList


/* get the vote decision of a pool */
int	getVotePool(int* pool, int min, int max, int size)
{
	int		i;
	int*	vote;
	int		voteRange;
	int		maxVote = -1;
	int		val = 0;

	voteRange = max - min + 1;
	vote = (int*)malloc(sizeof(int) * voteRange);
	memset(vote, 0, sizeof(int) * voteRange);

	for ( i = 0; i < size; i++)
	{
		vote[pool[i]]++;
	}

	for ( i = 0; i < voteRange; i++)
	{
		if ( vote[i] > maxVote)
		{
			maxVote = vote[i];
			val = min + i;
		}
	}

	return val;


}//end getVotePool


/* svm test */
void svmTest(float ** features, int nSample, int featureSize, int * sampleLabel, char * modelFileName)
{
	int i, j;
	errno_t		err;
	FILE* fp;
	char parameters[500];
	char path[260];

	char *svmTestFile = "../../image/testFile.dat";

	err = fopen_s(&fp, svmTestFile, "w");
	if ( err != 0)
	{
		printf("Error opening svm testing file to write!\n");
		system("pause");
		exit(-1);
	}

	
	//write to file
	printf("Prepare to write features, it may take a while depending on disk IO performance...\n");
	printf("# Features: %d, Feature Size: %d\n", nSample, featureSize);
	for (i=0;i<nSample;i++)
	{
		fprintf(fp, "%d qid:1 ", sampleLabel[i]);
		for ( j = 0; j < featureSize; j++)
		{
			if ( features[i][j] != 0)
				fprintf(fp, "%d:%f ", j+1, features[i][j]);
		}
		fprintf(fp,"\n");
	}

	fclose(fp);
	printf("Finished writing features, will start SVM testing...\n");

	
	//test
	sprintf(path, "../../image/predict.txt");
	sprintf(parameters, "..\\..\\predict.exe %s %s %s",svmTestFile, modelFileName, path);
	system(parameters);

	//unlink(svmTestFile);




}//end svmTest