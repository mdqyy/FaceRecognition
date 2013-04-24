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
#include <io.h>
#include <stdio.h>
#include <stdlib.h>


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





	




}


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
	




}

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



}

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

}

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
	}
}




inline void resetHist(UInt* hist, int n)
{
	int		i;
	for ( i = 0; i < n; i++)
	{
		hist[i] = 0;
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
				feature[ptrFeat] = hist[k];
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
	fwrite(&(gf->features.id), sizeof(int), 1, pFaceFeatBin);
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
	bool		bUseLBP, bUseGabor, bUseIntensity;
	int			oneFeatLen;
	int			binaryLen;
	int			oneFeatLenInByte;
	int			numFaces;
	int			i;
	int			tmpLoadedID;

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
	fread(&(bUseLBP), sizeof(bool), 1, pFaceFeatBin);
	fread(&(bUseGabor), sizeof(bool), 1, pFaceFeatBin);
	fread(&(bUseIntensity), sizeof(bool), 1, pFaceFeatBin);

	//read feat length for each face
	fread(&oneFeatLen, sizeof(int), 1, pFaceFeatBin);

	assert( ( bUseLBP == gf->bUseLBP) && ( bUseGabor == gf->bUseGabor) && (bUseIntensity == gf->bUseIntensity));

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
		if ( gf->bUseLBP)
		{
			gf->loadedFeatures[i].featLBP = (float*)malloc(sizeof(float) * gf->featLenLBP);
			fread(gf->loadedFeatures[i].featLBP, sizeof(float), gf->featLenLBP, pCurrent);
		}
		if ( gf->bUseGabor)
		{
			gf->loadedFeatures[i].featGabor = (float*)malloc(sizeof(float) * gf->featLenGabor);
			fread(gf->loadedFeatures[i].featGabor, sizeof(float), gf->featLenGabor, pCurrent);
		}
		if ( gf->bUseIntensity)
		{
			gf->loadedFeatures[i].featIntensity = (float*)malloc(sizeof(float) * gf->featLenIntensity);
			fread(gf->loadedFeatures[i].featIntensity, sizeof(float), gf->featLenIntensity, pCurrent);
		}

	}

	fclose(pFaceFeatBin);
	pCurrent = NULL;

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
					feature[ptrFeat] = hist[k];
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
				feature[featPtr] = hist[k];
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
				feature[featPtr] = hist[k];
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

		if ( sumDist < minDist)
		{
			matchedID = loadedFeatures[i].id;
			minDist = sumDist;
		}
	}

	//return 
	return matchedID;




}//end matchFaceID


float matchFeatureHistDist(float* feature1, float* feature2, int length)
{
	float	h1, h2;
	int		i;
	float	dist;
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
	return (dist / length);

}//end matchFeatureDist