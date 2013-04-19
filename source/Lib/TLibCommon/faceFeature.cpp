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
			tab[i] = 59;
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




inline void resetLBPHist(UInt* hist, int n)
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
			resetLBPHist(hist, numBins);
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
			resetLBPHist(hist, numBins);
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



void dumpFeatures(gFaceReco* gf, bool bAdd)
{
	FILE*		pFaceFeatBin;
	errno_t		err;

	if ( bAdd )	
	{
		//addition mode, wont clear exist records
		err = fopen_s(&pFaceFeatBin, gf->faceBinPath, "ab");
	}
	else
	{
		err = fopen_s(&pFaceFeatBin, gf->faceBinPath, "wb");
	}

	if (err != 0)
	{
		printf("Can't open feature binary file to write!\n");
		system("pause");
		exit(-1);
	}


	//write switches first
	fwrite(&(gf->bUseLBP), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->bUseGabor), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->bUseIntensity), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->featLenTotal), sizeof(int), 1, pFaceFeatBin);

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


	fclose(pFaceFeatBin);

}//end dumpFeatures



void loadFeatures(gFaceReco* gf)
{
	FILE*		pFaceFeatBin;
	FILE*		pCurrent;
	errno_t		err;
	bool		bUseLBP, bUseGabor, bUseIntensity;
	int			oneFeatLen;
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
	fseek( pFaceFeatBin, 0, SEEK_END);
	numFaces = (ftell(pFaceFeatBin) - ftell(pCurrent)) / oneFeatLenInByte;
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




}//end loadFeatures




