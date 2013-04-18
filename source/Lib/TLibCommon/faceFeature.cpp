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

void initGlobalStruct(gFaceReco* gf)
{
	gf->face = (UChar*)malloc(sizeof(int) * gf->faceWidth * gf->faceHeight);
	gf->face1 = (UChar*)malloc(sizeof(int) * gf->faceWidth1 * gf->faceHeight1);
	gf->face2 = (UChar*)malloc(sizeof(int) * gf->faceWidth2 * gf->faceHeight2);

	if ( gf->bUseLBP)
	{
		gf->features.featLBP = (float*)malloc(sizeof(float) * gf->featLenLBP);
		gf->LBPHist = (UInt*)malloc(sizeof(UInt) * gf->numBinsLBP);
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

	if ( gf->bUseWeight)
	{
		gf->weight = (float*)malloc(sizeof(float) * gf->featLenTotal);
	}




}

void freeGlobalStruct(gFaceReco* gf)
{
	if (gf->features.featLBP != NULL)
	{
		free(gf->features.featLBP);
		gf->features.featLBP = NULL;
	}

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


}


/*  Extract LBP Features  */
void extractLBPFeatures(gFaceReco* gf)
{
	int		i, j, gr, gc;
	int		numBins;
	int		ptrFeat, ptrImg;
	UChar	currVal;
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

	//extraction at face1(1/2 face0)
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


					if(currVal < (*(ptr - 1)-threshold)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
					if(currVal < (*(ptr + 1)-threshold)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
					if(currVal < (*(ptr - width)-threshold)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
					if(currVal < (*(ptr + width)-threshold)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

					if(currVal < (*(ptr - width + 1)-threshold)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
					if(currVal < (*(ptr - width - 1)-threshold)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
					if(currVal < (*(ptr + width + 1)-threshold)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
					if(currVal < (*(ptr + width - 1)-threshold)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;

				}
			}



}//end extractLBPFeatures

inline void resetLBPHist(UInt* hist, int n)
{
	int		i;
	for ( i = 0; i < n; i++)
	{
		hist[i] = 0;
	}
}
