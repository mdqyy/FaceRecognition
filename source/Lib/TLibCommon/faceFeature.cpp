#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "faceFeature.h"

#define GABOR_BIN_DIR "../../image/Gabor.bin"
#define MAX_FLOAT        3.402823466e+38F //< maximum single float number

void config(FACE3D_Type * gf)
{
	gf->gwStep = 5;
	//gf->tWidth = 256;
	//gf->tHeight = 192;
	//gf->LBP_H_Step = 24;
	//gf->LBP_W_Step = 32;

	// York
	gf->tWidth			= 80;
	gf->tHeight			= 80;
	gf->LBP_H_Step		= LBP_STEP;    
	gf->LBP_W_Step		= LBP_STEP;
	gf->LBP_H_Window    = LBP_WINDOW;
	gf->LBP_W_Window	= LBP_WINDOW;

	gf->RX0				= 0;
	gf->RY0				= 0;
	gf->RX1				= 80;
	gf->RY1				= 80;

	gf->featureLength	= 0;
}

void init(FACE3D_Type * gf, int width, int height)
{
	FILE * fp;
	int nGabors, gaborWSize;
	double tmpf;
	double * ptr;
	int k, m;
	int FRAME_WIDTH, FRAME_HEIGHT, W, H;
	int TN, TN1;

	FRAME_HEIGHT = gf->FRAME_HEIGHT = height;
	FRAME_WIDTH = gf->FRAME_WIDTH = width;

	gf->mask = (unsigned char *)malloc(FRAME_WIDTH * FRAME_HEIGHT);

	//configure the system
	config(gf);

	fp = fopen("gabor.dat", "rt");
	fscanf(fp, "%d", &nGabors);
	fscanf(fp, "%d", &gaborWSize);

	gf->gaborWSize = gaborWSize;
	gf->nGabors = nGabors;

	gf->gaborCoefficients = (double **)malloc(nGabors * 2 * sizeof(double *));

	//read real and imaginary parts of Gabor coefficients
	for(k=0; k<(nGabors*2); k++)
	{
		gf->gaborCoefficients[k] = (double *)malloc(gaborWSize * gaborWSize * sizeof(double));

		ptr = gf->gaborCoefficients[k];
		for(m=0; m<(gaborWSize * gaborWSize); m++)
		{
			fscanf(fp, "%f", &tmpf);
			ptr[m] = tmpf;
		}
	}
	fclose(fp);

	//normalized face images at different scales
	W = gf->tWidth;
	H = gf->tHeight;

	gf->fImage0 = (int *)malloc(W * H * sizeof(int));
	gf->fImage1 = (int *)malloc((W * H)/4 * sizeof(int));
	gf->fImage2 = (int *)malloc((W * H)/16 * sizeof(int));

	//feature vector
	W = gf->tWidth;
	H = gf->tHeight;
	TN = (W * H) / (gf->gwStep * gf->gwStep);
	TN = TN * 4;		

	W = gf->tWidth / 2;
	H = gf->tHeight / 2;
	TN1 = (W * H) / (gf->LBP_H_Step * gf->LBP_W_Step) * 128 * 2;

	gf->faceFeatures = (float *)malloc((TN + TN1) * sizeof(float));
	gf->featurePtr = 0;

	gf->LBPHist = (int *)malloc(256 * sizeof(int));

}





void initFaceFeature(FACE3D_Type * gf, int width, int height)
{
	FILE * fp;
	int nGabors, gaborWSize;
	//double tmpf;
	double * ptr;
	int k, m;
	int FRAME_WIDTH, FRAME_HEIGHT, W, H;
	int TN = 0, TN1 = 0;

	FRAME_HEIGHT = gf->FRAME_HEIGHT = height;
	FRAME_WIDTH = gf->FRAME_WIDTH = width;

	gf->mask = (unsigned char *)malloc(FRAME_WIDTH * FRAME_HEIGHT);

	//configure the system
	config(gf);

#if 1
	fp = fopen(GABOR_BIN_DIR, "rb");
	if(fp==NULL){
		printf("open file Gabor.bin failed!\n");fflush(stdout);
		exit(-1);
	}
	fread(&nGabors,sizeof(int),1,fp);
	fread(&gaborWSize, sizeof(int),1,fp);

	gf->gaborWSize = gaborWSize;
	gf->nGabors = nGabors;

	gf->gaborCoefficients = (double **)malloc(nGabors * 2 * sizeof(double *));

	//read real and imaginary parts of Gabor coefficients
	for(k=0; k<(nGabors*2); k++)
	{
		gf->gaborCoefficients[k] = (double *)malloc(gaborWSize * gaborWSize * sizeof(double));

		ptr = gf->gaborCoefficients[k];
		for(m=0; m<(gaborWSize * gaborWSize); m++)
		{
			fread(&ptr[m], sizeof(double), 1, fp);
			//fscanf(fp, "%f", &tmpf);
			//ptr[m] = tmpf;
		}
	}
	fclose(fp);
#endif

	//normalized face images at different scales
	W = gf->tWidth;
	H = gf->tHeight;

	gf->fImage0 = (int *)malloc(W * H * sizeof(int));
	gf->fImage1 = (int *)malloc((W * H)/4 * sizeof(int));
	gf->fImage2 = (int *)malloc((W * H)/16 * sizeof(int));

#if FLIP_MATCH
	gf->fImage0flip = (int *)malloc(W * H * sizeof(int));
	gf->fImage1flip = (int *)malloc((W * H)/4 * sizeof(int));
	gf->fImage2flip = (int *)malloc((W * H)/16 * sizeof(int));
	
#endif

	//feature vector
	W = gf->tWidth;
	H = gf->tHeight;
#if USE_GABOR
	TN = (W * H) / (gf->gwStep * gf->gwStep);
	TN += TN / 4;
	TN *= nGabors;
#endif

#if USE_LBP
	W = gf->tWidth / 2;
	H = gf->tHeight / 2;
	TN1 = ( (W - gf->LBP_W_Window) / gf->LBP_W_Step + 1) * ((H - gf->LBP_H_Window) / gf->LBP_H_Step + 1)  * NUM_BIN;
	W = W/2;
	H = H/2;
	TN1 += ( (W - gf->LBP_W_Window/2) / gf->LBP_W_Step + 1) * ((H - gf->LBP_H_Window/2) / gf->LBP_H_Step + 1)  * NUM_BIN;
#endif
	gf->faceFeatures = (float *)malloc((TN + TN1) * sizeof(float));
#if FLIP_MATCH
	gf->faceFeaturesFlip = (float *)malloc((TN + TN1) * sizeof(float));
#endif
	gf->featurePtr = 0;

	gf->LBPHist = (int *)malloc(NUM_BIN * sizeof(int));

	//reset moved here 2013.2.27
	memset(gf->faceFeatures, 0, sizeof(float) * (TN + TN1));

#if ROTATE_INVARIANT_LBP
	//build look up table for rotation invariant LBP.
	unsigned int x, count, tmpCount, pos, numLeftShift, tmp;
	bool bitVal;
	gf->lookupTable[0] = 0;
	gf->lookupTable[255] = 255;
	for (unsigned int i=1; i<255; i++)
	{
		//count num of most continuous 0s and it's heading position
		x = i;
		int j;
		// make the 1st bit 1
		for ( j = 0; j < 8; j++)
		{
			if ( x & 1)
				break;
			x = x >> 1;
		}
		count = 0;
		tmpCount = 0;
		pos = 0;
		for (j = 0; j < 8; j++)
		{
			bitVal = ( x >> j) & 1;
			if (!bitVal)
			{
				tmpCount++;
				if (tmpCount > count)
				{
					count = tmpCount;
					pos = j;
				}
			}
			else
			{
				tmpCount = 0;
			}
		}
		numLeftShift = 7 - pos;
		while (numLeftShift > 0)
		{
			//do left shift
			x <<= 1;
			tmp = x & 0x100; //take the left shifted bit;
			x = x & 0xFF; //keep 8 bit;
			if ( tmp != 0)
			{
				x +=1; // put the left shifted bit back to the 1st bit;
			}
			numLeftShift--;
		}
			

		// count is number to be right shifted
		gf->lookupTable[i] = x;
	}
#endif
			




}





void extractFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf)
{
	int RX0, RX1, RY0, RY1, i, j, k, r, c;
	unsigned char * ptrChar;
	int tWidth, tHeight, vR, vC, W, H, tWidth0, tWidth1, tWidth2;
	double rStep, cStep, sumF;
	int *fImage0, *fImage1, *fImage2, *ptr, *LBPHist;
	int sum, currVal, LBPVal;
	int colorR, colorG, colorB, lum, gwStep, gr, gc, gaborWSize, nGabors;
	double * ptrGF;
	float * faceFeatures;
	int featurePtr;
	int LBP_H_Step, LBP_W_Step;
	int LBPFlag[8];

	RX0 = gf->RX0;
	RX1 = gf->RX1;
	RY0 = gf->RY0;
	RY1 = gf->RY1;
	tWidth = gf->tWidth;
	tHeight = gf->tHeight;
	fImage0 = gf->fImage0;
	fImage1 = gf->fImage1;
	fImage2 = gf->fImage2;
	gaborWSize = gf->gaborWSize;
	nGabors = gf->nGabors;
	faceFeatures = gf->faceFeatures;
	LBP_H_Step = gf->LBP_H_Step;
	LBP_W_Step = gf->LBP_W_Step;
	LBPHist = gf->LBPHist;
	
	//----------------------------------------------------------
	//extract ROI
	rStep = ((RY1 - RY0) * 1.0) / tHeight;
	cStep = ((RX1 - RX0) * 1.0) / tWidth;

	for(r=0; r<tHeight; r++)
	for(c=0; c<tWidth; c++)
	{
		vR = (int)(RY0 + (r * rStep));
		vC = (int)(RX0 + (c * cStep));
		ptrChar = imageData + vR * widthStep + vC * 3;
		
		colorB = ptrChar[0];
		colorG = ptrChar[1];
		colorR = ptrChar[2];

		lum = (colorR + colorG + colorG + colorB) >> 2;

		*(fImage0 + r * tWidth + c) = lum;
	}

	tWidth0 = tWidth;
	tWidth1 = tWidth / 2;
	tWidth2 = tWidth / 4;

	//----------------------------------------------------------
	//down sample by 2

	H = tHeight / 2;
	W = tWidth / 2;

	for(r=0; r<H; r++)
	for(c=0; c<W; c++)
	{
		ptr = fImage0 + (2 * r) * tWidth0;
		sum = *ptr;	
		sum = sum + *(ptr + 1);
		sum = sum + *(ptr + tWidth0);
		sum = sum + *(ptr + tWidth0 + 1);

		sum = sum >> 2;
		*(fImage1 + r * W + c) = sum;
	}

	//----------------------------------------------------------
	//down sample by 4

	H = tHeight / 4;
	W = tWidth / 4;


	for(r=0; r<H; r++)
	for(c=0; c<W; c++)
	{
		ptr = fImage1 + (2 * r) * tWidth1;
		sum = *ptr;	
		sum = sum + *(ptr + 1);
		sum = sum + *(ptr + tWidth1);
		sum = sum + *(ptr + tWidth1 + 1);

		sum = sum >> 2;
		*(fImage2 + r * W + c) = sum;
	}

	//----------------------------------------------------------
	featurePtr = 0;
	
	//----------------------------------------------------------
	//extract Gabor coefficients at fImage0
	W = tWidth;
	H = tHeight;
	gwStep = gf->gwStep;

	for(gr=0; gr<=(H-gaborWSize); gr+=gwStep)
	for(gc=0; gc<=(W-gaborWSize); gc+=gwStep)
	{
		for(k=0; k<(nGabors*2); k++)
		{
			ptrGF = gf->gaborCoefficients[k];

			sumF = 0;
			for(i=0; i<gaborWSize; i++)
			{
				ptr = fImage0 + (gr + i) * tWidth0 + gc + j;//////  bug founded
				for(j=0; j<gaborWSize; j++)
				{
					sumF = sumF + (*ptr) * (*ptrGF);
					ptr++;
					ptrGF++;
				}
			}

			//save to the feature vector
			faceFeatures[featurePtr] = sumF;
			featurePtr++;
		}
	}


	//----------------------------------------------------------
	//extract Gabor coefficients at fImage1
	W = tWidth/2;
	H = tHeight/2;
	gwStep = gf->gwStep;

	for(gr=0; gr<=(H-gaborWSize); gr+=gwStep)
	for(gc=0; gc<=(W-gaborWSize); gc+=gwStep)
	{
		for(k=0; k<(nGabors*2); k++)
		{
			ptrGF = gf->gaborCoefficients[k];

			sumF = 0;
			for(i=0; i<gaborWSize; i++)
			{
				ptr = fImage1 + (gr + i) * tWidth1 + gc + j;
				for(j=0; j<gaborWSize; j++)
				{
					sumF = sumF + (*ptr) * (*ptrGF);
					ptr++;
					ptrGF++;
				}
			}

			//save to the feature vector
			faceFeatures[featurePtr] = sumF;
			featurePtr++;
		}
	}


	//----------------------------------------------------------
	//extract LBP features at fImage1

	W = tWidth / 2;
	H = tHeight / 2;
	for(gr=0; gr<=(H-LBP_H_Step); gr+=LBP_H_Step)
	for(gc=0; gc<=(W-LBP_W_Step); gc+=LBP_W_Step)
	{
		//reset
		for(i=0; i<256; i++) LBPHist[i] = 0; 

		for(i=1; i<(LBP_H_Step-1); i++)
		for(j=1; j<(LBP_W_Step-1); j++)
		{
			LBPVal = 0;

			ptr = fImage1 + (gr + i) * tWidth1 + ( gc + j);
			currVal = *ptr;

			if(currVal > (*(ptr - 1))) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
			if(currVal > (*(ptr + 1))) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
			if(currVal > (*(ptr - tWidth1))) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
			if(currVal > (*(ptr + tWidth1))) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

			if(currVal > (*(ptr - tWidth1 + 1))) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
			if(currVal > (*(ptr - tWidth1 - 1))) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
			if(currVal > (*(ptr + tWidth1 + 1))) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
			if(currVal > (*(ptr + tWidth1 - 1))) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;

			LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
		}

		//save to the feature vector
		for(i=0; i<256; i++)
		{
			faceFeatures[featurePtr] = LBPHist[i];
			featurePtr++;
		}
		
	
	}

	//----------------------------------------------------------
	//extract LBP features at fImage2

	W = tWidth / 4;
	H = tHeight / 4;
	for(gr=0; gr<=(H-LBP_H_Step); gr+=LBP_H_Step)
	for(gc=0; gc<=(W-LBP_W_Step); gc+=LBP_W_Step)
	{
		//reset
		for(i=0; i<256; i++) LBPHist[i] = 0; 

		for(i=1; i<(LBP_H_Step-1); i++)
		for(j=1; j<(LBP_W_Step-1); j++)
		{
			LBPVal = 0;

			ptr = fImage2 + (gr + i) * tWidth2 + ( gc + j);
			currVal = *ptr;

			if(currVal > (*(ptr - 1))) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
			if(currVal > (*(ptr + 1))) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
			if(currVal > (*(ptr - tWidth1))) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
			if(currVal > (*(ptr + tWidth1))) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

			if(currVal > (*(ptr - tWidth1 + 1))) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
			if(currVal > (*(ptr - tWidth1 - 1))) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
			if(currVal > (*(ptr + tWidth1 + 1))) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
			if(currVal > (*(ptr + tWidth1 - 1))) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;

			LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
		}

		//save to the feature vector
		for(i=0; i<256; i++)
		{
			faceFeatures[featurePtr] = LBPHist[i];
			featurePtr++;
		}
	
	}

}




//Extract Gabor Features in image0 & image1.
void extractGaborFeatures(FACE3D_Type* gf)
{
	int *fImage0, *fImage1, *ptr;
	int gwStep, numStep, gaborWSize, nGabors, featurePtr;
	float* faceFeatures;
	int tWidth, tHeight, tWidth1;
	double** gaborCoefficients;
	double* ptrGaborCoefsReal, *ptrGaborCoefsImg;
	int ptrImg, ptrFilter;
	int i, j, k, l, n;
	float tmpSumReal, tmpSumImg;

	tWidth = gf->tWidth;
	tHeight = gf->tHeight;
	assert( tWidth == tHeight);

	tWidth1 = tWidth / 2;
	fImage0 = gf->fImage0;				//original image
	fImage1 = gf->fImage1;				//downsampled image
	gwStep = gf->gwStep;				//gabor window step
	numStep = tWidth / gwStep;			//gabor window number
	gaborWSize = gf->gaborWSize;		//gabor kernel size
	gaborCoefficients = gf->gaborCoefficients;
	nGabors = gf->nGabors;
	faceFeatures = gf->faceFeatures;	//faceFeature ptr
	featurePtr = gf->featureLength;		//reset to current length to combine different features


	//extract gabor features in image0
	for (n = 0; n < nGabors; n++)
	{
		ptrGaborCoefsReal = gaborCoefficients[(n*2)];
		ptrGaborCoefsImg = gaborCoefficients[(n*2+1)];

		for (i = 0; i < numStep; i++)
		{
			for (j = 0; j < numStep; j++)
			{
				ptrImg = gwStep * (i * tWidth + j);
				ptrFilter = 0;
				tmpSumReal = 0;
				tmpSumImg = 0;

				//adjust pointer since kernel size is a odd number
				if ( i == (numStep -1) )
				{
					ptrImg -= tWidth;
				}
				if ( j == (numStep -1))
				{
					ptrImg -= 1;
				}
				
				//extract gabor feature in current window
				for ( k = 0; k < gaborWSize; k++)
				{
					for ( l = 0; l < gaborWSize; l++)
					{
						tmpSumReal += fImage0[ptrImg] * ptrGaborCoefsReal[ptrFilter];
						tmpSumImg += fImage0[ptrImg] * ptrGaborCoefsImg[ptrFilter];
						ptrImg++;
						ptrFilter++;
					}
					ptrImg += tWidth - gaborWSize;
				}

				faceFeatures[featurePtr] = tmpSumReal * tmpSumReal + tmpSumImg * tmpSumImg;
				featurePtr++;

			}	//end col steps
		}	//end row steps
	} //end different gabor kernels

	//extract features in downsampled image1
	numStep /= 2;

	for (n = 0; n < nGabors; n++)
	{
		ptrGaborCoefsReal = gaborCoefficients[(n*2)];
		ptrGaborCoefsImg = gaborCoefficients[(n*2+1)];

		for (i = 0; i < numStep; i++)
		{
			for (j = 0; j < numStep; j++)
			{
				ptrImg = gwStep * (i * tWidth1 + j);
				ptrFilter = 0;
				tmpSumReal = 0;
				tmpSumImg = 0;

				//adjust pointer since kernel size is a odd number
				if ( i == (numStep -1) )
				{
					ptrImg -= tWidth1;
				}
				if ( j == (numStep -1))
				{
					ptrImg -= 1;
				}
				
				//extract gabor feature in current window
				for ( k = 0; k < gaborWSize; k++)
				{
					for ( l = 0; l < gaborWSize; l++)
					{
						tmpSumReal += fImage1[ptrImg] * ptrGaborCoefsReal[ptrFilter];
						tmpSumImg += fImage0[ptrImg] * ptrGaborCoefsImg[ptrFilter];
						ptrImg++;
						ptrFilter++;
					}
					ptrImg += tWidth1 - gaborWSize;
				}

				faceFeatures[featurePtr] = tmpSumReal * tmpSumReal + tmpSumImg * tmpSumImg;
				featurePtr++;

			}	//end col steps
		}	//end row steps
	} //end different gabor kernels

	gf->featureLength = featurePtr;
	
}// end function







void extractLBPFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf, bool isFlip)
{
	int RX0, RX1, RY0, RY1, i, j, k, r, c;
	unsigned char * ptrChar;
	int tWidth, tHeight, vR, vC, W, H, tWidth0, tWidth1, tWidth2;
	double rStep, cStep, sumF;
	int *fImage0, *fImage1, *fImage2, *ptr, *LBPHist;
	int sum, currVal, LBPVal;
	int colorR, colorG, colorB, lum, gwStep, gr, gc;
	double * ptrGF;
	float * faceFeatures;
	int featurePtr;
	int LBP_H_Step, LBP_W_Step, LBP_H_Window, LBP_W_Window;
	int LBPFlag[8];

	RX0 = gf->RX0;
	RX1 = gf->RX1;
	RY0 = gf->RY0;
	RY1 = gf->RY1;
	tWidth = gf->tWidth;
	tHeight = gf->tHeight;
	if (!isFlip)
	{
		fImage0 = gf->fImage0;
		fImage1 = gf->fImage1;
		fImage2 = gf->fImage2;
		faceFeatures = gf->faceFeatures;
	}
#if FLIP_MATCH
	else
	{
		fImage0 = gf->fImage0flip;
		fImage1 = gf->fImage1flip;
		fImage2 = gf->fImage2flip;
		faceFeatures = gf->faceFeaturesFlip;
	}
#endif
	
	LBP_H_Step = gf->LBP_H_Step;
	LBP_W_Step = gf->LBP_W_Step;
	LBP_H_Window = gf->LBP_H_Window;
	LBP_W_Window = gf->LBP_W_Window;
	LBPHist = gf->LBPHist;

	//uniform LBP look up table
#if UNIFORM_LBP
	unsigned int  lookupTab[256];
	unsigned int  temp[8] = { 1, 2, 4, 8, 16, 32, 64, 128};
	unsigned int  curNum = 0;
	for (int i = 0; i < 256; i++ )
	{
		lookupTab[i] = 0xF4; // a non-uniform pattern of LBP
	}

	// fill-in the uniform patterns
	lookupTab[0] = 0; // flat
	lookupTab[255] = 255; // spot
	int pos = 0;

	for (int addTimes = 1; addTimes < 7; addTimes++)
	{
		for (int i = 0; i < 8; i++ )
		{
			curNum = 0;
			for (int j = 0; j < addTimes; j++ )
			{
				pos = i + j;
				pos = pos % 8;
				curNum += temp[ pos ];
			}
			lookupTab[curNum] = curNum;
		}
	}
#endif

	tWidth0 = tWidth;
	tWidth1 = tWidth / 2;
	tWidth2 = tWidth / 4;

#if 0 // 2013.2.19 downsampling moved out of this module
	//----------------------------------------------------------
	//extract ROI
	rStep = ((RY1 - RY0) * 1.0) / tHeight;
	cStep = ((RX1 - RX0) * 1.0) / tWidth;

	for(r=0; r<tHeight; r++)
		for(c=0; c<tWidth; c++)
		{
			vR = (int)(RY0 + (r * rStep));
			vC = (int)(RX0 + (c * cStep));
			ptrChar = imageData + vR * widthStep + vC * 3;

			colorB = ptrChar[0];
			colorG = ptrChar[1];
			colorR = ptrChar[2];

			//lum = (colorR + colorG + colorG + colorB) >> 2; //?
			lum = (colorR*38 + colorG*75 + colorB*15) >> 7;

			*(fImage0 + r * tWidth + c) = lum;
		}

		tWidth0 = tWidth;
		tWidth1 = tWidth / 2;
		tWidth2 = tWidth / 4;

		//----------------------------------------------------------
		//down sample by 2

		H = tHeight / 2;
		W = tWidth / 2;

		for(r=0; r<H; r++)
			for(c=0; c<W; c++)
			{
				ptr = fImage0 + (2 * r) * tWidth1;
				sum = *ptr;	
				sum = sum + *(ptr + 1);
				sum = sum + *(ptr + tWidth0);
				sum = sum + *(ptr + tWidth0 + 1);

				sum = sum >> 2;
				*(fImage1 + r * W + c) = sum;
			}

			
			//----------------------------------------------------------
			//down sample by 4

			H = tHeight / 4;
			W = tWidth / 4;


			for(r=0; r<H; r++)
				for(c=0; c<W; c++)
				{
					ptr = fImage1 + (2 * r) * tWidth2;
					sum = *ptr;	
					sum = sum + *(ptr + 1);
					sum = sum + *(ptr + tWidth1);
					sum = sum + *(ptr + tWidth1 + 1);

					sum = sum >> 2;
					*(fImage2 + r * W + c) = sum;
				}

#endif

				//----------------------------------------------------------
				featurePtr = gf->featureLength; //2013.2.20 in order to combine different features

				// reset. 2013.01.24
				//memset(faceFeatures, 0, sizeof(float) * FACE_FEATURE_LEN );

				//----------------------------------------------------------
				//extract Gabor coefficients at fImage0
				
					//----------------------------------------------------------
					//extract Gabor coefficients at fImage1
					//----------------------------------------------------------
#if 0
						//extract LBP features at fImage0

						W = tWidth ;
						H = tHeight ;
						for(gr=0; gr<=(H-LBP_H_Step); gr+=LBP_H_Step)
						{
							for(gc=0; gc<=(W-LBP_W_Step); gc+=LBP_W_Step)
							{
								//reset
								for(i=0; i<256; i++) LBPHist[i] = 0; 

								for(i=1; i<(LBP_H_Step-1); i++) 
								{
									for(j=1; j<(LBP_W_Step-1); j++)
									{
										LBPVal = 0;

										ptr = fImage1 + (gr + i) * tWidth1 + ( gc + j);
										currVal = *ptr;

										if(currVal > (*(ptr - 1))) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
										if(currVal > (*(ptr + 1))) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
										if(currVal > (*(ptr - tWidth1))) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
										if(currVal > (*(ptr + tWidth1))) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

										if(currVal > (*(ptr - tWidth1 + 1))) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
										if(currVal > (*(ptr - tWidth1 - 1))) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
										if(currVal > (*(ptr + tWidth1 + 1))) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
										if(currVal > (*(ptr + tWidth1 - 1))) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;
#if UNIFORM_LBP
										LBPHist[lookupTab[LBPVal]] ++;
#else
										LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
#endif
									}
								}

									//save to the feature vector
								
									for(i=0; i<256; i++)
									{
										faceFeatures[featurePtr] = LBPHist[i];
										featurePtr++;
									}


							}
						}
#endif
						//----------------------------------------------------------
						//extract LBP features at fImage1

						W = tWidth / 2;
						H = tHeight / 2;
						for(gr=0; gr<=(H-LBP_H_Window); gr+=LBP_H_Step)
						{
							for(gc=0; gc<=(W-LBP_W_Window); gc+=LBP_W_Step)
							{
								//reset
								for(i=0; i<256; i++) LBPHist[i] = 0; 

								for(i=1; i<(LBP_H_Window-1); i++) 
								{
									for(j=1; j<(LBP_W_Window-1); j++)
									{
										LBPVal = 0;

										ptr = fImage1 + (gr + i) * tWidth1 + ( gc + j);
										currVal = *ptr;


										if(currVal < (*(ptr - 1)-THRESHOLD)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
										if(currVal < (*(ptr + 1)-THRESHOLD)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
										if(currVal < (*(ptr - tWidth1)-THRESHOLD)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
										if(currVal < (*(ptr + tWidth1)-THRESHOLD)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

										if(currVal < (*(ptr - tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
										if(currVal < (*(ptr - tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
										if(currVal < (*(ptr + tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
										if(currVal < (*(ptr + tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;
#if UNIFORM_LBP
										LBPHist[lookupTab[LBPVal]] ++;
#elif ROTATE_INVARIANT_LBP

										LBPHist[gf->lookupTable[LBPVal]] ++;
#else
										LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
#endif
									}
								}

									//save to the feature vector
								
									for(i=0; i<256; i++)
									{
										faceFeatures[featurePtr] = LBPHist[i];
										featurePtr++;
									}


							}
						}

							//----------------------------------------------------------
							//extract LBP features at fImage2

							W = tWidth / 4;
							H = tHeight / 4;
							LBP_H_Window /= 2;
							LBP_W_Window /= 2;
							for(gr=0; gr<=(H-LBP_H_Window); gr+=LBP_H_Step)
								for(gc=0; gc<=(W-LBP_W_Window); gc+=LBP_W_Step)
								{
									//reset
									for(i=0; i<256; i++) LBPHist[i] = 0; 

									for(i=1; i<(LBP_H_Window-1); i++)
									{
										for(j=1; j<(LBP_W_Window-1); j++)
										{
											LBPVal = 0;

											ptr = fImage2 + (gr + i) * tWidth2 + ( gc + j);
											currVal = *ptr;

											if(currVal < (*(ptr - 1)-THRESHOLD)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
											if(currVal < (*(ptr + 1)-THRESHOLD)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
											if(currVal < (*(ptr - tWidth1)-THRESHOLD)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
											if(currVal < (*(ptr + tWidth1)-THRESHOLD)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

											if(currVal < (*(ptr - tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
											if(currVal < (*(ptr - tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
											if(currVal < (*(ptr + tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
											if(currVal < (*(ptr + tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;

#if UNIFORM_LBP
										LBPHist[lookupTab[LBPVal]] ++;
#elif ROTATE_INVARIANT_LBP

										LBPHist[gf->lookupTable[LBPVal]] ++;
#else
										LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
#endif
										}
									}

										//save to the feature vector
										for(i=0; i<256; i++)
										{
											faceFeatures[featurePtr] = LBPHist[i];
											featurePtr++;
										}

								}


								// record feature length.
								gf->featureLength	= featurePtr;

}//end: void extractLBPFaceFeatures()










//global binary feature
void extractGBPFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf)
{
	int RX0, RX1, RY0, RY1, i, j, k, r, c;
	unsigned char * ptrChar;
	int tWidth, tHeight, vR, vC, W, H, tWidth0, tWidth1, tWidth2;
	double rStep, cStep, sumF;
	int *fImage0, *fImage1, *fImage2, *ptr, *LBPHist;
	int sum, currVal, LBPVal;
	int colorR, colorG, colorB, lum, gwStep, gr, gc, gaborWSize, nGabors;
	double * ptrGF;
	float * faceFeatures;
	int featurePtr;
	int LBP_H_Step, LBP_W_Step;
	int LBPFlag[8];

	RX0 = gf->RX0;
	RX1 = gf->RX1;
	RY0 = gf->RY0;
	RY1 = gf->RY1;
	tWidth = gf->tWidth;
	tHeight = gf->tHeight;
	fImage0 = gf->fImage0;
	fImage1 = gf->fImage1;
	fImage2 = gf->fImage2;
	gaborWSize = gf->gaborWSize;
	nGabors = gf->nGabors;
	faceFeatures = gf->faceFeatures;
	LBP_H_Step = gf->LBP_H_Step;
	LBP_W_Step = gf->LBP_W_Step;
	LBPHist = gf->LBPHist;

	//uniform LBP look up table


	tWidth0 = tWidth;
	tWidth1 = tWidth / 2;
	tWidth2 = tWidth / 4;



				//----------------------------------------------------------
				featurePtr = gf->featureLength; //2013.2.20 in order to combine different features

				// reset. 2013.01.24
				//memset(faceFeatures, 0, sizeof(float) * FACE_FEATURE_LEN );

				//----------------------------------------------------------
				//extract Gabor coefficients at fImage0
				
					//----------------------------------------------------------
					//extract Gabor coefficients at fImage1
					//----------------------------------------------------------

						//----------------------------------------------------------
						//extract LBP features at fImage1

						W = tWidth / 2;
						H = tHeight / 2;
						currVal = 0;
						for( i = 0; i < W*H; i++)
						{
							currVal += fImage1[i];
						}
						currVal /= W*H; //mean value of entire frame.

						for(gr=0; gr<=(H-LBP_H_Step); gr+=LBP_H_Step)
						{
							for(gc=0; gc<=(W-LBP_W_Step); gc+=LBP_W_Step)
							{
								//reset
								for(i=0; i<256; i++) LBPHist[i] = 0; 

								for(i=1; i<(LBP_H_Step-1); i++) 
								{
									for(j=1; j<(LBP_W_Step-1); j++)
									{
										LBPVal = 0;

										ptr = fImage1 + (gr + i) * tWidth1 + ( gc + j);
										//currVal = *ptr;


										if(currVal < (*(ptr - 1)-THRESHOLD)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
										if(currVal < (*(ptr + 1)-THRESHOLD)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
										if(currVal < (*(ptr - tWidth1)-THRESHOLD)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
										if(currVal < (*(ptr + tWidth1)-THRESHOLD)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

										if(currVal < (*(ptr - tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
										if(currVal < (*(ptr - tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
										if(currVal < (*(ptr + tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
										if(currVal < (*(ptr + tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;
#if UNIFORM_LBP
										LBPHist[lookupTab[LBPVal]] ++;
#else
										LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
#endif
									}
								}

									//save to the feature vector
								
									for(i=0; i<256; i++)
									{
										faceFeatures[featurePtr] = LBPHist[i];
										featurePtr++;
									}


							}
						}

							//----------------------------------------------------------
							//extract LBP features at fImage2

							W = tWidth / 4;
							H = tHeight / 4;

							currVal = 0;
							for( i = 0; i < W*H; i++)
							{
								currVal += fImage1[i];
							}
							currVal /= W*H; //mean value of entire frame.

							for(gr=0; gr<=(H-LBP_H_Step); gr+=LBP_H_Step)
								for(gc=0; gc<=(W-LBP_W_Step); gc+=LBP_W_Step)
								{
									//reset
									for(i=0; i<256; i++) LBPHist[i] = 0; 

									for(i=1; i<(LBP_H_Step-1); i++)
									{
										for(j=1; j<(LBP_W_Step-1); j++)
										{
											LBPVal = 0;

											ptr = fImage2 + (gr + i) * tWidth2 + ( gc + j);
											//currVal = *ptr;

											if(currVal < (*(ptr - 1)-THRESHOLD)) LBPVal = LBPVal + 1;				//LBPFlag[0] = 1;
											if(currVal < (*(ptr + 1)-THRESHOLD)) LBPVal = LBPVal + 2;				//LBPFlag[1] = 1;
											if(currVal < (*(ptr - tWidth1)-THRESHOLD)) LBPVal = LBPVal + 4;		//LBPFlag[2] = 1;
											if(currVal < (*(ptr + tWidth1)-THRESHOLD)) LBPVal = LBPVal + 8;		//LBPFlag[3] = 1;

											if(currVal < (*(ptr - tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 16;	//LBPFlag[4] = 1;
											if(currVal < (*(ptr - tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 32;	//LBPFlag[5] = 1;
											if(currVal < (*(ptr + tWidth1 + 1)-THRESHOLD)) LBPVal = LBPVal + 64;	//LBPFlag[6] = 1;
											if(currVal < (*(ptr + tWidth1 - 1)-THRESHOLD)) LBPVal = LBPVal + 128;	//LBPFlag[7] = 1;
#if UNIFORM_LBP
										LBPHist[lookupTab[LBPVal]] ++;
#else
											LBPHist[LBPVal] = LBPHist[LBPVal] + 1;
#endif
										}
									}

										//save to the feature vector
										for(i=0; i<256; i++)
										{
											faceFeatures[featurePtr] = LBPHist[i];
											featurePtr++;
										}

								}


								// record feature length.
								gf->featureLength	= featurePtr;

}//end: void extractGBPFaceFeatures()


void extractColorTag(unsigned char * imageData, int widthStep, FACE3D_Type * gf)
{
	int i, j;
	unsigned char * ptr, *mask, *maskPtr;
	int r, g, b;
	float rc, gc, bc;
	float h, s, v, max, min, delta;
	int FRAME_WIDTH, FRAME_HEIGHT;

	FRAME_HEIGHT = gf->FRAME_HEIGHT;
	FRAME_WIDTH = gf->FRAME_WIDTH;

	mask = gf->mask;

	for(i=0; i<FRAME_HEIGHT; i++)
	for(j=0; j<FRAME_WIDTH; j++)
	{
		ptr = imageData + i * widthStep + j * 3;
		maskPtr = mask + i * FRAME_WIDTH + j;

		r = ptr[2];
		g = ptr[1];
		b = ptr[0];

		rc = (float)r / 255.0;
		gc = (float)g / 255.0;
		bc = (float)b / 255.0;

		max = rc; 
		if(gc > max) max = gc;
		if(bc > max) max = bc;

		min = rc;
		if(gc < min) min = gc;
		if(bc < min) min = bc;

		delta = max - min;
		v = max;

		if (max != 0.0)
			s = delta / max;
		else
			s = 0.0;

		if (s == 0.0) 
		{
			h = 0.0; 
		}
		else {
			if (rc == max)
				h = (gc - bc) / delta;
			else if (gc == max)
				h = 2 + (bc - rc) / delta;
			else if (bc == max)
				h = 4 + (rc - gc) / delta;

			h *= 60.0;
			if (h < 0)	h += 360.0;
		}


		if((v > 0.45) && (s > 0.25) && (h>150) && (h<250)) 
			*maskPtr  = 1;
		else
			*maskPtr = 0;

	}

}


void loadFaceData( FACE3D_Type * gf )
{
	FILE *fpFaceDataFile	= fopen( FACE_DATUM_FILENAME, "rb");
	long lSize;
	size_t resultReadFile;

	if(fpFaceDataFile==NULL){
		printf("open file %s failed!\n", FACE_DATUM_FILENAME);fflush(stdout);
		exit(-1);
	}

	/* obtain file size.*/
	fseek( fpFaceDataFile, 0, SEEK_END );
	lSize					= ftell( fpFaceDataFile );
	rewind( fpFaceDataFile );

	gf->bufFaceDataLen		= lSize;

	/* allocate memory to contain loaded face data.*/
	gf->bufferFaceData		= (unsigned char*)malloc( sizeof(unsigned char) * lSize );
	if( (gf->bufferFaceData) == NULL) printf("\nMemory Error Allocating Face Data Buffer!\n");

	/* read data from file.*/
	resultReadFile			= fread( gf->bufferFaceData, 1, lSize, fpFaceDataFile );

	//
	fclose(fpFaceDataFile);

}//end: loadFaceData( FACE3D_Type * gf )


/************************************************************************/
/* Match face feature among loaded ones.                                */
/************************************************************************/
int	 matchFace( FACE3D_Type * gf )
{
	float			*queryFeat = gf->faceFeatures;
#if FLIP_MATCH
	float			*queryFeatFlip = gf->faceFeaturesFlip;
#endif
	int				matchedFaceID;
	int				featEntryLen;
	int				i, idxBuf2Fill;
	int				loadedDataLen;
	int				unitDataInByte;
	int				ptrOneLoadedData;
	int				currTarID;
	int				idVoted, tmpMostVotedID, tmpMostVote;
	float			sumDist, tmpDist;
	float			maxMatchingDistance;
	float*			tarFeat;
	float*			ptrFeatDistance;
	int*			ptrUsedDistFlag;
	int*			ptrBestDistID;
	int*			cntIDVote;
	unsigned char*	ptrLoadedData;
#if DEBUG_MODE
	char			curImageName[200];
	char*			bestDistIDFileName[NUM_NEAREST_NBOR+1];
	for (int kk = 0; kk<NUM_NEAREST_NBOR+1;kk++)
		bestDistIDFileName[kk] = gf->bestDistImageName[kk];
#endif

	unitFaceFeatClass* ptrCurFetchedData;

	// init.
	matchedFaceID	= 0;
	featEntryLen	= TOTAL_FEATURE_LEN;
	ptrFeatDistance = gf->featDistance;
	ptrUsedDistFlag = gf->usedDistFlag;
	ptrBestDistID	= gf->bestDistID;
	loadedDataLen	= gf->bufFaceDataLen;
	ptrLoadedData	= gf->bufferFaceData;
	cntIDVote		= gf->voteCntFaceID;
	unitDataInByte	= sizeof(unitFaceFeatClass);


	for (i=0; i<NUM_NEAREST_NBOR; i++)
	{
		ptrBestDistID[i]	= 0;
		ptrUsedDistFlag[i]	= 0;
		ptrFeatDistance[i]	= MAX_FLOAT;

	}//end:	init. nearest neighbor.

	// match.

	for ( ptrOneLoadedData = 0; ptrOneLoadedData < loadedDataLen; ptrOneLoadedData += unitDataInByte )
	{
		if (ptrOneLoadedData >= (loadedDataLen-unitDataInByte-1) )	break;

		// load one face data.

		ptrCurFetchedData	= (unitFaceFeatClass*)(ptrLoadedData + ptrOneLoadedData);
		tarFeat				= ptrCurFetchedData->feature;
		currTarID			= ptrCurFetchedData->id;
#if DEBUG_MODE
		for (int kk = 0; kk<200; kk++)
			curImageName[kk]		= ptrCurFetchedData->imagename[kk];
#endif


		// distance.

		sumDist				= 0;

#if KAI_DISTANCE // use normalized distance
		for (i=0; i<featEntryLen; i++)
		{
			tmpDist			= (tarFeat[i])-(queryFeat[i]);
			if( !(tarFeat[i] == 0 && queryFeat[i] ==0))
			{
				sumDist += (tmpDist * tmpDist)/(tarFeat[i] + queryFeat[i]);
			}
		}
#else
		for (i=0; i<featEntryLen; i++)
		{
			tmpDist			= (float)(tarFeat[i])-(queryFeat[i]);

			if (tmpDist >= 0)
			{	sumDist		= sumDist + tmpDist;
			} 
			else
			{	sumDist		= sumDist - tmpDist;
			}
		}
#endif

#if FLIP_MATCH
		float sumDistFlip = 0;
		for (i=0; i<featEntryLen; i++)
		{
			tmpDist			= (float)(tarFeat[i])-(queryFeatFlip[i]);

			if (tmpDist >= 0)
			{	sumDistFlip		+=tmpDist;
			} 
			else
			{	sumDistFlip		-=tmpDist;
			}
		}
		sumDist = (sumDistFlip < sumDist)? sumDistFlip:sumDist; //take smaller distance of original or flipped image
#endif


		// fill the top nearest neighbor.

		maxMatchingDistance	= -2;
		idxBuf2Fill			= 0;

		for (i=0; i<NUM_NEAREST_NBOR; i++)
		{
			if (ptrUsedDistFlag[i] == 0)
			{	
				idxBuf2Fill			= i;
				break;
			}

			if (ptrFeatDistance[i] > maxMatchingDistance)
			{
				idxBuf2Fill			= i;
				maxMatchingDistance = ptrFeatDistance[i];
			}

		}//end: find slot to fill.

	
		if ( ptrUsedDistFlag[idxBuf2Fill]	== 0 )
		{
			// the first time fill this buffer.
			// fill it anyway.
			ptrBestDistID[idxBuf2Fill]		= currTarID;
			ptrFeatDistance[idxBuf2Fill]	= sumDist;
			ptrUsedDistFlag[idxBuf2Fill]	= 1;
#if DEBUG_MODE
			for (int kk = 0; kk<200; kk++)
			bestDistIDFileName[idxBuf2Fill][kk] = curImageName[kk];
#endif
		}
		else
		{
			// check if the distance is smaller.
			if ( ptrFeatDistance[idxBuf2Fill] > sumDist )
			{
				ptrFeatDistance[idxBuf2Fill]= sumDist;
				ptrBestDistID[idxBuf2Fill]	= currTarID;
				ptrUsedDistFlag[i]			= 1;
#if DEBUG_MODE
				for (int kk = 0; kk<200; kk++)
				bestDistIDFileName[idxBuf2Fill][kk] = curImageName[kk];
#endif
			}
			
		}
		
	}//end: going through each archived face.


	// vote.

	for (i=0; i<MAX_FACE_ID; i++)
	{	
		cntIDVote[i]	= 0;
	}

	for (i=0; i<NUM_NEAREST_NBOR; i++)
	{
		if (ptrUsedDistFlag[i] == 1)
		{
			idVoted				= ptrBestDistID[i];
			cntIDVote[idVoted]	= cntIDVote[idVoted] + 1;
		}	
	}

	tmpMostVote			= -1;
	tmpMostVotedID		= 0;

	for (i=0; i<MAX_FACE_ID; i++)
	{
		if ( cntIDVote[i] > tmpMostVote )
		{
			tmpMostVotedID	= i;
			tmpMostVote		= cntIDVote[i];
		}
		else if ( cntIDVote[i] == tmpMostVote )  // fixed a bug when equal votes happens which is quite frequently occurs.
		{
			float distOld = MAX_FLOAT, distNew = MAX_FLOAT;
			for ( int jj = 0; jj < NUM_NEAREST_NBOR; jj++)
			{
				if ( (tmpMostVotedID == ptrBestDistID[jj]) && ( ptrFeatDistance[jj] < distOld))
					distOld = ptrFeatDistance[jj];
				if ( (i == ptrBestDistID[jj]) && ( ptrFeatDistance[jj] < distNew))
					distNew = ptrFeatDistance[jj];
			}
			if (distOld > distNew)
			{
				tmpMostVotedID = i;
			}
		}

	}

	//This fixed a bug that if the max vote == 1, the matchedID will always be the first. 2013.2.27
#if 0
	if (tmpMostVote == 1)
	{
		float minDistInBin = ptrFeatDistance[0];
		for(i = 1; i<NUM_NEAREST_NBOR; i++)
		{
			if ( ptrFeatDistance[i] < minDistInBin)
			{
				minDistInBin = ptrFeatDistance[i];
				tmpMostVotedID = ptrBestDistID[i];
			}
		}
	}
#endif



	matchedFaceID		= tmpMostVotedID;

	// return.
	return matchedFaceID;

}//end: matchFace( FACE3D_Type * gf )

#if WEIGHTED_MATCH
void trainWeight( float* weight, FACE3D_Type * gf)
{
	int				matchedFaceID;
	int				featEntryLen;
	int				i, idxBuf2Fill;
	int				loadedDataLen;
	int				unitDataInByte;
	int				ptrOneLoadedData;
	int				currTarID;
	int				idVoted, tmpMostVotedID, tmpMostVote;
	float			sumDist, tmpDist;
	float			maxMatchingDistance;
	float*			tarFeat;
	float*			ptrFeatDistance;
	int*			ptrUsedDistFlag;
	int*			ptrBestDistID;
	int*			cntIDVote;
	unsigned char*	ptrLoadedData;
	long int        inClassDist[MAX_FACE_ID][FACE_FEATURE_LEN];

	unitFaceFeatClass* ptrCurFetchedData;

	// init.
	matchedFaceID	= 0;
	featEntryLen	= FACE_FEATURE_LEN;
	ptrFeatDistance = gf->featDistance;
	ptrUsedDistFlag = gf->usedDistFlag;
	ptrBestDistID	= gf->bestDistID;
	loadedDataLen	= gf->bufFaceDataLen;
	ptrLoadedData	= gf->bufferFaceData;
	cntIDVote		= gf->voteCntFaceID;
	unitDataInByte	= sizeof(unitFaceFeatClass);

	for (i = 0; i< MAX_FACE_ID * FACE_FEATURE_LEN; i++)
	{
		inClassDist[i] = 0;
		weight[i]
	}


	// match.

	for ( ptrOneLoadedData = 0; ptrOneLoadedData < loadedDataLen; ptrOneLoadedData += unitDataInByte )
	{
		if (ptrOneLoadedData >= (loadedDataLen-unitDataInByte-1) )	break;

		// load one face data.

		ptrCurFetchedData	= (unitFaceFeatClass*)(ptrLoadedData + ptrOneLoadedData);
		tarFeat				= ptrCurFetchedData->feature;
		currTarID			= ptrCurFetchedData->id;

		// distance.

#if KAI_DISTANCE // use normalized distance
		for (i=0; i<featEntryLen; i++)
		{
			tmpDist			= (tarFeat[i])-(queryFeat[i]);
			if( !(tarFeat[i] == 0 && queryFeat[i] ==0))
			{
				inClassDist[currTarID][i] += (tmpDist * tmpDist)/(tarFeat[i] + queryFeat[i]);
			}
		}
#else
		for (i=0; i<featEntryLen; i++)
		{
			tmpDist			= (tarFeat[i])-(queryFeat[i]);

			if (tmpDist >= 0)
			{	inClassDist[currTarID][i] += tmpDist;
			} 
			else
			{	inClassDist[currTarID][i] -= tmpDist;
			}
		}
#endif
	} //end loading features
	




}// end function trainWeight

#endif // weighted match