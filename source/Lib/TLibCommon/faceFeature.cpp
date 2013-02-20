#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "faceFeature.h"

void config(FACE3D_Type * gf)
{
	gf->gwStep = 4;
	//gf->tWidth = 256;
	//gf->tHeight = 192;
	//gf->LBP_H_Step = 24;
	//gf->LBP_W_Step = 32;

	// York
	gf->tWidth			= 80;
	gf->tHeight			= 80;
	gf->LBP_H_Step		= 10;    
	gf->LBP_W_Step		= 10;

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

#if 0
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
#endif

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




void extractLBPFaceFeatures(unsigned char * imageData, int widthStep, FACE3D_Type * gf)
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
				featurePtr = 0;

				// reset. 2013.01.24
				memset(faceFeatures, 0, sizeof(float) * LBP_FACE_FEATURE_LEN );

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
int	 matchFace( float * queryFeat, FACE3D_Type * gf )
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

	unitFaceFeatClass* ptrCurFetchedData;

	// init.
	matchedFaceID	= 0;
	featEntryLen	= LBP_FACE_FEATURE_LEN;
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
		ptrFeatDistance[i]	= -1;

	}//end:	init. nearest neighbor.

	// match.

	for ( ptrOneLoadedData = 0; ptrOneLoadedData < loadedDataLen; ptrOneLoadedData += unitDataInByte )
	{
		if (ptrOneLoadedData >= (loadedDataLen-unitDataInByte-1) )	break;

		// load one face data.

		ptrCurFetchedData	= (unitFaceFeatClass*)(ptrLoadedData + ptrOneLoadedData);
		tarFeat				= ptrCurFetchedData->feature;
		currTarID			= ptrCurFetchedData->id;

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
			tmpDist			= (tarFeat[i])-(queryFeat[i]);

			if (tmpDist >= 0)
			{	sumDist		= sumDist + tmpDist;
			} 
			else
			{	sumDist		= sumDist - tmpDist;
			}
		}
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
		}
		else
		{
			// check if the distance is smaller.
			if ( ptrFeatDistance[idxBuf2Fill] > sumDist )
			{
				ptrFeatDistance[idxBuf2Fill]= sumDist;
				ptrBestDistID[idxBuf2Fill]	= currTarID;
				ptrUsedDistFlag[i]			= 1;
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
	}

	matchedFaceID		= tmpMostVotedID;

	// return.
	return matchedFaceID;

}//end: matchFace( FACE3D_Type * gf )
