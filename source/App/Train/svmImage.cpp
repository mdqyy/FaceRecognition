

#include <cv.h>
#include <time.h>
#include <highgui.h>
#include <math.h>
//#include "stdafx.h"
#include <windows.h>
#include <mmsystem.h>
#include <stdio.h>

#include "Global.h"
#include "Define.h"

#include "svmImage.h"
#include "features.h"

	//BK	BU	GY	OR	RD	WH	YL
static int modelTree[6][7]={
	{1,		-1,	 1,	-1,	-1,	 1,	-1},	//model 0
	{1,		 0,	-1,	 0,	 0,	-1,  0},	//model 1
	{0,		 0, -1,	 0,	 0,	 1,	 0},	//model 2
	{0,		 1,  0,	-1,	 1,	 0,	-1},	//model 3
	{0,		-1,  0,  0,  1,  0,  0},	//model 4
	{0,		 0,  0,	 1,	 0,	 0,	-1}};	//model 5	



void initSVMTraning(SVM_GST * gst)
{
	int numTotal, featLength;
	int i, j;
	bool tmpLabel;
	double tmp;
	FILE* pF = fopen("C:/Users/Zhi/Desktop/tmp/svmTrain.bin","rb");
	FILE* pLabel = fopen("C:/Users/Zhi/Desktop/tmp/svmLabel.bin","rb");

	fread(&numTotal, sizeof(int), 1, pLabel);
	fread(&featLength, sizeof(int), 1, pLabel);
	for ( i = 0; i < numTotal; i++)
	{
		fread(&tmpLabel, sizeof(bool), 1, pLabel);
		gst->classLable[i] = tmpLabel? 1: -1;
	}
	gst->featureSize = featLength;
	gst->nSamples = numTotal;

	//tmp = gst->features[numTotal-1][featLength-1];

	for ( i = 0; i < numTotal; i++)
	{
		for ( j = 0; j < featLength; j++)
		{
			fread(&tmp, sizeof(double), 1, pF);
			gst->features[i][j] = tmp;
		}
	}

	fclose(pF);
	fclose(pLabel);

	
	
}

void initSystem(SVM_GST * gst)
{
	double ** features, **featuresNew;
	int k;

	gst->sampleLable = (int *)malloc(MAX_SAMPLE_SIZE * sizeof(int));
	gst->classLable = (int *)malloc(MAX_SAMPLE_SIZE * sizeof(int));

	features = (double **)malloc(MAX_SAMPLE_SIZE * sizeof(double *));
	for(k=0; k<MAX_SAMPLE_SIZE; k++)
		features[k] = (double *)malloc(MAX_FEATURE_SIZE * sizeof(double));

	//featuresNew = (double **)malloc(MAX_SAMPLE_SIZE * sizeof(double *));
	//for(k=0; k<MAX_SAMPLE_SIZE; k++)
		//featuresNew[k] = (double *)malloc(MAX_FEATURE_SIZE * sizeof(double));


	gst->features = features;
	//gst->featuresNew = featuresNew;

	gst->nClasses = 2;

	//initSVMTraning(gst);
}



void generateSVMTrainingData(SVM_GST * gst)
{
	char * path;
	char * imageListFileName;
	char ss[256], tt[256];
	char * tmpStr;
	FILE * imageListFP;
	int ct, lable, i, len, n, k, nSamples;
	int * sampleLable, * classLable;
	double * feature;
	double ** featuresNew;
	IplImage *src		= 0;


	sampleLable = gst->sampleLable;
	classLable = gst->classLable;
	path = gst->path;
	imageListFileName = gst->imageListFileName;
	featuresNew = gst->featuresNew;

	sprintf(ss, "%s%s", path, imageListFileName);
	imageListFP = fopen(ss, "rt");

	fscanf(imageListFP, "%d", &nSamples);

	gst->featureSize = 144;

	ct = 0;
	while(ct<(nSamples-1))
	{
		fscanf(imageListFP, "%d", &lable);
		sampleLable[ct] = lable;

		fgetc(imageListFP);						//remove the blank space

		fgets(ss, 200, imageListFP);
		len = sprintf(tt, "%s%s\0", path, ss);
		tt[len-1] = 0;							//the last char should be \0

		src = cvLoadImage(tt, CV_LOAD_IMAGE_COLOR);

		feature = gst->features[ct];

		extractImageFeatures_Type1((unsigned char *)src->imageData, 0, 0, src->height, src->width, src->widthStep, 
			feature, gst->featureSize);
		

		//printf("\nSample %6d %s", ct, tt);
		//cvShowImage("Image", src);
		//cvWaitKey(1);

		ct++;
	}


	fclose(imageListFP);
	gst->nSamples = ct;


#if 1
	int sampleIncluded, tmpLB;

	//training binary tree models
	for(n=0; n<(gst->nClasses-1); n++)
	{

		//now train SVM model for each class
		for(i=0; i<MAX_SAMPLE_SIZE; i++) classLable[i] = 0;

		ct = 0; 
		for(k=0; k<gst->nSamples; k++)
		{
			tmpLB = sampleLable[k];
			sampleIncluded = modelTree[n][tmpLB];

			if((sampleIncluded == 1) || (sampleIncluded == -1))
			{
				if(sampleIncluded == 1)
				{
					//positive
					classLable[ct] = 1;

					//copy the feature data
					for(i=0; i<gst->featureSize; i++)
						featuresNew[ct][i] = gst->features[k][i];
					
					ct++;

				}
				else
				{
					//negative
					classLable[ct] = -1;

					//copy the feature data
					for(i=0; i<gst->featureSize; i++)
						featuresNew[ct][i] = gst->features[k][i];
					
					ct++;

				}
			}
		}


		sprintf(ss, "%s%03d.mod", path, n);
		svmTraining(featuresNew, ct, gst->featureSize, classLable, ss);
		//getchar();
			
	}

#else

	//training binary tree models
	for(n=0; n<(gst->nClasses-1); n++)
	{

		//now train SVM model for each class
		for(i=0; i<MAX_SAMPLE_SIZE; i++) classLable[i] = 0;

		ct = 0; 
		for(k=0; k<gst->nSamples; k++)
		{
			if(sampleLable[k] >=n)
			{
				if(sampleLable[k] == n)
				{
					//positive
					classLable[ct] = 1;

					//copy the feature data
					for(i=0; i<gst->featureSize; i++)
						featuresNew[ct][i] = gst->features[k][i];
					
					ct++;

				}
				else
				{
					//negative
					classLable[ct] = -1;

					//copy the feature data
					for(i=0; i<gst->featureSize; i++)
						featuresNew[ct][i] = gst->features[k][i];
					
					ct++;

				}
			}
		}


		sprintf(ss, "%s%03d.mod", path, n);
		svmTraining(featuresNew, ct, gst->featureSize, classLable, ss);
		//getchar();
			
	}
#endif

}


void svmTraining(double ** features, int nSample, int featureSize, int * sampleLable, 
				 char * modelFileName)
{
	int i;

	char *svmTrainFile = "../trainFile";

	FILE *fp = fopen(svmTrainFile,"wb");

	int version = 200;
	int data_typeid = 6;  //feature default
	int target_typeid = 3; //target +/-1 int 

	fwrite(&version, sizeof(int), 1, fp);
	fwrite(&data_typeid, sizeof(int), 1, fp);
	fwrite(&target_typeid, sizeof(int), 1, fp);
	fwrite(&nSample, sizeof(int), 1, fp);
	fwrite(&featureSize, sizeof(int), 1, fp);
	fclose (fp);


	fp = fopen(svmTrainFile,"ab");

	for (i=0;i<nSample;i++)
	{
		fwrite(&sampleLable[i], sizeof(int), 1, fp);
		//printf("%d\n",sampleLable[i]);
		fwrite(features[i],sizeof(double),featureSize,fp);
	}

	fclose(fp);

	trainmodel(svmTrainFile, modelFileName);

	test(svmTrainFile, modelFileName);










}
