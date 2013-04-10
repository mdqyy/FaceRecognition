

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
#include "svm_classifer_clean.h"

//decision for 1, decision for -1, next model for decision 1, next model for decision -1

	//BK	BU	GY	OR	RD	WH	YL
static int modelTree[6][7]={
	{1,		-1,	 1,	-1,	-1,	 1,	-1},	//model 0
	{1,		 0,	-1,	 0,	 0,	-1,  0},	//model 1
	{0,		 0, -1,	 0,	 0,	 1,	 0},	//model 2
	{0,		 1,  0,	-1,	 1,	 0,	-1},	//model 3
	{0,		-1,  0,  0,  1,  0,  0},	//model 4
	{0,		 0,  0,	 1,	 0,	 0,	-1}};	//model 5	



void initSVMTest()
{


}


void initSystem(SVM_GST * gst,svm_classifer_clean<int,double>*svm)
{
	gst->path = "C:/Users/Zhi/Desktop/FaceRecognition/image/svm/";

	int k;
	char modelFilePath[256];

	gst->sampleLable = (int *)malloc(MAX_SAMPLE_SIZE * sizeof(int));
	gst->classLable = (int *)malloc(MAX_SAMPLE_SIZE * sizeof(int));

	gst->feature = (double *)malloc(MAX_SAMPLE_SIZE * sizeof(double));

	gst->nClasses = 2;

	/*Load Module*/
	for (k=0;k<NUMBER_OF_MODULES;k++)
	{
		sprintf(modelFilePath,"%s%d.mod",gst->path,k);
		svm[k].svm_init_clean(modelFilePath);
	}

	
	initSVMTest();
}


void svmTestSamples(SVM_GST * gst,svm_classifer_clean<int,double> *svm)
{
	char * path;
	char * imageListFileName;
	char ss[256], tt[256];
	char * tmpStr;
	FILE * imageListFP;
	int ct, lable, i, len, n;
	int * sampleLable, * classLable;
	double * feature;
	IplImage *src		= 0;
	float score; 
	int classFlag, classResult;
	char *colorTag[]={"Black", "Blue", "Gray", "Orange", "Red", "White", "Yellow"};


	sampleLable = gst->sampleLable;
	classLable = gst->classLable;
	path = gst->path;
	imageListFileName = gst->imageListFileName;

	sprintf(ss, "%stestData/%s", path, imageListFileName);
	imageListFP = fopen(ss, "rt");

	gst->featureSize = 144;

	ct = 0;
	//while(!feof(imageListFP))
	while(ct < 300)
	{
		//fscanf(imageListFP, "%d", &lable);
		//sampleLable[ct] = lable;

		//fgetc(imageListFP);						//remove the blank space

		fgets(ss, 128, imageListFP);
		len = sprintf(tt, "%s%s%s", path, "testData/", ss);
		tt[len-1] = 0;							//the last char should be \0

		src = cvLoadImage(tt, CV_LOAD_IMAGE_COLOR);

		feature = gst->feature;

		extractImageFeatures_Type1((unsigned char *)src->imageData, 0, 0, src->height, src->width, src->widthStep, 
			feature, gst->featureSize);
		

#if 1
		classResult = -1;

		svmTest(feature, gst->featureSize, 0, &score,svm);
		if(score > 0)					//Black, White, or Gray
		{
			svmTest(feature, gst->featureSize, 1, &score,svm);
			if(score > 0)				//Black
				classResult = 0;
			else						
			{							//White or gray
				svmTest(feature, gst->featureSize, 2, &score,svm);
				if(score > 0) classResult = 5;
				else classResult = 2;
			}
		}
		else
		{								//Red, blue, orange, yellow
			svmTest(feature, gst->featureSize, 3, &score,svm);
			if(score > 0)				
			{							//red, blue
				svmTest(feature, gst->featureSize, 4, &score,svm);
				if(score > 0) classResult = 4;
				else classResult = 1;
			}
			else						
			{							//orange, yellow
				svmTest(feature, gst->featureSize, 5, &score,svm);
				if(score > 0) classResult = 3;
				else classResult = 6;
			}

		}


#else

		classFlag = -1;
		classResult = -1;
		n = 0;

		while((classFlag < 0) && (n < (gst->nClasses-1)))
		{
			svmTest(feature, gst->featureSize, n,&scores[n],svm);

			if(scores[n] >= 0) 	
			{
				classFlag = 1;
				classResult = n;	
			}
			else classFlag = -1;

			n++;
		}

		if(classResult == -1) classResult = gst->nClasses-1;
#endif

		printf("\nSample %6d %s\n ", ct, colorTag[classResult]);
	
		
		cvShowImage("Image", src);
		cvWaitKey(1000);

		ct++;
	}


	fclose(imageListFP);
	gst->nSamples = ct;
}


void svmTest(double * feature, int featureSize, int n,float * scores,svm_classifer_clean<int,double> *svm)
{
	int temp = 6;
	svm[n].svm_classifier_clean(&temp,feature,scores,featureSize,1);
	
}
