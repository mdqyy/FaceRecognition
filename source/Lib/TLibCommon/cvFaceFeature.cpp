/**
* Program Name: faceRecognition
*
* Script File: cvFaceFeature.cpp
*
* Description:
*  
*
*  Features initialization and processing( OpenCV included)
*   
*  
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
**/

#include "cvFaceFeature.h"
#include "faceFeature.h"
#include <cv.h>
#include <highgui.h>
#include <istream>
#include <io.h>

using namespace cv;
using namespace std;


void initGlobalCVStruct(gFaceRecoCV* gcv, gFaceReco* gf)
{
	int i;
	gcv->eyeDet = new eyesDetector;
	gcv->faceDet = new faceDetector;
	
	gcv->gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);
	gcv->warpedImg = cvCreateImage( cvSize(gf->faceWidth, gf->faceHeight), IPL_DEPTH_8U, gf->faceChannel );

	gcv->faceTags = (IplImage**)malloc(sizeof(IplImage*) * gf->maxFaceTags);
	for ( i = 0; i < gf->maxFaceTags; i++)
	{
		gcv->faceTags[i] = NULL;
	}

}




void freeGlobalCVStruct(gFaceRecoCV* gcv, gFaceReco* gf)
{
	int i;

	if (gcv->faceDet != NULL)
		delete gcv->faceDet;
	if (gcv->eyeDet != NULL)
		delete gcv->eyeDet;
	if (gcv->gray_face_CNN != NULL)
		cvReleaseImage(&(gcv->gray_face_CNN));
	if (gcv->warpedImg != NULL)
		cvReleaseImage(&(gcv->warpedImg));

	if(gcv->faceTags != NULL)
	{
		for ( i = 0; i < gf->maxFaceTags; i++)
		{
			if ( gcv->faceTags[i] != NULL)
				cvReleaseImage(&(gcv->faceTags[i]));
		}
		free(gcv->faceTags);
	}

}


/*Run face detection and eye detection*/
bool runFaceAndEyesDetect(IplImage* pFrame, gFaceReco* gf, gFaceRecoCV* gcv)
{
	int		llx, lly, lrx, lry, rlx, rly, rrx, rry;	//eyes coordinates
	if ( gcv->faceDet->runFaceDetector(pFrame))
	{
		//face detected, and further to detect eyes
		IplImage* clonedImage = cvCloneImage(pFrame);
		gcv->eyeDet->runEyeDetector(clonedImage, gcv->gray_face_CNN, gcv->faceDet, gcv->pointPos);
		cvReleaseImage(&clonedImage);

		//calculate eyes' centers
		llx = gcv->pointPos[0].x;
		lly = gcv->pointPos[0].y;
		lrx = gcv->pointPos[1].x;
		lry = gcv->pointPos[1].y;
		rlx = gcv->pointPos[2].x;
		rly = gcv->pointPos[2].y;
		rrx = gcv->pointPos[3].x;
		rry = gcv->pointPos[3].y;

		gf->actLeftEyeX = (int)((llx + lrx) / 2);
		gf->actLeftEyeY = (int)((lly + lry) / 2);
		gf->actRightEyeX = (int)((rlx + rrx) / 2);
		gf->actRightEyeY = (int)((rly + rry) / 2);

		return 1;
	}
	else
	{
		return 0;	//face not detected
	}
}//end runFaceAndEyesDetect


/*Face Rotation given eyes coordinates*/
void faceAlign(IplImage* src, IplImage* dst, gFaceReco* gf)
{
	int		distX, distY;
	int		ptrSrcImg, ptrTarImg;
	int		wStepSrc, wStepTar;
	int		wSrcImg, hSrcImg;
	char	tmpChar;
	double	resAX, resAY, resBX, resBY;		//resolution along imaginary coordinates
	int		r, c;						// row & column
	int		dr, dc;						// distance to left eye center in row/column
	double	wr, wc;						// warped row & column
	int		i;
	UChar	tmpVal;

	IplImage* face0 = cvCreateImage(cvSize(gf->faceWidth, gf->faceHeight), IPL_DEPTH_8U, 1);
	IplImage* face1 = cvCreateImage(cvSize(gf->faceWidth1, gf->faceHeight1), IPL_DEPTH_8U, 1);
	IplImage* face2 = cvCreateImage(cvSize(gf->faceWidth2, gf->faceHeight2), IPL_DEPTH_8U, 1);

	wSrcImg		= src->width;
	hSrcImg		= src->height;
	wStepSrc	= src->widthStep;
	wStepTar	= dst->widthStep;

	distX = gf->actRightEyeX - gf->actLeftEyeX;
	distY = gf->actLeftEyeY - gf->actRightEyeY;		//y coordinates are inverse of natural images

	resAX = 1.0 * distX / (gf->rightEyeX - gf->leftEyeX);
	resBX = 1.0 * distY / (gf->rightEyeX - gf->leftEyeX);
	resAY = (-1) * resBX;
	resBY = resAX;


	for ( r=0; r<gf->faceHeight; r++ )
	{
		for ( c=0; c<gf->faceWidth; c++ )
		{
			dr = r - gf->leftEyeY;
			dc = c - gf->leftEyeX;
			wr = gf->actLeftEyeY + resAY * dc + resBY * dr;
			wc = gf->actLeftEyeX + resAX * dc + resBX * dr;
			// if within image, do interpolation.

			if (	( wr	>=0 )			&&
					( wr	<(hSrcImg-1) )	&&
					( wc	>=0 )			&&
					( wc	<(wSrcImg-1) )		)
			{
				ptrTarImg = (r * wStepTar + c * gf->faceChannel);

				//Blue Channel
				ptrSrcImg = ((int)(wr )* wStepSrc)		+ ((int)(wc) * gf->faceChannel);
				tmpChar = *( src->imageData + ptrSrcImg + wStepSrc + gf->faceChannel );
				*( dst->imageData + ptrTarImg)		= tmpChar;

				//Green Channel
				tmpChar = *( src->imageData + ptrSrcImg + wStepSrc + gf->faceChannel + 1);
				*( dst->imageData + ptrTarImg +1)	= tmpChar;

				//Red Channel
				tmpChar = *( src->imageData + ptrSrcImg + wStepSrc + gf->faceChannel + 2);
				*( dst->imageData + ptrTarImg +2)	= tmpChar;

			}
			else
			{
				//out of range
				//printf("Warning: Warping out of range!\n");
			}
		}//end columns
	}//end rows


	//convert to gray and downsample

	if ( dst->nChannels > 1)
	{
		cvCvtColor(dst, face0, CV_RGB2GRAY);
	}
	else
	{
		//already gray
		cvCopyImage(dst, face0);
	}

	if ( gf->bHistEqu)
	{
		//Histogram Equalization
		cvEqualizeHist(face0, face0);
	}

	//original face
	for (i = 0; i < gf->faceWidth * gf->faceHeight; i++)
	{
		
		memcpy(&tmpVal, &(face0->imageData[i]), sizeof(char));
		gf->face[i] = tmpVal;
	}

	//downsample by 2
	cvResize(face0, face1, CV_INTER_LINEAR);
	for (i = 0; i < gf->faceWidth1 * gf->faceHeight1; i++)
	{
		
		memcpy(&tmpVal, &(face1->imageData[i]), sizeof(char));
		gf->face1[i] = tmpVal;
	}

	//downsample by 4
	cvResize(face0, face2, CV_INTER_LINEAR);
	for (i = 0; i < gf->faceWidth2 * gf->faceHeight2; i++)
	{
		
		memcpy(&tmpVal, &(face2->imageData[i]), sizeof(char));
		gf->face2[i] = tmpVal;
	}

	//clean-ups
	cvReleaseImage(&face0);
	cvReleaseImage(&face1);
	cvReleaseImage(&face2);


}//end faceAlign





void cameraCapture(gFaceReco* gf, gFaceRecoCV* gcv)
{
	char		key;
	char		path[260];
	int			assignID;
	int			frame;
	//init capture
	CvCapture*	capture = cvCaptureFromCAM(1);
	IplImage*	pFrame;
	cvNamedWindow("Camera Input", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Face", CV_WINDOW_AUTOSIZE);

	//start
	printf("Start camera capture, press 'r' to record, press 'q' to exit...\n");


	frame = 0;
	assignID = 0;
	while((key = cvWaitKey(10)) != 'q')
	{
		pFrame = cvQueryFrame(capture);
		cvShowImage("Camera Input", pFrame);

		if ( key == 'r')
		{
			printf("Input ID:\n");
			cin >> assignID;
		}

		if ( assignID > 0)
		{
			if ( gcv->faceDet->runFaceDetector(pFrame))
			{
				//face detected
				sprintf(path, "%s%d/%d_%d.jpg", gf->cameraCaptureDir, assignID, assignID, frame);
				cvSaveImage(path, pFrame);
				printf(".");
			}
		}

		frame++;
	}//end while

	cvDestroyWindow("Camera Input");
	cvDestroyWindow("Face");
	cvReleaseCapture(&capture);

	system("pause");

}//end cameraCapture
			

/* process Training Images, prepare path, id...*/
void processTrainInput(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			numImages;
	int			startID, endID;
	char		path[260];
	char		to_search[260];
	int			i, j;
	long		handle;                             //search handle
	struct		_finddata_t fileinfo;               // file info struct
	pathStruct*	list;


	startID		= gf->trainStartID;
	endID		= gf->trainEndID;
	numImages	= 0;
	list		= gf->imageList;

	if ( access(gf->trainImageDir, 4) != -1)
	{
		//read permission
		for ( i = startID; i <= endID; i++)
		{
			sprintf(path, "%s%d/", gf->trainImageDir, i);
			if ( access(path, 4) != -1)
			{
				//directory exists and is readable
				sprintf(to_search, "%s%d/*.jpg", gf->trainImageDir, i);
				handle = _findfirst(to_search, &fileinfo);
				if ( handle != -1)
				{
					do
					{
						sprintf(path, "%s%d/%s", gf->trainImageDir, i, fileinfo.name);
						list[numImages].id = i;
						sprintf(list[numImages].path,"%s", path);
						numImages++;

					}while(_findnext(handle,&fileinfo) == 0);
					_findclose(handle);
				}//end searching in folder
			}
		}//end folder loop
	}//end reading folder list

	gf->numImageInList = numImages;	//save # image in list

}//end processTrainIput
			

/* load tagged faces*/
void loadTagFaces(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			i;
	char		path[260];
	int			numTags;


	numTags = 0;
	//load tagged images
	for ( i = 1; i <= gf->maxFaceTags; i++)
	{
		sprintf(path, "%s%d.jpg", gf->imageTagDir, i);
		gcv->faceTags[i-1] = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		{
			if ( gcv->faceTags[i-1] != NULL)
				numTags++;
		}
	}
	printf("Number of loaded face tags: %d\n", numTags);
	gf->numTags = numTags;

}//end loadTagFaces

/* process test input */
void processMatchInput(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			numImages;
	char		path[260];
	char		to_search[260];
	int			i, j;
	long		handle;                             //search handle
	struct		_finddata_t fileinfo;               // file info struct
	pathStruct*	list;

	numImages	= 0;
	list		= gf->imageList;


	//load tagged faces
	loadTagFaces(gf, gcv);

	//load testing images to list
	if ( access(gf->matchImageDir, 4) != -1)
	{
		//read permission
		for ( i = 1; i <= gf->maxFaceTags; i++)
		{
			sprintf(path, "%s%d/", gf->matchImageDir, i);
			if ( access(path, 4) != -1)
			{
				//directory exists and is readable
				sprintf(to_search, "%s%d/*.jpg", gf->matchImageDir, i);
				handle = _findfirst(to_search, &fileinfo);
				if ( handle != -1)
				{
					do
					{
						sprintf(path, "%s%d/%s", gf->matchImageDir, i, fileinfo.name);
						list[numImages].id = i;
						sprintf(list[numImages].path,"%s", path);
						numImages++;

					}while(_findnext(handle,&fileinfo) == 0);
					_findclose(handle);
				}//end searching in folder
			}
		}//end folder loop
	}//end reading folder list

	gf->numImageInList = numImages;	//save # image in list



}//end processMatchInput

	
/* main training procedure */
void train(gFaceReco* gf, gFaceRecoCV* gcv)
{
	
	int			i;
	int			numPercent;
	int			numValidFaces;
	IplImage*	pFrame = NULL;
	FILE*		pFaceFeatBin;
	errno_t		err;


	if ( !(gf->bOverWriteBin) )	
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
	if ( gf->bOverWriteBin)
	{
		fwrite(&(gf->bUseReferDist), sizeof(bool), 1, pFaceFeatBin);
		fwrite(&(gf->bUseLBP), sizeof(bool), 1, pFaceFeatBin);
		fwrite(&(gf->bUseGabor), sizeof(bool), 1, pFaceFeatBin);
		fwrite(&(gf->bUseIntensity), sizeof(bool), 1, pFaceFeatBin);
		fwrite(&(gf->featLenTotal), sizeof(int), 1, pFaceFeatBin);
	}

	//process list
	processTrainInput(gf, gcv);

	//main procedure
	numPercent  = (int)(gf->numImageInList / 100);
	numValidFaces = 0;

	printf("Start...\n");
	for ( i = 0; i < gf->numImageInList; i++)
	{
		//percentage progress
		if ( numPercent > 0)
		{
			if ( i % (5 * numPercent) == 0)
			{
				printf("%d%%...\n",i / numPercent);
			}
		}
		pFrame = cvLoadImage(gf->imageList[i].path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image in train list!\n");
			system("pause");
			exit(-1);
		}
		gf->features.id  = gf->imageList[i].id;

		//run face and eyes detection
		if (runFaceAndEyesDetect(pFrame, gf, gcv))
		{
			//face alignment
			faceAlign(pFrame, gcv->warpedImg, gf);
			//cvSaveImage("C:/Users/Zhi/Desktop/face.jpg", gcv->warpedImg);

			//feature extraction
			if ( gf->bUseLBP)
			{
				extractLBPFeatures(gf);
			}
			
			if ( gf->bUseGabor)
			{
				extractGaborFeatures(gf);
			}

			if ( gf->bUseIntensity)
			{
				extractIntensityFeatures(gf);
			}

			if ( gf->bUseReferDist)
			{
				//copy features to buffer
				copyOneFeatureToBuffer(gf, numValidFaces);
			}
			else
			{
				if ( gf->bVerification)
				{
					copyOneFeatureToBuffer(gf, numValidFaces);
				}
				//write features to binary file
				dumpFeatures(gf, pFaceFeatBin);
			}

			numValidFaces++;
		}//end face detected

		cvReleaseImage(&pFrame);
	}//end list

	gf->numValidFaces = numValidFaces;

	if ( gf->bUseReferDist)
	{
		extractReferDistFeatures(gf, pFaceFeatBin);
	}


	//close binary file
	fclose(pFaceFeatBin);

}//end train


void match(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			i;
	int			matchedID;
	IplImage*	pFrame = NULL;
	FILE*		pResultOutput;
	errno_t		err;
	UInt*		correctMatch;
	UInt*		totalInClass;
	int			overallCorrect, overallTotal;
	//debug only
	FILE*		pSVMDebug = fopen("../../image/svmDebug.txt","w");

	//open result output text file
	err = fopen_s(&pResultOutput, gf->resultTxtPath, "w");
	if (err != 0)
	{
		printf("Can't open result text file to write!\n");
		system("pause");
		exit(-1);
	}

	//load saved features from binary file
	loadFeatures(gf);

	//process list
	processMatchInput(gf, gcv);

	//statistic initilization
	correctMatch = (UInt*)malloc(sizeof(UInt) * gf->numTags);
	totalInClass = (UInt*)malloc(sizeof(UInt) * gf->numTags);
	for ( i = 0; i < gf->numTags; i++)
	{
		correctMatch[i] = 0;
		totalInClass[i] = 0;
	}
	overallCorrect = 0;
	overallTotal = 0;

	//cvWindow
	cvNamedWindow("Query Image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Matched Face", CV_WINDOW_AUTOSIZE);

	for ( i = 0; i < gf->numImageInList; i++)
	{
		pFrame = cvLoadImage(gf->imageList[i].path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image in train list!\n");
			system("pause");
			exit(-1);
		}
		//run face and eyes detection
		runFaceAndEyesDetect(pFrame, gf, gcv);

		//face alignment
		faceAlign(pFrame, gcv->warpedImg, gf);
		

		//feature extraction
		if ( gf->bUseLBP)
		{
			extractLBPFeatures(gf);
		}
		
		if ( gf->bUseGabor)
		{
			extractGaborFeatures(gf);
		}

		if ( gf->bUseIntensity)
		{
			extractIntensityFeatures(gf);
		}

		if ( gf->bUseReferDist)
		{
			extractReferDistFeaturesInMatch(gf);
		}
		gf->features.id = gf->imageList[i].id;

		
		if ( gf->bVerification)
		{
			matchedID = matchFaceIDVerification(gf, pSVMDebug);
		}
		else
		{
			matchedID = matchFaceID(gf);
		}

		//Result
		printf("Matched Face ID: %d\n----------------------------\n", matchedID);
		cvShowImage("Query Image", pFrame);
		cvShowImage("Matched Face", gcv->faceTags[matchedID-1]);
		cvWaitKey(100);

		//Benchmark only
		if ( matchedID == gf->imageList[i].id)
		{
			//correct
			correctMatch[gf->imageList[i].id - 1] += 1;
			overallCorrect++;
		}
		totalInClass[gf->imageList[i].id - 1] += 1;
		overallTotal++;


		cvReleaseImage(&pFrame);
	}//end list


	//write benchmark result
	fprintf(pResultOutput, "Benchmark of match result:\n");
	fprintf(pResultOutput, "Overall match number: %d, Overall Correct Matches:%d, Accuracy: %.2f\n", 
		overallTotal, overallCorrect, 100.0*overallCorrect/overallTotal);
	fprintf(pResultOutput, "------------------------------------------------------\n\n");
	for ( i = 0; i < gf->numTags; i++)
	{
		fprintf(pResultOutput, "ID: %d	%d of %d correct, Accuracy:%.2f\n-------------------------------------\n", 
			i+1, correctMatch[i], totalInClass[i], 100.0*correctMatch[i]/totalInClass[i]);
	}



	
	//clean-ups
	fclose(pResultOutput);
	free(correctMatch);
	free(totalInClass);


	//debug
	fclose(pSVMDebug);


}//end match



/* train verification module using SVM */
void trainVerification(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int**	svmPair;
	int*	list;
	int		numPairs;
	int		i, j, cntIntra, cntInter, tmpPtr;
	int		numIntraSamples, numInterSamples, numSamples;

	numSamples = gf->svmNumSamples;
	numIntraSamples = numSamples / ( 1 + gf->svmInterIntraRatio);
	numInterSamples = numSamples - numIntraSamples;

	//get features first
	train(gf, gcv);

	printf("Now start SVM Training...\n");
	//calculate total number of pairs
	numPairs = 0;
	for ( i = 1; i < gf->numValidFaces; i++)
	{
		numPairs += i;
	}
	//allocate temp pairs
	svmPair = (int**)malloc(sizeof(int*) * numPairs);
	for ( i = 0; i < numPairs; i++)
	{
		svmPair[i] = (int*)malloc(sizeof(int) * 2);
	}
	list = (int*)malloc(sizeof(int) * numPairs);

	//assign pairs
	cntIntra = 0;
	cntInter = 0;
	for ( i = 0; i < gf->numValidFaces; i++)
	{
		for ( j = i + 1; j < gf->numValidFaces; j++)
		{
			//intra class up-down, inter classes reverse order
			if ( gf->bufferFeatures[i].id == gf->bufferFeatures[j].id)
			{
				svmPair[cntIntra][0] = i;
				svmPair[cntIntra][1] = j;
				cntIntra++;
			}
			else
			{
				tmpPtr = numPairs - cntInter - 1;
				svmPair[tmpPtr][0] = i;
				svmPair[tmpPtr][1] = j;
				cntInter++;
			}
		}
	}

	if ( !gf->bUseAllSamples)	//limited samples
	{
		//generate SVM training data
		assert( (cntIntra >= numIntraSamples) && (cntInter >= numInterSamples) && (cntInter > cntIntra));

		//first pick up intra samples
		for ( i = 0; i < cntIntra; i++)
		{
			list[i] = i;
		}
		shuffle(list, cntIntra);
		for ( i = 0; i < numIntraSamples; i++)
		{
			extractAbsDist(gf, &gf->bufferFeatures[svmPair[list[i]][0]], &gf->bufferFeatures[svmPair[list[i]][1]], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[i], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			gf->svmSampleLabels[i] = 2;
		}

		//then inter samples
		for ( i = 0; i < cntInter; i++)
		{
			list[i] = i + cntIntra;
		}
		shuffle(list, cntInter);
		for ( i = 0; i < numInterSamples; i++)
		{
			extractAbsDist(gf, &gf->bufferFeatures[svmPair[list[i]][0]], &gf->bufferFeatures[svmPair[list[i]][1]], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[i+numIntraSamples], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			gf->svmSampleLabels[i+numIntraSamples] = 1;
		}
	}
	else
	{
		//use all the samples
		gf->svmTrainFeatures = (float**)malloc(sizeof(float*) * numPairs);
		for ( i = 0; i < numPairs; i++)
		{
			gf->svmTrainFeatures[i] = (float*)malloc(sizeof(float) * gf->featLenTotal);
		}
		gf->svmSampleLabels = (int*)malloc(sizeof(float*) * numPairs);
		gf->svmNumSamples = numPairs;
		numSamples = numPairs;

		for ( i = 0; i < cntIntra; i++)
		{
			extractAbsDist(gf, &gf->bufferFeatures[svmPair[i][0]], &gf->bufferFeatures[svmPair[i][1]], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[i], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			gf->svmSampleLabels[i] = 2;
		}
		for ( i = cntIntra; i < cntIntra + cntInter; i++)
		{
			extractAbsDist(gf, &gf->bufferFeatures[svmPair[i][0]], &gf->bufferFeatures[svmPair[i][1]], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[i], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			gf->svmSampleLabels[i] = 1;
		}

	}


	//call svm training
	svmTraining(gf->svmTrainFeatures, numSamples, gf->svmFeatureLen, gf->svmSampleLabels, gf->svmModelPath, gf->magicNumber);


	//clean-ups
	for ( i = 0; i < numPairs; i++)
	{
		free(svmPair[i]);
		svmPair[i] = NULL;
	}
	free(svmPair);
	svmPair = NULL;
	
	free(list);
	list = NULL;



}//end trainVerification



/* train white list for several people */
void trainWhiteList(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			i, j, ptr;
	int			sizeList, startID, endID;
	int			numPairs;
	int*		whiteList = NULL;
	char		path[260];
	bool		b1, b2;
	FILE*		fp;
	errno_t		err;

	
	//preprocessing for training information
	startID = gf->trainStartID;
	endID = gf->trainEndID;
	sprintf(path, "%swhiteList.txt", gf->svmModelDir);
	err = fopen_s(&fp, path, "w");
	if ( err != 0)
	{
		printf("Error opening whiteList.txt to write!\n");
		system("pause");
		exit(-1);
	}
	printf("---------White List Training---------\n");
	printf("Training start from ID:%d to ID:%d\n", startID, endID);
	printf("Please input the number of people in the white list!\n");
	cin >> sizeList;
	fprintf(fp, "%d\n", sizeList);
	if ( sizeList > (endID - startID + 1))
	{
		//error
		printf("Error: number larger than the size of training data!\n");
		system("pause");
		fclose(fp);
		exit(1);
	}
	else
	{
		whiteList = (int*)malloc(sizeof(int) * sizeList);
		for ( i = 0; i < sizeList; i++)
		{
			printf("Input the ID of %d:\n", i+1);
			cin >> whiteList[i];
			fprintf(fp, "%d\n", whiteList[i]);
			if ( (whiteList[i] < startID) || (whiteList[i] > endID))
			{
				printf("Error: input ID exceeds the range of training data!\n");
				system("pause");
				fclose(fp);
				exit(1);
			}
		}
	}
	fclose(fp);
	//end preprocessing

	//start training
	//train(gf, gcv);
	loadFeatures(gf);

	printf("Features loaded, now start SVM training...\n");

	//SVM
	//calculate # pairs
	numPairs = 0;
	for ( i = 0; i < gf->numLoadedFaces; i++)
	{
		for ( j = i + 1; j < gf->numLoadedFaces; j++)
		{
			if ( isInList(whiteList, sizeList, gf->loadedFeatures[i].id) || isInList(whiteList, sizeList, gf->loadedFeatures[j].id))
				numPairs++;
		}
	}

	gf->svmNumSamples = numPairs;
	gf->svmTrainFeatures = (float**)malloc(sizeof(float*) * numPairs);
	for ( i = 0; i < numPairs; i++)
	{
		gf->svmTrainFeatures[i] = (float*)malloc(sizeof(float) * gf->featLenTotal);
	}
	gf->svmSampleLabels = (int*)malloc(sizeof(float) * numPairs);

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
				//both in the list
				extractAbsDist(gf, &gf->loadedFeatures[i], &gf->loadedFeatures[j], gf->svmTmpFeature);
				memcpy(gf->svmTrainFeatures[ptr], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
				gf->svmSampleLabels[ptr] = 2;
				ptr++;
			}
			else if ( b1 ^ b2)
			{
				//one in one out
				extractAbsDist(gf, &gf->loadedFeatures[i], &gf->loadedFeatures[j], gf->svmTmpFeature);
				memcpy(gf->svmTrainFeatures[ptr], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
				gf->svmSampleLabels[ptr] = 1;
				ptr++;
			}
		}
	}
	assert(ptr == numPairs);
	sprintf(path, "%swhiteList.model", gf->svmModelDir);
	svmTraining(gf->svmTrainFeatures, numPairs, gf->svmFeatureLen, gf->svmSampleLabels, path, gf->magicNumber);

	//train in-list one to the rest models
	if ( sizeList > 1)
	{
		for ( i = 0; i < sizeList; i++)
		{
			trainOneToRestModels(gf, whiteList[i], whiteList, sizeList);
		}
	}



	//clean-ups
	if ( whiteList != NULL)
	{
		free(whiteList);
	}
	whiteList = NULL;




}//end trainWhiteList

/* white list matching */
void checkWhiteList(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			i, j, k;
	int			matchedID;
	bool		bIsInList;
	IplImage*	pFrame = NULL;
	FILE*		pResultOutput;
	errno_t		err;
	UInt*		correctMatch;
	UInt*		totalInClass;
	int			overallCorrect, overallTotal;

	//open result output text file
	err = fopen_s(&pResultOutput, gf->resultTxtPath, "w");
	if (err != 0)
	{
		printf("Can't open result text file to write!\n");
		system("pause");
		exit(-1);
	}

	//load saved features from binary file
	loadFeatures(gf);

	//process list
	processMatchInput(gf, gcv);

	//statistic initilization
	correctMatch = (UInt*)malloc(sizeof(UInt) * gf->numTags);
	totalInClass = (UInt*)malloc(sizeof(UInt) * gf->numTags);
	for ( i = 0; i < gf->numTags; i++)
	{
		correctMatch[i] = 0;
		totalInClass[i] = 0;
	}
	overallCorrect = 0;
	overallTotal = 0;

	matchedID = 0;
	for ( i = 0; i < gf->numImageInList; i++)
	{
		pFrame = cvLoadImage(gf->imageList[i].path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image in train list!\n");
			system("pause");
			exit(-1);
		}
		//run face and eyes detection
		runFaceAndEyesDetect(pFrame, gf, gcv);

		//face alignment
		faceAlign(pFrame, gcv->warpedImg, gf);
		

		//feature extraction
		if ( gf->bUseLBP)
		{
			extractLBPFeatures(gf);
		}
		
		if ( gf->bUseGabor)
		{
			extractGaborFeatures(gf);
		}

		if ( gf->bUseIntensity)
		{
			extractIntensityFeatures(gf);
		}

		if ( gf->bUseReferDist)
		{
			extractReferDistFeaturesInMatch(gf);
		}
		gf->features.id = gf->imageList[i].id;

		if ( gf->bWhiteList)
		{
			bIsInList = matchFaceWhiteList(gf);
			if ( bIsInList)
			{
				float maxProb, tmpProb;
				//bIsInList == 1 if in list
				printf("ID: %d IN LIST!   ", gf->features.id);
				if ( gf->sizeList > 1)
				{
					//find out which one it belongs to
					maxProb = -1;
					matchedID = 0;
					for ( j = 0; j < gf->sizeList; j++)
					{
						tmpProb = matchOneInList(gf, gf->whiteList[j]);
						if (tmpProb > maxProb)
						{
							maxProb = tmpProb;
							matchedID = gf->whiteList[j];
						}
					}
				}
				else
				{
					matchedID = gf->whiteList[0];
				}
				printf("Matched ID: %d\n", matchedID);
			}
			else
			{
				printf("ID: %d NOT IN LIST!\n", gf->features.id);
			}
		}
		


		//Benchmark only
		if ( bIsInList && isInList(gf->whiteList, gf->sizeList, gf->features.id))
		{
			if ( matchedID == gf->features.id)
			{
				//correct
				correctMatch[gf->imageList[i].id - 1] += 1;
				overallCorrect++;
			}
		}
		else if ( !bIsInList && !(isInList(gf->whiteList, gf->sizeList, gf->features.id)))
		{
			//correct
			correctMatch[gf->imageList[i].id - 1] += 1;
			overallCorrect++;
		}
		totalInClass[gf->imageList[i].id - 1] += 1;
		overallTotal++;


		cvReleaseImage(&pFrame);
	}//end list


	//write benchmark result
	fprintf(pResultOutput, "Benchmark of match result:\n");
	fprintf(pResultOutput, "Overall match number: %d, Overall Correct Matches:%d, Accuracy: %.2f\n", 
		overallTotal, overallCorrect, 100.0*overallCorrect/overallTotal);
	fprintf(pResultOutput, "------------------------------------------------------\n\n");
	for ( i = 0; i < gf->numTags; i++)
	{
		fprintf(pResultOutput, "ID: %d	%d of %d correct, Accuracy:%.2f\n-------------------------------------\n", 
			i+1, correctMatch[i], totalInClass[i], 100.0*correctMatch[i]/totalInClass[i]);
	}

	printf("Overall match number: %d, Overall Correct Matches:%d, Accuracy: %.2f\n", 
		overallTotal, overallCorrect, 100.0*overallCorrect/overallTotal);

	
	//clean-ups
	fclose(pResultOutput);
	free(correctMatch);
	free(totalInClass);


}//end checkWhiteList


/* Do matching using live camera */
void cameraMatch(gFaceReco* gf, gFaceRecoCV* gcv)
{
	char		key;
	int			matchedID;
	int			frame;
	int			pool[20];
	int			poolSize = 20;
	int			cutOff = 50;
	int			decision;
	int			sFrame;
	int			cnt;
	int			i;
	//init capture
	CvCapture*	capture = cvCaptureFromCAM(0);
	IplImage*	pFrame;
	cvNamedWindow("Camera Input", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Matched Face", CV_WINDOW_AUTOSIZE);


	//init
	//load saved features from binary file
	loadFeatures(gf);

	//process list
	loadTagFaces(gf, gcv);

	//start
	printf("Start Matching, press 'q' to exit...\n");


	frame = 0;
	matchedID = 0;
	decision = 0;
	sFrame = 0;
	cnt = 0;
	memset(pool, 0, sizeof(int) * poolSize);
	while((key = cvWaitKey(10)) != 'q')
	{
		pFrame = cvQueryFrame(capture);
		cvShowImage("Camera Input", pFrame);

		//make desicion
		if ( ((frame - sFrame) > cutOff) || (cnt >= poolSize))
		{
			decision = getVotePool(pool, 0, gf->numTags, cnt);

			if ( decision > 0)
				cvShowImage("Matched Face", gcv->faceTags[decision-1]);
			else
				printf("No matched Face!\n");

			sFrame = frame;
			cnt = 0;
			decision = 0;
			memset(pool, 0, sizeof(int) * poolSize);
		}



		//run face and eyes detection
		runFaceAndEyesDetect(pFrame, gf, gcv);

		//face alignment
		faceAlign(pFrame, gcv->warpedImg, gf);
		

		//feature extraction
		if ( gf->bUseLBP)
		{
			extractLBPFeatures(gf);
		}
		
		if ( gf->bUseGabor)
		{
			extractGaborFeatures(gf);
		}

		if ( gf->bUseIntensity)
		{
			extractIntensityFeatures(gf);
		}

		if ( gf->bUseReferDist)
		{
			extractReferDistFeaturesInMatch(gf);
		}
		
		matchedID = matchFaceID(gf);

		pool[cnt] = matchedID;
		cnt++;
		frame++;
		printf(".");
		
	}//end while

	cvDestroyWindow("Camera Input");
	cvDestroyWindow("Matched Face");
	cvReleaseCapture(&capture);

	system("pause");

}//end cameraMatch



/* LFW verification training procedure */
void trainLFWVerification(gFaceReco* gf, gFaceRecoCV* gcv)
{
	
	int			i;
	int			numPairs;
	int			numValidPairs;
	bool		validPair;
	IplImage*	pFrame = NULL;
	FILE*		pList;
	errno_t		err;
	char		name[100];
	char		path[260];
	int			index;


	err = fopen_s(&pList, gf->lfwPairsTrain, "r");

	if (err != 0)
	{
		printf("Can't open LFW pairsDevTrain.txt to read!\n");
		system("pause");
		exit(-1);
	}

	

	//main procedure
	numValidPairs = 0;
	printf("Start...\n");
	fscanf(pList, "%d\n", &numPairs);
	gf->svmTrainFeatures = (float**)malloc(sizeof(float*) * numPairs * 2);
	for ( i = 0; i < numPairs * 2; i++)
	{
		gf->svmTrainFeatures[i] = (float*)malloc(sizeof(float) * gf->featLenTotal);
	}
	gf->svmSampleLabels = (int*)malloc(sizeof(int) * numPairs * 2);
	gf->svmNumSamples = numPairs;

	for ( i = 0; i < numPairs * 2; i++)
	{
		validPair = TRUE;
		printf("%d/%d\n", i+1, numPairs * 2);
		//--------------------------------scan first image--------------------------------------//
		fscanf(pList,"%s\t%d\t", &name, &index);
		sprintf(path, "%s%s/%s_%04d.jpg", gf->lfwDir, name, name, index);
		pFrame = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image %s in train list!\n", path);
			system("pause");
			exit(-1);
		}

		//run face and eyes detection
		if (runFaceAndEyesDetect(pFrame, gf, gcv))
		{
			//face alignment
			faceAlign(pFrame, gcv->warpedImg, gf);
			//cvSaveImage("C:/Users/Zhi/Desktop/face.jpg", gcv->warpedImg);

			//feature extraction
			if ( gf->bUseLBP)
			{
				extractLBPFeatures(gf);
			}
			
			if ( gf->bUseGabor)
			{
				extractGaborFeatures(gf);
			}

			if ( gf->bUseIntensity)
			{
				extractIntensityFeatures(gf);
			}

			copyOneFeatureToBuffer(gf, 0);


		}//end face detected
		else
		{
			validPair = FALSE;
			printf("WARNING: no face detected in %s\n", path);
		}

		cvReleaseImage(&pFrame);

		//----------------------------load second image--------------------------------//
		if ( i < numPairs)
		{
			fscanf(pList, "%d\n", &index);
		}
		else
		{
			fscanf(pList, "%s\t%d\n", &name, &index);
		}

		sprintf(path, "%s%s/%s_%04d.jpg", gf->lfwDir, name, name, index);
		pFrame = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image %s in train list!\n", path);
			system("pause");
			exit(-1);
		}

		//run face and eyes detection
		if (runFaceAndEyesDetect(pFrame, gf, gcv))
		{
			//face alignment
			faceAlign(pFrame, gcv->warpedImg, gf);
			//cvSaveImage("C:/Users/Zhi/Desktop/face.jpg", gcv->warpedImg);

			//feature extraction
			if ( gf->bUseLBP)
			{
				extractLBPFeatures(gf);
			}
			
			if ( gf->bUseGabor)
			{
				extractGaborFeatures(gf);
			}

			if ( gf->bUseIntensity)
			{
				extractIntensityFeatures(gf);
			}

			copyOneFeatureToBuffer(gf, 1);


		}//end face detected
		else
		{
			validPair = FALSE;
			printf("WARNING: no face detected in %s\n", path);
		}

		cvReleaseImage(&pFrame);

		if ( validPair)
		{
			extractAbsDist(gf, &gf->bufferFeatures[0], &gf->bufferFeatures[1], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[numValidPairs], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			if ( i < numPairs)
				gf->svmSampleLabels[numValidPairs] = 2;
			else
				gf->svmSampleLabels[numValidPairs] = 1;

			numValidPairs++;
		}

	}//end list

	svmTraining(gf->svmTrainFeatures, numValidPairs, gf->svmFeatureLen, gf->svmSampleLabels, gf->svmModelPath, 1.0);
	//svmTest(gf->svmTrainFeatures, numValidPairs, gf->svmFeatureLen, gf->svmSampleLabels, gf->svmModelPath, gf);



	//close binary file
	fclose(pList);

}//end trainLFWVerification



/* test LFW verification */
void testLFWVerification(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			i;
	int			numPairs;
	int			numValidPairs;
	bool		validPair;
	IplImage*	pFrame = NULL;
	FILE*		pList;
	errno_t		err;
	char		name[100];
	char		path[260];
	int			index;


	err = fopen_s(&pList, gf->lfwPairsTest, "r");

	if (err != 0)
	{
		printf("Can't open LFW pairsDevTest.txt to read!\n");
		system("pause");
		exit(-1);
	}

	

	//main procedure
	numValidPairs = 0;
	printf("Start...\n");
	fscanf(pList, "%d\n", &numPairs);

	//memory allocation
	gf->svmTrainFeatures = (float**)malloc(sizeof(float*) * numPairs * 2);
	for ( i = 0; i < numPairs * 2; i++)
	{
		gf->svmTrainFeatures[i] = (float*)malloc(sizeof(float) * gf->featLenTotal);
	}
	gf->svmSampleLabels = (int*)malloc(sizeof(int) * numPairs * 2);
	gf->svmNumSamples = numPairs;
	gf->bufferFeatures = (featStruct*)malloc(sizeof(featStruct) * 2);
	for ( i = 0; i < 2; i++)
	{
		initOneFeature(&(gf->bufferFeatures[i]), gf);
	}

	for ( i = 0; i < numPairs * 2; i++)
	{
		validPair = TRUE;
		printf("%d/%d\n", i+1, numPairs * 2);
		//--------------------------------scan first image--------------------------------------//
		fscanf(pList,"%s\t%d\t", &name, &index);
		sprintf(path, "%s%s/%s_%04d.jpg", gf->lfwDir, name, name, index);
		pFrame = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image %s in train list!\n", path);
			system("pause");
			exit(-1);
		}

		//run face and eyes detection
		if (runFaceAndEyesDetect(pFrame, gf, gcv))
		{
			//face alignment
			faceAlign(pFrame, gcv->warpedImg, gf);
			//cvSaveImage("C:/Users/Zhi/Desktop/face.jpg", gcv->warpedImg);

			//feature extraction
			if ( gf->bUseLBP)
			{
				extractLBPFeatures(gf);
			}
			
			if ( gf->bUseGabor)
			{
				extractGaborFeatures(gf);
			}

			if ( gf->bUseIntensity)
			{
				extractIntensityFeatures(gf);
			}

			copyOneFeatureToBuffer(gf, 0);


		}//end face detected
		else
		{
			validPair = FALSE;
			printf("WARNING: no face detected in %s\n", path);
		}

		cvReleaseImage(&pFrame);

		//----------------------------load second image--------------------------------//
		if ( i < numPairs)
		{
			fscanf(pList, "%d\n", &index);
		}
		else
		{
			fscanf(pList, "%s\t%d\n", &name, &index);
		}

		sprintf(path, "%s%s/%s_%04d.jpg", gf->lfwDir, name, name, index);
		pFrame = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		if ( pFrame == NULL)
		{
			printf("Error load image %s in train list!\n", path);
			system("pause");
			exit(-1);
		}

		//run face and eyes detection
		if (runFaceAndEyesDetect(pFrame, gf, gcv))
		{
			//face alignment
			faceAlign(pFrame, gcv->warpedImg, gf);
			//cvSaveImage("C:/Users/Zhi/Desktop/face.jpg", gcv->warpedImg);

			//feature extraction
			if ( gf->bUseLBP)
			{
				extractLBPFeatures(gf);
			}
			
			if ( gf->bUseGabor)
			{
				extractGaborFeatures(gf);
			}

			if ( gf->bUseIntensity)
			{
				extractIntensityFeatures(gf);
			}

			copyOneFeatureToBuffer(gf, 1);


		}//end face detected
		else
		{
			validPair = FALSE;
			printf("WARNING: no face detected in %s\n", path);
		}

		cvReleaseImage(&pFrame);

		if ( validPair)
		{
			extractAbsDist(gf, &gf->bufferFeatures[0], &gf->bufferFeatures[1], gf->svmTmpFeature);
			memcpy(gf->svmTrainFeatures[numValidPairs], gf->svmTmpFeature, sizeof(float) * gf->featLenTotal);
			if ( i < numPairs)
				gf->svmSampleLabels[numValidPairs] = 2;
			else
				gf->svmSampleLabels[numValidPairs] = 1;

			numValidPairs++;
		}

	}//end list

	svmTest(gf->svmTrainFeatures, numValidPairs, gf->svmFeatureLen, gf->svmSampleLabels, gf->svmModelPath);



	//close binary file
	fclose(pList);

	//clean-ups
	freeOneFeature(&gf->bufferFeatures[0]);
	freeOneFeature(&gf->bufferFeatures[1]);
	free(gf->bufferFeatures);
	gf->bufferFeatures = NULL;


}//end testLFWVerification