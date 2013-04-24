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
	char		assignName[260];
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
			printf("\n-----------------\nInput Name:\n");
			gets(assignName);
			printf("Input ID:\n");
			cin >> assignID;
		}

		if ( assignID != 0)
		{
			if ( gcv->faceDet->runFaceDetector(pFrame))
			{
				//face detected
				sprintf(path, "%s%d/%d.jpg", gf->cameraCaptureDir, assignID, frame);
				cvSaveImage(path, pFrame);
				printf(".");
			}
		}

		frame++;
		cvReleaseImage(&pFrame);
	}//end while

	cvDestroyWindow("Camera Input");
	cvDestroyWindow("Face");

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
						

/* process test input */
void processMatchInput(gFaceReco* gf, gFaceRecoCV* gcv)
{
	int			numImages;
	char		path[260];
	char		to_search[260];
	int			i, j;
	int			numTags;
	long		handle;                             //search handle
	struct		_finddata_t fileinfo;               // file info struct
	pathStruct*	list;

	numImages	= 0;
	numTags		= 0;
	list		= gf->imageList;


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
	fwrite(&(gf->bUseReferDist), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->bUseLBP), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->bUseGabor), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->bUseIntensity), sizeof(bool), 1, pFaceFeatBin);
	fwrite(&(gf->featLenTotal), sizeof(int), 1, pFaceFeatBin);

	//process list
	processTrainInput(gf, gcv);

	//main procedure
	numPercent  = (int)(gf->numImageInList / 100);
	numValidFaces = 0;

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
		extractReferDistFeatures(gf);
		dumpFeatures(gf, pFaceFeatBin);
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
		//cvSaveImage("c:/Users/Zhi/Desktop/dump.jpg",gcv->warpedImg);

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
		gf->features.id = gf->imageList[i].id;
		matchedID = matchFaceID(gf);

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
	fprintf(pResultOutput, "Overall match number: %d, Overall Correct Matches:%d, Accuracy: %.2f\n", overallTotal, overallCorrect, 100.0*overallCorrect/overallTotal);
	fprintf(pResultOutput, "------------------------------------------------------\n\n");
	for ( i = 0; i < gf->numTags; i++)
	{
		fprintf(pResultOutput, "ID: %d	%d of %d correct, Accuracy:%.2f\n-------------------------------------\n", i+1, correctMatch[i], totalInClass[i], 100.0*correctMatch[i]/totalInClass[i]);
	}



	
	//clean-ups
	fclose(pResultOutput);
	free(correctMatch);
	free(totalInClass);




}//end match