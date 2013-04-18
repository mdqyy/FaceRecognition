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
#include <cv.h>

using namespace cv;


void initGlobalCVStruct(gFaceRecoCV* gcv, gFaceReco* gf)
{
	gcv->faceDet = new faceDetector();
	gcv->eyeDet = new eyesDetector;
	gcv->gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);
	gcv->warpedImg = cvCreateImage( cvSize(gf->faceWidth, gf->faceHeight), IPL_DEPTH_8U, gf->faceChannel );

	gcv->faceTags = (IplImage**)malloc(sizeof(IplImage*) * gf->maxFaceTags);

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
	if (gcv->curImg != NULL)
		cvReleaseImage(&(gcv->curImg));
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
	distY = gf->actRightEyeY - gf->actLeftEyeY;

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
				//should not happen
				printf("Warning: Warping out of range!\n");
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





