/** @file */

/*
EYE DETECTOR CLASS
Copyright (C) 2009 Rohan Anil (rohan.anil@gmail.com) -BITS Pilani Goa Campus
http://code.google.com/p/pam-face-authentication/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <windows.h>
#include <stdio.h>

#include "EyesDetector.h"
#include "pam_face_defines.h"

#include <highgui.h>
#include <cv.h>

double CenterofMass(IplImage* src,int flagXY);

//
string facePara = "../../source/XML/33_18795_60__3400_120_12_s_q.txt";
string leftEyeCorner = "../../source/XML/41268_60__340_lefteye_cnr_10.txt";
string leftEyeBall = "../../source/XML/41268_60__340_lefteye_cn_10.txt";
string rightEyeCorner = "../../source/XML/41268_60__340_righteye_cnr_10.txt";
string rightEyeBall = "../../source/XML/41268_60__340_lefteye_cn_10.txt";


#  define iabs(x)                     (((x) < 0)   ? -(x) : (x))


eyesDetector::eyesDetector()
{
	faceCNN = NULL;
	leftEyeCNN_cnr   = NULL;
	leftEyeCNN_ball  = NULL;
	rightEyeCNN_cnr  = NULL;
	rightEyeCNN_ball = NULL;
	face_data = NULL; 
	face_data_int = NULL;

	// load global face paramter
	faceCNN = new ConvNN(33, 33, 12, 120, 0.034);
	faceCNN->LoadCNNPara(facePara);

	// load left eye corner parameter
	leftEyeCNN_cnr = new ConvNN(33, 33, 4, 100, 0.034);
	leftEyeCNN_cnr->LoadCNNPara(leftEyeCorner);

	rightEyeCNN_cnr = new ConvNN(33, 33, 4, 100, 0.034);
	rightEyeCNN_cnr->LoadCNNPara(rightEyeCorner);


	face_data = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_32FC1 );
	face_data_int = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_8UC1 );

	faceCNNIO = new CNNIO;
	faceCNNIO->init(6, CNNFACECLIPWIDTH, CNNFACECLIPHEIGHT, faceCNN);

	leftCNNIO = new CNNIO;
	leftCNNIO->init(4, CNNFACECLIPWIDTH, CNNFACECLIPHEIGHT, leftEyeCNN_cnr);

	rightCNNIO = new CNNIO;
	rightCNNIO->init(4, CNNFACECLIPWIDTH, CNNFACECLIPHEIGHT, rightEyeCNN_cnr);

}


eyesDetector::~eyesDetector()
{
	if(faceCNN!=NULL)          delete faceCNN;
	if(leftEyeCNN_cnr!=NULL)   delete leftEyeCNN_cnr;
	if(leftEyeCNN_ball!=NULL)  delete leftEyeCNN_ball;
	if(rightEyeCNN_cnr!=NULL)  delete rightEyeCNN_cnr;
	if(rightEyeCNN_ball!=NULL) delete rightEyeCNN_ball;
	
	cvReleaseMat(&face_data);
	cvReleaseMat(&face_data_int);

	delete faceCNNIO;

	delete leftCNNIO;

	delete rightCNNIO;

}

int eyesDetector::checkEyeDetected()
{
	if (bothEyesDetected==1)
	{
		return 1;
	}
	else
	{
		return 0;
	}

}

void eyesDetector::copyInputData(IplImage *grayImg)
{
	if(grayImg->nChannels !=1)
	{
		printf("Input must be gray image\n");
		exit(0);
	}

	UCHAR *pSrc = (UCHAR *)grayImg->imageData;
	UCHAR *pDst = (UCHAR *)face_data_int->data.ptr;

	int srcLen = grayImg->widthStep;
	int dstLen = CNNFACECLIPWIDTH;

	for(int jj =0, m=0, n=0; jj< grayImg->height; jj++, m+=srcLen,n+=dstLen)
	{
		for(int ii =0; ii< grayImg->width; ii++)
		{
			UCHAR temp = *((UCHAR *)(pSrc+m+ii));
			*(UCHAR *)(pDst+n+ii) = temp;
		}
	}

	cvConvertScale(face_data_int, face_data, 1.0, 0);
}

void eyesDetector::runEyeDetector(IplImage * pFrame,IplImage *gray_face, faceDetector * faceDet, CvPoint pos[])

{


	//cvNamedWindow( "test");

#ifdef WRITEOUT
	static int frameNum = 0;
#endif

	// Feature point positions
	CvPoint avgLeftEye[2];
	CvPoint avgRightEye[2];
	CvPoint leftMth;
	CvPoint rightMth;
	CvPoint leftEyeCenter, rightEyeCenter;

	//Arrays to store eye feature point positions
	int xPos[4], yPos[4];


	//Flipped eye feature points positions
	int flipX[3][4];
	int flipY[3][4];

	CvPoint leftEye[2], rightEye[2];

	// Distance between eye centers
	int maxRad, maxRadX , maxRadY;
	int maxRadL, maxRadR;

	// upper left point of detected face
	int UL_x = faceDet->faceInformation.LT.x;
	int UL_y = faceDet->faceInformation.LT.y;

	// face width and height
	int clipHeight = faceDet->faceInformation.Height;
	int clipWidth  = faceDet->faceInformation.Width;

	double dClipHeight = (double)clipHeight;
	double dClipWidth  = (double)clipWidth;

#ifdef WRITEOUT
	char buffer[33];
	itoa(frameNum, buffer, 10);
#endif

	// Resize face image to standard size
	IplImage *smoothedImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),8, 1);
	cvCvtColor(pFrame, smoothedImg, CV_BGR2GRAY);
	cvSetImageROI(smoothedImg,cvRect(UL_x, UL_y, clipWidth, clipHeight));
	cvResize(smoothedImg, gray_face, CV_INTER_LINEAR);
	cvResetImageROI(smoothedImg);
	cvReleaseImage(&smoothedImg);

#ifdef DISPLAY
	IplImage *orgFace     = cvCreateImage(cvSize(clipWidth, clipHeight),8,pFrame->nChannels);
	cvSetImageROI(pFrame,cvRect(UL_x, UL_y, clipWidth, clipHeight));
	cvResize(pFrame, orgFace, CV_INTER_LINEAR);
	cvResetImageROI(pFrame);
#endif

	/* copy the data to face_data as the input of CNN ***/
	copyInputData(gray_face);		
	faceCNNIO->fpredict= icvCNNModelPredict_face(faceCNN->m_cnn,face_data,faceCNNIO->output);

	for(int jj=0; jj<4; jj++)
	{
		xPos[jj] = (int)(((cvmGet(faceCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(dClipWidth) + 0.5)      ;//+ UL_x;
		yPos[jj] = (int)(((cvmGet(faceCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(dClipHeight) + 0.5)  ;//+ UL_y;

		flipX[0][jj] = xPos[jj] ;
		flipY[0][jj] = yPos[jj] ;
	}

	// Positions of mouth feature points
	leftMth.x  =  (int)(((cvmGet(faceCNNIO->fpredict,  8, 0) + 1.0)/2.0)*(dClipWidth) + 0.5) + UL_x;
	leftMth.y  =  (int)(((cvmGet(faceCNNIO->fpredict,  9, 0) + 1.0)/2.0)*(dClipHeight) + 0.5) + UL_y;
	rightMth.x =  (int)(((cvmGet(faceCNNIO->fpredict, 10, 0) + 1.0)/2.0)*(dClipWidth) + 0.5) + UL_x;
	rightMth.y =  (int)(((cvmGet(faceCNNIO->fpredict, 11, 0) + 1.0)/2.0)*(dClipHeight) + 0.5) + UL_y;

	/*cvCircle(pFrame, cvPoint(flipX[0][0],flipY[0][0]),  4, cvScalar(255,0,0), -1);
	cvCircle(pFrame, cvPoint(flipX[0][1],flipY[0][1]),  4, cvScalar(255,0,0), -1);
	cvCircle(pFrame, cvPoint(flipX[0][2],flipY[0][2]), 4, cvScalar(255,0,0), -1);
	cvCircle(pFrame, cvPoint(flipX[0][3],flipY[0][3]), 4, cvScalar(255,0,0), -1);*/


#ifdef DISPLAY
	string fileName = "D:\\FaceData\\VideoClip\\video1\\output";
	fileName.append(buffer);
	fileName.append(".bmp");
	cvReleaseImage(&orgFace);
#endif


#ifdef DISPLAY
	orgFace     = cvCreateImage(cvSize(clipWidth, clipHeight),8,pFrame->nChannels);

	cvSetImageROI(pFrame,cvRect(UL_x, UL_y, clipWidth, clipHeight));
	cvResize(pFrame, orgFace, CV_INTER_LINEAR);
	cvResetImageROI(pFrame);
	cvFlip(orgFace  ,NULL, 1);
#endif

	cvFlip(gray_face,NULL, 1);
	copyInputData(gray_face);
	faceCNNIO->fpredict= icvCNNModelPredict_face(faceCNN->m_cnn,face_data, faceCNNIO->output);

	for(int jj=0; jj<4; jj++)
	{
		xPos[jj] = (int)(((cvmGet(faceCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(dClipWidth) + 0.5)      ;//+ UL_x;
		yPos[jj] = (int)(((cvmGet(faceCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(dClipHeight) + 0.5)  ;//+ UL_y;

		flipX[1][jj] = clipWidth-xPos[jj];
		flipY[1][jj] = yPos[jj] ;
	}

	// get the average value
	for(int jj=0; jj<4; jj++)
	{
		flipX[2][jj] = (flipX[0][jj]+flipX[1][3-jj])/2 + UL_x;
		flipY[2][jj] = (flipY[0][jj]+flipY[1][3-jj])/2 + UL_y;
	}



	/*cvCircle(pFrame, cvPoint(flipX[2][0],flipY[2][0]),  4, cvScalar(255,255,0), -1);
	cvCircle(pFrame, cvPoint(flipX[2][1],flipY[2][1]),  4, cvScalar(255,255,0), -1);
	cvCircle(pFrame, cvPoint(flipX[2][2],flipY[2][2]), 4, cvScalar(255,255,0), -1);
	cvCircle(pFrame, cvPoint(flipX[2][3],flipY[2][3]), 4, cvScalar(255,255,0), -1);

	cvShowImage("test", pFrame);*/

#ifdef DISPLAY
	cvFlip(orgFace, NULL, 1);

	cvCircle(orgFace, cvPoint((flipX[0][0]+flipX[3][1])/2, (flipY[0][0]+flipY[3][1])/2),     4, cvScalar(0,0,255), -1);
	cvCircle(orgFace, cvPoint((flipX[1][0]+flipX[2][1])/2, (flipY[1][0]+flipY[2][1])/2),     4, cvScalar(0,0,255), -1);
	cvCircle(orgFace, cvPoint((flipX[2][0]+flipX[1][1])/2, (flipY[2][0]+flipY[1][1])/2),     4, cvScalar(0,0,255), -1);
	cvCircle(orgFace, cvPoint((flipX[3][0]+flipX[0][1])/2, (flipY[3][0]+flipY[0][1])/2),     4, cvScalar(0,0,255), -1);

	itoa(frameNum, buffer, 10);
	fileName = "D:\\FaceData\\VideoClip\\video1\\output";
	fileName.append(buffer);
	fileName.append("_flip_new_1.bmp");
	cvSaveImage(fileName.c_str(), orgFace);
	cvReleaseImage(&orgFace);
#endif

	maxRad = maxRadX = maxRadY=0;

	if(iabs(flipX[2][0] - flipX[2][1])>maxRadX)	maxRadX = iabs(flipX[2][0] - flipX[2][1]);
	if(iabs(flipX[2][2] - flipX[2][3])>maxRadX)	maxRadX = iabs(flipX[2][2] - flipX[2][3]);
	if(iabs(flipY[2][0] - flipY[2][1])>maxRadY)	maxRadY = iabs(flipY[2][0] - flipY[2][1]);
	if(iabs(flipY[2][2] - flipY[2][3])>maxRadY)	maxRadY = iabs(flipY[2][2] - flipY[2][3]);

	//Get eye radius. 1.3 is just a factor
	maxRad = (int)((sqrt((double)maxRadX*maxRadX + maxRadY*maxRadY)+0.5)*1.3);


	/********create left eye image********************/
	leftEyeCenter.x = (flipX[2][0] + flipX[2][1])/2;
	leftEyeCenter.y = (flipY[2][0] + flipY[2][1])/2;

	runLeftEyesDetector(pFrame,leftEyeCenter, maxRad, leftCNNIO);
	
	for(int jj=0; jj<2; jj++)
	{
		xPos[jj] = (int)((((cvmGet(leftCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(CNNFACECLIPWIDTH))*maxRad*2/33.0+0.5)  + leftEyeCenter.x-maxRad;
			//+UL_x;//+ transUL_x;
		yPos[jj] = (int)((((cvmGet(leftCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(CNNFACECLIPHEIGHT))*maxRad*2/33.0+0.5) + leftEyeCenter.y-maxRad;
			//+ UL_y;

		leftEye[jj].x = xPos[jj];// - UL_x;
		leftEye[jj].y = yPos[jj];// - UL_y;
	}
	
#ifdef DISPLAY
	fileName.clear();
	fileName = "D:\\FaceData\\VideoClip\\video1\\output_lefteye";
	fileName.append(buffer);
	fileName.append("_final1");
	fileName.append(".bmp");

	if(frameNum %4 == 0)
		//cvSaveImage(fileName.c_str(), clonedImg);

		fileName.clear();
	fileName = "D:\\FaceData\\VideoClip\\video1\\output_lefteye";
	fileName.append(buffer);
	fileName.append("_final2");
	fileName.append(".bmp");
#endif

	/*******Create right eye image *****************/

	rightEyeCenter.x = (flipX[2][2] + flipX[2][3])/2;
	rightEyeCenter.y = (flipY[2][2] + flipY[2][3])/2;
	
	runRightEyesDetector(pFrame, rightEyeCenter, maxRad, rightCNNIO);

	for(int jj=0; jj<2; jj++)
	{
		xPos[jj] = (int)((((cvmGet(rightCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(CNNFACECLIPWIDTH))*maxRad*2/33.0+0.5)  + rightEyeCenter.x-maxRad;
			//+UL_x;//+ transUL_x;
		yPos[jj] = (int)((((cvmGet(rightCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(CNNFACECLIPHEIGHT))*maxRad*2/33.0+0.5) + rightEyeCenter.y-maxRad;
			//+ UL_y;

		rightEye[jj].x = xPos[jj];// - UL_x;
		rightEye[jj].y = yPos[jj];// - UL_y;			
	}

#ifdef DISPLAY
	fileName.clear();
	fileName = "D:\\FaceData\\VideoClip\\video1\\output_righteye_1";
	fileName.append(buffer);
	fileName.append("_final_2");
	fileName.append(".bmp");
#endif


	//printf("second round eye starts\n");
	/************ Second round eye location processing****************/

	double disX = (double)(leftEye[0].x-leftEye[1].x);
	double disY = (double)(leftEye[0].y-leftEye[1].y);
	maxRadL = (int) sqrt(disX * disX + disY*disY);

	disX = (double)(rightEye[0].x-rightEye[1].x);
	disY = (double)(rightEye[0].y-rightEye[1].y);
	maxRadR = (int) sqrt(disX * disX + disY*disY);

	maxRad = maxRadR >maxRadL ?maxRadR :maxRadL;
	maxRadL = maxRadR = maxRad;

	leftEyeCenter.x = (leftEye[0].x + leftEye[1].x)/2;
	leftEyeCenter.y = (leftEye[0].y + leftEye[1].y)/2;

	for(int jj=0; jj<2; jj++)
	{
		avgLeftEye[jj].x = 0;
		avgLeftEye[jj].y = 0;
	}


	int shiftx = 0;
	int shifty = 0;

	for( int jj=0; jj< 3; jj++)
	{
		leftEyeCenter.x += shiftx;
		leftEyeCenter.y += shifty;

		randomLeftEye(pFrame,leftEyeCenter, leftCNNIO, avgLeftEye, maxRadL, UL_x,UL_y);

		leftEyeCenter.x -= shiftx;
		leftEyeCenter.y -= shifty;

		shiftx =  rand()%9 - 4;
		shifty =  rand()%9 - 4;
	}

	// Apply a scale factor of 0.9
	randomLeftEye(pFrame,leftEyeCenter, leftCNNIO, avgLeftEye, (int)(maxRadL*0.9), UL_x,UL_y);
	
	// Apply a scale factor of 1.1
	randomLeftEye(pFrame,leftEyeCenter, leftCNNIO, avgLeftEye, (int)(maxRadL*1.1), UL_x,UL_y);


#ifdef DISPLAY
	fileName.clear();
	fileName = "D:\\FaceData\\VideoClip\\video1\\output_lefteye";
	fileName.append(buffer);
	fileName.append("_final1");
	fileName.append(".bmp");

	fileName.clear();
	fileName = "D:\\FaceData\\VideoClip\\video1\\output_lefteye";
	fileName.append(buffer);
	fileName.append("_final2");
	fileName.append(".bmp");
#endif

	for(int jj=0; jj<2; jj++)
	{
		avgRightEye[jj].x = 0;
		avgRightEye[jj].y = 0;
	}

	shiftx = 0;
	shifty = 0;

	for( int jj=0; jj< 3; jj++)
	{
		rightEyeCenter.x += shiftx;
		rightEyeCenter.y += shifty;

		randomLeftEye(pFrame,rightEyeCenter, rightCNNIO, avgRightEye, maxRadR, UL_x,UL_y);

		rightEyeCenter.x -= shiftx;
		rightEyeCenter.y -= shifty;

		shiftx =  rand()%9 - 4;
		shifty =  rand()%9 - 4;
	}

	// Apply a scale factor of 0.9
	randomRightEye(pFrame,rightEyeCenter, rightCNNIO, avgRightEye, (int)(maxRadR*0.9), UL_x,UL_y);

	// Apply a scale factor of 1.1
	randomRightEye(pFrame,rightEyeCenter, rightCNNIO, avgRightEye, (int)(maxRadR*1.1), UL_x,UL_y);



	for(int jj=0; jj<2; jj++)
	{
		avgLeftEye[jj].x = avgLeftEye[jj].x/5;
		avgLeftEye[jj].y = avgLeftEye[jj].y/5;

		avgRightEye[jj].x = avgRightEye[jj].x/5;
		avgRightEye[jj].y = avgRightEye[jj].y/5;
	}

	pos[0] = avgLeftEye[0];
	pos[1] = avgLeftEye[1];

	pos[2] = avgRightEye[0];
	pos[3] = avgRightEye[1];

	pos[4] = leftMth;
	pos[5] = rightMth;
}

void eyesDetector::runLeftEyesDetector(IplImage *orgFace, CvPoint eyeCenter, int maxRad, CNNIO *leftCNNIO)
{
	IplImage * leftEyeImg = cvCreateImage(cvSize(CNNFACECLIPWIDTH,CNNFACECLIPWIDTH),orgFace->depth, orgFace->nChannels);
	IplImage * leftEyeGrayImg = cvCreateImage(cvSize(CNNFACECLIPWIDTH,CNNFACECLIPWIDTH),orgFace->depth, 1);

	cvSetImageROI(orgFace,cvRect(eyeCenter.x-maxRad, eyeCenter.y-maxRad, maxRad*2, maxRad*2));
	cvResize(orgFace, leftEyeImg, CV_INTER_LINEAR);
	cvResetImageROI(orgFace);

	if(leftEyeImg->nChannels == 3)
	{
		cvCvtColor(leftEyeImg,leftEyeGrayImg, CV_BGR2GRAY);
	}
	else
		leftEyeGrayImg = cvCloneImage(leftEyeImg);
   
	copyInputData(leftEyeGrayImg);	

	//leftCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,leftCNNIO->probs,leftCNNIO->output);
	leftCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,leftCNNIO->output);

	cvReleaseImage(&leftEyeImg);
	cvReleaseImage(&leftEyeGrayImg);

}


void eyesDetector::runRightEyesDetector(IplImage *orgFace, CvPoint eyeCenter, int maxRad, CNNIO *rightCNNIO)
{

	IplImage * rightEyeImg = cvCreateImage(cvSize(CNNFACECLIPWIDTH,CNNFACECLIPWIDTH),orgFace->depth, orgFace->nChannels);
	IplImage * rightEyeGrayImg = cvCreateImage(cvSize(CNNFACECLIPWIDTH,CNNFACECLIPWIDTH),orgFace->depth, 1);

	cvSetImageROI(orgFace,cvRect(eyeCenter.x-maxRad , eyeCenter.y-maxRad, maxRad*2, maxRad*2));
	cvResize(orgFace, rightEyeImg, CV_INTER_LINEAR);
	cvResetImageROI(orgFace);

	if(rightEyeImg->nChannels == 3)
	{
		cvCvtColor(rightEyeImg,rightEyeGrayImg, CV_BGR2GRAY);
	}
	else
		rightEyeGrayImg = cvCloneImage(rightEyeImg);


	copyInputData(rightEyeGrayImg);	

	//rightCNNIO->fpredict= icvCNNModelPredict_face(rightEyeCNN_cnr->m_cnn,face_data,rightCNNIO->probs,rightCNNIO->output);
	rightCNNIO->fpredict= icvCNNModelPredict_face(rightEyeCNN_cnr->m_cnn,face_data, rightCNNIO->output);

	cvReleaseImage(&rightEyeImg);
	cvReleaseImage(&rightEyeGrayImg);

}

void eyesDetector::randomLeftEye(IplImage *orgFace, CvPoint leftEyeCenter, CNNIO *leftCNNIO, CvPoint avgLeftEye[2], \
								 int maxRadL, int UL_x, int UL_y)
{
	

	int xPos[2], yPos[2];

	runLeftEyesDetector(orgFace,leftEyeCenter, maxRadL, leftCNNIO);

	//leftCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,leftCNNIO->probs,leftCNNIO->output);
	leftCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,leftCNNIO->output);

	for(int jj=0; jj<2; jj++)
	{

		xPos[jj] = (int)((((cvmGet(leftCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(CNNFACECLIPWIDTH))*maxRadL*2/33.0+0.5)  + leftEyeCenter.x-maxRadL;
			//+UL_x;//+ transUL_x;
		yPos[jj] = (int)((((cvmGet(leftCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(CNNFACECLIPHEIGHT))*maxRadL*2/33.0+0.5) + leftEyeCenter.y-maxRadL;
			//+ UL_y;

		avgLeftEye[jj].x += xPos[jj];
		avgLeftEye[jj].y += yPos[jj];
	}
}


void eyesDetector::randomRightEye(IplImage *orgFace, CvPoint rightEyeCenter, CNNIO *rightCNNIO, CvPoint avgRightEye[2], \
								 int maxRadR, int UL_x, int UL_y)
{

	int xPos[2], yPos[2];

	runLeftEyesDetector(orgFace,rightEyeCenter, maxRadR, rightCNNIO);

	//rightCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,rightCNNIO->probs,rightCNNIO->output);
	rightCNNIO->fpredict= icvCNNModelPredict_face(leftEyeCNN_cnr->m_cnn,face_data,rightCNNIO->output);

	for(int jj=0; jj<2; jj++)
	{

		xPos[jj] = (int)((((cvmGet(rightCNNIO->fpredict, jj*2, 0) + 1.0)/2.0)*(CNNFACECLIPWIDTH))*maxRadR*2/33.0+0.5)  + rightEyeCenter.x-maxRadR;
			//+UL_x;//+ transUL_x;
		yPos[jj] = (int)((((cvmGet(rightCNNIO->fpredict, (jj*2+1), 0) + 1.0)/2.0)*(CNNFACECLIPHEIGHT))*maxRadR*2/33.0+0.5) + rightEyeCenter.y-maxRadR;
			//+ UL_y;

		avgRightEye[jj].x += xPos[jj];
		avgRightEye[jj].y += yPos[jj];
	}
}
