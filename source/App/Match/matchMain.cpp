/**
* @mainpage LWIFace
*   <b>Codes to detect faces</b>
*
*   LWIFace reads the model from Cov_eye and Cov_face. Then it uses the
*   model to detect faces.
*
*   inputfile name: testinput.txt
*   input image folder: input/
*   output image folder: output/
*
*   @author unspecified
*/

/** \file FERAmain.cpp
    \brief Defines the entry point for the console application.

    Details.
*/

#include <stdio.h>
#include <io.h>
#include <highgui.h>
#include <math.h>
#include <cv.h>
#include <time.h>

#include "TLibCommon/EyesDetector.h"
#include "TLibCommon/Detector.h"
#include "TLibCommon/global.h"
#include <fstream>
#include "TLibCommon/ConvNN.h"
//#include "TLibCommon/CvGaborFace.h"

#include "TLibCommon/faceFeature.h"
#include "TLibCommon/affineWarping.h"
#include <cxcore.h>

//#define WRITE_FEATURE_DATUM_2_FILE
//#define	STR_INPUT_IMAGE_DIR		"../../image/train/input_align_2/"
#define MATCH_IMAGE_DIR			"../../image/match/"
#define MATCH_INPUT_TXT_FILE    "../../image/matchIn.txt"
#define MATCH_OUTPUT_TXT_FILE	"../../image/matchOut.txt"
#define BIN_FILE				"../../image/faces.bin"
#define IMAGE_TAG_DIR			"../../image/ImgTag/"
#define RESULT_TXT_DIR			"../../image/matchResult.txt"

#define DO_MATCH
#define	STR_INPUT_IMAGE_DIR		"../../image/match/Query3/"
//#define	STR_INPUT_IMAGE_DIR		"input_align/"

// demo only
#define MAX_NUM_FACE_ID_TAG			MAX_FACE_ID

using namespace std;

int    NSAMPLES = 1;
int    MAX_ITER = 1;
int    NTESTSAMPLES = 1;

#define BUFSIZE 20
#if DEBUG_MODE
#include <direct.h>
#endif

void testCamera();

#if 0
#pragma  comment(lib, "opencv_calib3d242d.lib")
#pragma  comment(lib, "opencv_contrib242d.lib")
#pragma  comment(lib, "opencv_core242d.lib")

#pragma  comment(lib, "opencv_features2d242d.lib")
#pragma  comment(lib, "opencv_flann242d.lib")

#pragma  comment(lib, "opencv_gpu242d.lib")
#pragma  comment(lib, "opencv_highgui242d.lib")

#pragma  comment(lib, "opencv_legacy242d.lib")
#pragma  comment(lib, "opencv_ml242d.lib")

#pragma  comment(lib, "opencv_objdetect242d.lib")

#pragma  comment(lib, "opencv_ts242d.lib")
#pragma  comment(lib, "opencv_video242d.lib")
#pragma  comment(lib, "opencv_imgproc242d.lib")

#endif


FACE3D_Type			gf;

unitFaceFeatClass	*bufferSingleFeatureID;

void showResults(IplImage * frame, FACE3D_Type * gf);


int testLiveFace()
{

	CvFont font;
	double hScale=0.5;
	double vScale=0.5;
	int    lineWidth=2;
	IplImage* pFrame = NULL; 
	IplImage* orgFace = NULL;

	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, hScale,vScale,0,lineWidth);
	eyesDetector * detectEye = new eyesDetector;

	faceDetector * faceDet =  new faceDetector();
	IplImage*  gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);

	// Feature points array
	CvPoint pointPos[6];
	CvPoint *leftEye    = &pointPos[0];
	CvPoint *rightEye   = &pointPos[2];
	CvPoint *leftMouth  = &pointPos[4];
	CvPoint *rightMouth = &pointPos[5];

	IplImage *frame = 0;
	CvCapture *capture  = 0;
	capture = cvCaptureFromCAM(-1);

	frame = cvQueryFrame( capture );

	cvNamedWindow("show");

	//Count image numbers
	static int frameNum = 0;

	while(1)
	{

		frame = cvQueryFrame( capture );
		if ( !frame )	break;


		pFrame = frame;

		if( faceDet->runFaceDetector(pFrame))
		{

			IplImage * clonedImg = cvCloneImage(pFrame);

			detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

			cvReleaseImage(&clonedImg);

			int UL_x = faceDet->faceInformation.LT.x;
			int UL_y = faceDet->faceInformation.LT.y;

			// face width and height
			CvPoint pt1, pt2;
			pt1.x =  faceDet->faceInformation.LT.x;
			pt1.y = faceDet->faceInformation.LT.y;
			pt2.x = pt1.x + faceDet->faceInformation.Width;
			pt2.y = pt1.y + faceDet->faceInformation.Height;

			cvRectangle(pFrame, pt1, pt2, cvScalar(0,0,255),2, 8, 0);

			cvCircle(pFrame, leftEye[0],  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, leftEye[1],  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rightEye[0], 2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rightEye[1], 2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, *leftMouth,  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, *rightMouth,  2, cvScalar(255,0,0), -1);

			printf("(%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)"
				" (%d,%d) (%d,%d) (%d,%d)\n",\
				pt1.x, pt1.y, pt2.x, pt2.y,
				pointPos[0].x,pointPos[0].y,
				pointPos[1].x,pointPos[1].y,
				pointPos[2].x,pointPos[2].y,
				pointPos[3].x,pointPos[3].y,
				pointPos[4].x,pointPos[4].y,
				pointPos[5].x,pointPos[5].y);

		}
		else
		{
			cvPutText(pFrame, "NO Face Found", cvPoint(500,30), &font, cvScalar(255,255,255));
		}

		cvShowImage("show", pFrame);

		frameNum++;

		if (cvWaitKey(1) == 'q') exit;
	}

	return NULL;
}


int testVideoData2()
{

	CvFont font;
	double hScale=0.5;
	double vScale=0.5;
	int    lineWidth=2;
	int tmpW = 800, tmpH = 600;
	int resultCnt[MAX_FACE_ID],correctCnt[MAX_FACE_ID]; //Result rate calculation

	for (int i = 0; i<MAX_FACE_ID; i++)
	{
		resultCnt[i] = 0;
		correctCnt[i] = 0;
	}
#if DEBUG_MODE
	char debugPath[500], tmpDebugPath[500];
#endif
	

	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, hScale,vScale,0,lineWidth);

//	string faceData = "yalebfaceFileList.txt";//"lwifaceFileList.txt";
//	CvGaborFace * gFace = new CvGaborFace;
//	gFace->loadTrnList(faceData);


	eyesDetector * detectEye = new eyesDetector;
#if 0
	CvCapture* capture = cvCaptureFromCAM( 0 );
#elif 0
	CvCapture* capture = cvCaptureFromAVI("D:\\out_york1.avi");
#endif
	//cvNamedWindow("show");

	printf("Write out face images with feature points marked\n=============\n");

	faceDetector * faceDet =  new faceDetector();
//	CvMat*  face_data = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_32FC1 );
//	CvMat*  face_data_int = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_8UC1 );
	IplImage*  gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);

	// Feature points array
	CvPoint pointPos[6];
	CvPoint *leftEye    = &pointPos[0];
	CvPoint *rightEye   = &pointPos[2];
	CvPoint *leftMouth  = &pointPos[4];
	CvPoint *rightMouth = &pointPos[5];

#if 0
	// Image list.
	FILE *fpinput = fopen(MATCH_INPUT_TXT_FILE,"r");
	if(fpinput==NULL){
		printf("open file testinput.txt failed!\n");fflush(stdout);
		exit(-1);
	}
#endif
	FILE *fpoutput = fopen(MATCH_OUTPUT_TXT_FILE,"w");
	if(fpoutput==NULL){
		printf("open file testoutput.txt failed!\n");fflush(stdout);
		exit(-1);
	}


	//char inputdir[100]="input_align_2/";
	char inputdir[100]= STR_INPUT_IMAGE_DIR;
	char tmpname[500];

	char tagImgdir[100]= IMAGE_TAG_DIR;
	char tagImgName[500];
	int tmpFaceID;
	int matchedFaceID;
	int	i;

	// demo display.

	IplImage *	imgFaceIDTag[MAX_NUM_FACE_ID_TAG];
	IplImage *	inputQueryImgResized;
	int			numTaggedFaces = 0;

	for (i=0; i<MAX_NUM_FACE_ID_TAG; i++)
	{
		sprintf( tagImgName, "%s%d.JPG",tagImgdir, i+1);
		imgFaceIDTag[i]		= cvLoadImage(tagImgName,3);

		if (imgFaceIDTag[i] == NULL)
		{
			printf("\nNum of tagged faces: %d\n", i);fflush(stdout);
			//exit(-1);
			numTaggedFaces = i;
			break;
		}
	
	}

	

	// Face feature archive.
	bufferSingleFeatureID	= (unitFaceFeatClass *)malloc( sizeof(unitFaceFeatClass) );

#ifdef WRITE_FEATURE_DATUM_2_FILE
	FILE *fpOutBinaryFile = fopen( BIN_FILE, "w+b");  // 2013.2.11 a+b to w+b
#endif

	/************************************************************************/
	/* look over face image samples.                                        */
	/************************************************************************/
	char to_search[500] = "./";
	// for each face tag
	for ( int ii = 0; ii < numTaggedFaces; ii++)
	{
	sprintf(to_search, "%s%d/*.jpg",MATCH_IMAGE_DIR,ii+1);
	long handle;                                                //search handle
	struct _finddata_t fileinfo;                          // file info struct
	handle=_findfirst(to_search,&fileinfo);         
	if(handle != -1) 
	{
		do
		{
	//while(!feof(fpinput)){

#if 0
		strcpy(tmpname,"");
		fgets(tmpname,500,fpinput);
		//printf(" %d ", strlen(tmpname)); 
		if(strlen(tmpname)==0)
			break;
		tmpname[strlen(tmpname)-1]='\0';
		puts(tmpname);
#endif
		//fscanf(fpinput, "%s	%d", tmpname, &tmpFaceID);

		// Store captured frame
		IplImage* pFrame = NULL; 
		IplImage* orgFace = NULL;

		//Count image numbers
		static int frameNum = 0;

		//Image ID
		//fscanf(fpinput, "%d", &tmpFaceID);

#if 0
		pFrame = cvQueryFrame( capture );
#else
		char tmppath[500];
		//sprintf(tmppath,"%s%s",inputdir,tmpname);
		sprintf(tmppath,"%s",fileinfo.name);
		puts(tmppath);
		sprintf(tmppath,"%s%d/%s",MATCH_IMAGE_DIR, ii+1,fileinfo.name);
		pFrame = cvLoadImage(tmppath,3);
		if(pFrame == NULL)
		{
			printf("read image file error!\n");fflush(stdout);
			exit(-1);
		}
#endif

		if( faceDet->runFaceDetector(pFrame))
		{	
			/* there was a face detected by OpenCV. */
			
			IplImage * clonedImg = cvCloneImage(pFrame);

			detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

			cvReleaseImage(&clonedImg);

			int UL_x = faceDet->faceInformation.LT.x;
			int UL_y = faceDet->faceInformation.LT.y;

			// face width and height
			CvPoint pt1, pt2;
			pt1.x =  faceDet->faceInformation.LT.x;
			pt1.y = faceDet->faceInformation.LT.y;
			pt2.x = pt1.x + faceDet->faceInformation.Width;
			pt2.y = pt1.y + faceDet->faceInformation.Height;

			// face warping.

			IplImage * tarImg = cvCreateImage( cvSize(warpedImgW, warpedImgH), IPL_DEPTH_8U, warpedImgChNum );
			
#if 0 // disabled 2013.2.8
			faceWarping(	pFrame,  tarImg,
							( leftEye[0].y + leftEye[1].y )/2, ( leftEye[0].x + leftEye[1].x )/2,
							( rightEye[0].y + rightEye[1].y )/2, ( rightEye[0].x + rightEye[1].x )/2,
							( leftMouth[0].y), ( leftMouth[0].x),
							( rightMouth[0].y), ( rightMouth[0].x),
							FIXED_LEFT_EYE_Y, FIXED_LEFT_EYE_X,
							FIXED_RIGHT_EYE_Y, FIXED_RIGHT_EYE_X,
							FIXED_LEFT_MOUTH_Y, FIXED_LEFT_MOUTH_X,
							FIXED_RIGHT_MOUTH_Y, FIXED_RIGHT_MOUTH_X);
#endif
#if 0 // disabled 2013.2.11
			//2013.2.8 simply apply rotation 
			cv::Point2f src_center(faceDet->faceInformation.Width/2.0F, faceDet->faceInformation.Height/2.0F);
			int lEyeCenterY = ( leftEye[0].y + leftEye[1].y )/2, lEyeCenterX = ( leftEye[0].x + leftEye[1].x )/2;
			int rEyeCenterY = ( rightEye[0].y + rightEye[1].y )/2, rEyeCenterX = ( rightEye[0].x + rightEye[1].x )/2;

			//
			double tanAngle = 1.0 * (rEyeCenterY - lEyeCenterY)/(rEyeCenterX - lEyeCenterX);  
			double angle = 90 * atan( tanAngle ); //small adjustment: only take half adjust degree 180--¡·90


			IplImage * clonedImg2 = cvCloneImage(pFrame);
			cvSetImageROI( clonedImg2, cvRect(pt1.x, pt1.y, faceDet->faceInformation.Width, faceDet->faceInformation.Height));
			IplImage* faceFrame = cvCreateImage(cvSize(faceDet->faceInformation.Width, faceDet->faceInformation.Height), IPL_DEPTH_8U, warpedImgChNum );
			cvCopy( clonedImg2, faceFrame);
			cvReleaseImage(&clonedImg2);

			//sprintf(tmppath,"output_align/facecrop_%s",tmpname);
			//cvSaveImage(tmppath,faceFrame);

			tarImg = faceRotate( angle, src_center.x, src_center.y, faceFrame, false, warpedImgW, warpedImgH);

			cvReleaseImage(&faceFrame);
#endif
			//2013.2.11 face rotation
			faceRotate(leftEye, rightEye, pFrame, tarImg, faceDet->faceInformation.Width, faceDet->faceInformation.Height);
			
			//downsampleing twice
			grayDownsample(tarImg, &gf, frameNum,TRUE);


			// feature extraction.
			gf.featureLength = 0;
#if USE_GBP
			extractGBPFaceFeatures( (unsigned char*)(tarImg->imageData), (tarImg->widthStep), &gf);
#endif
#if USE_LBP
			extractLBPFaceFeatures( (unsigned char*)(tarImg->imageData), (tarImg->widthStep), &gf, FALSE);
#endif
#if USE_GABOR
			extractGaborFeatures( &gf);
#endif

#if FLIP_MATCH
			gf.featureLength = 0;
#if USE_LBP
			extractLBPFaceFeatures( (unsigned char*)(tarImg->imageData), (tarImg->widthStep), &gf, TRUE);
#endif
#endif

#ifdef WRITE_FEATURE_DATUM_2_FILE

			// write feature to a binary file.
			bufferSingleFeatureID->id	= tmpFaceID;
			memcpy( bufferSingleFeatureID->feature, gf.faceFeatures, sizeof(float)*FACE_FEATURE_LEN );

			fwrite( bufferSingleFeatureID, 1, sizeof(unitFaceFeatClass), fpOutBinaryFile );
#endif

#ifdef DO_MATCH

			matchedFaceID	= matchFace(&gf );
#endif


			//Count the result for rate calculation
			resultCnt[ii] ++;
			if ( matchedFaceID == (ii+1))
			{
				correctCnt[ii]++;
			}


			//system("pause");
			//sprintf(tmppath,"output_align/warped_%s",tmpname);

			//cvSaveImage(tmppath,tarImg);
			//

			


			// plot graphic results.
			cvRectangle(pFrame, pt1, pt2, cvScalar(0,0,255),2, 8, 0);

			int lEyeCenterY = ( leftEye[0].y + leftEye[1].y )/2, lEyeCenterX = ( leftEye[0].x + leftEye[1].x )/2;
			int rEyeCenterY = ( rightEye[0].y + rightEye[1].y )/2, rEyeCenterX = ( rightEye[0].x + rightEye[1].x )/2;
			CvPoint lEyeball = cvPoint(lEyeCenterX, lEyeCenterY);
			CvPoint rEyeball = cvPoint(rEyeCenterX, rEyeCenterY);

			//modified: only show eyeballs position
			cvCircle(pFrame, lEyeball,  5, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rEyeball,  5, cvScalar(255,0,0), -1);

			//cvCircle(pFrame, leftEye[0],  10, cvScalar(255,0,0), -1);
			//cvCircle(pFrame, leftEye[1],  10, cvScalar(255,0,0), -1);
			//cvCircle(pFrame, rightEye[0], 10, cvScalar(255,0,0), -1);
			//cvCircle(pFrame, rightEye[1], 10, cvScalar(255,0,0), -1);
			//cvCircle(pFrame, *leftMouth,  10, cvScalar(255,0,0), -1);
			//cvCircle(pFrame, *rightMouth, 10, cvScalar(255,0,0), -1);

			//save labeled pFrame
			//sprintf(tmppath,"output_align/labeled_%s",tmpname);
			//cvSaveImage(tmppath,pFrame);

			//fprintf(fpoutput,"%s: (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)"
			//	" (%d,%d) (%d,%d) (%d,%d)\n",tmpname,\
			//	pt1.x, pt1.y, pt2.x, pt2.y,
			//	pointPos[0].x,pointPos[0].y,
			//	pointPos[1].x,pointPos[1].y,
			//	pointPos[2].x,pointPos[2].y,
			//	pointPos[3].x,pointPos[3].y,
			//	pointPos[4].x,pointPos[4].y,
			//	pointPos[5].x,pointPos[5].y);

//			cvReleaseImage(&normalizedface);
//			cvReleaseImage(&grayFace);
#if DEBUG_MODE
			if( matchedFaceID != (ii+1))
			{
				IplImage * tmpDistImg = NULL;
				IplImage *tmpTarImg =cvCreateImage( cvSize(warpedImgW, warpedImgH), IPL_DEPTH_8U, warpedImgChNum );
				//sprintf(debugPath, "C://Users//Zhang//Desktop//DEBUG_MODE//", fileinfo.name);
				//if (_mkdir(debugPath) == 0)
				//{
					sprintf(tmpDebugPath, "C:/Users/Zhang/Desktop/DEBUG_MODE/00%s",fileinfo.name);
					cvSaveImage(tmpDebugPath, tarImg ); //original face
					for (int jj = 0; jj<NUM_NEAREST_NBOR; jj++)
					{
						tmpDistImg = cvLoadImage(gf.bestDistImageName[jj], 3);

						faceDet->runFaceDetector(tmpDistImg);
						IplImage * clonedImg = cvCloneImage(tmpDistImg);

						detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

						cvReleaseImage(&clonedImg);

						UL_x = faceDet->faceInformation.LT.x;
						UL_y = faceDet->faceInformation.LT.y;

						// face width and height
						pt1.x =  faceDet->faceInformation.LT.x;
						pt1.y = faceDet->faceInformation.LT.y;
						pt2.x = pt1.x + faceDet->faceInformation.Width;
						pt2.y = pt1.y + faceDet->faceInformation.Height;

						// face warping.

						

						lEyeCenterY = ( leftEye[0].y + leftEye[1].y )/2, lEyeCenterX = ( leftEye[0].x + leftEye[1].x )/2;
						rEyeCenterY = ( rightEye[0].y + rightEye[1].y )/2, rEyeCenterX = ( rightEye[0].x + rightEye[1].x )/2;
						lEyeball = cvPoint(lEyeCenterX, lEyeCenterY);
						rEyeball = cvPoint(rEyeCenterX, rEyeCenterY);

						//modified: only show eyeballs position
						//cvCircle(tmpDistImg, lEyeball,  5, cvScalar(255,0,0), -1);
						//cvCircle(tmpDistImg, rEyeball,  5, cvScalar(255,0,0), -1);
						//cvRectangle(tmpDistImg, pt1, pt2, cvScalar(0,0,255),2, 8, 0);
						faceRotate(leftEye, rightEye, tmpDistImg, tmpTarImg, faceDet->faceInformation.Width, faceDet->faceInformation.Height);
						cvCircle(tmpTarImg, cvPoint(FIXED_LEFT_EYE_X,FIXED_LEFT_EYE_Y),  4, cvScalar(255,0,0), -1);
						cvCircle(tmpTarImg, cvPoint(FIXED_RIGHT_EYE_X,FIXED_LEFT_EYE_Y),  4, cvScalar(255,0,0), -1);


						sprintf(tmpDebugPath, "C:/Users/Zhang/Desktop/DEBUG_MODE/00%s_%.2f.jpg",fileinfo.name,gf.featDistance[jj]);
						cvSaveImage(tmpDebugPath, tmpTarImg);
						
					}
				//}
				cvReleaseImage( &tmpDistImg);
				cvReleaseImage( &tmpTarImg);
			}
#endif


			cvReleaseImage(&tarImg);

#ifdef DO_MATCH
			// debug:
			printf("\nMatched ID: %d \n------------------\n", matchedFaceID);
			//printf("\nAdjusted angle: %.4f \n-------------------\n", angle);
			fprintf(fpoutput, "\n\n%s:	:		%d\n\n", fileinfo.name, matchedFaceID);
			

			if (pFrame->width > 800)
			{
				tmpW = 800;
				tmpH = (1.0 * pFrame->height / pFrame->width)* tmpW;
				inputQueryImgResized	= cvCreateImage(cvSize(tmpW, tmpH), IPL_DEPTH_8U, 3);
				cvResize(pFrame, inputQueryImgResized, 1);
				cvNamedWindow("Input Image");
				cvShowImage("Input Image", inputQueryImgResized);
				cvReleaseImage(&inputQueryImgResized);
			}
			else
			{
				cvNamedWindow("Input Image");
				cvShowImage("Input Image", pFrame);
			}

			cvNamedWindow("Matched Face");
			cvShowImage("Matched Face", imgFaceIDTag[matchedFaceID-1]);
			cvWaitKey(100);
#endif

		}
		else
		{
			cvPutText(pFrame, "NO Face Found", cvPoint(500,30), &font, cvScalar(255,255,255));
		}

		/************************************************************************/
		/* Display                                                              */
		/************************************************************************/
#if 0
		cvResize(pFrame, inputQueryImgResized, 1);
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", inputQueryImgResized);

		cvNamedWindow("Matched Face");
		cvShowImage("Matched Face", imgFaceIDTag[matchedFaceID-1]);
		cvWaitKey(100);

#endif
		// save original image.
		//sprintf(tmppath,"output_align/%s",tmpname);
		//cvSaveImage(tmppath,pFrame);
		//system("pause");
	
		cvReleaseImage(&pFrame);

		frameNum++;

		if (cvWaitKey(1) == 'q')
			exit;
	}while (!_findnext(handle,&fileinfo));

	_findclose(handle); 
	}
	} //end folder selection
	//fclose(fpinput);
	fclose(fpoutput);

	//Result calculation and output
	float rate[MAX_FACE_ID];
	float overallRate = 0;
	int	  totalCnt = 0, totalCorrect = 0;
	for ( i =0; i< numTaggedFaces; i++)
	{
		if (resultCnt[i] == 0)
		{
			rate[i] = -1;
		}
		else
		{
			rate[i] = (float) correctCnt[i] / resultCnt[i];
			totalCnt += resultCnt[i];
			totalCorrect += correctCnt[i];
		}
	}
	overallRate = (float)totalCorrect / totalCnt;


	FILE* fResult = fopen(RESULT_TXT_DIR, "w");
	if (fResult == NULL)
	{
		printf("Cannot open result txt file!\n");
	}
	fprintf(fResult, "Overall rate: %.2f, %d correct matching in %d test images\n----------------------------------\n", overallRate * 100, totalCorrect, totalCnt);
	for ( i =0; i< numTaggedFaces; i++)
	{
		if (resultCnt[i] !=0)
		{
			fprintf(fResult, "ID%d: %.2f, %d correct matching in %d test images\n----------------------------------\n", i+1, rate[i]*100, correctCnt[i], resultCnt[i]);
	
		}
		else
		{
			fprintf(fResult, "ID%d: no face found\n----------------------------------\n",i+1);
		}
	}
	fclose(fResult);
	free(faceDet);
	

	
#ifdef WRITE_FEATURE_DATUM_2_FILE
	fclose(fpOutBinaryFile);
#endif

	return NULL;
}



int getSampleFaceFeatures()
{

	CvFont font;
	double hScale=0.5;
	double vScale=0.5;
	int    lineWidth=2;

	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, hScale,vScale,0,lineWidth);

	eyesDetector * detectEye = new eyesDetector;

	cvNamedWindow("show");

	faceDetector * faceDet =  new faceDetector();
	IplImage*  gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);


	// Feature points array
	CvPoint pointPos[6];
	CvPoint *leftEye    = &pointPos[0];
	CvPoint *rightEye   = &pointPos[2];
	CvPoint *leftMouth  = &pointPos[4];
	CvPoint *rightMouth = &pointPos[5];

	FILE *fpinput = fopen("testinput_align.txt","r");
	if(fpinput==NULL){
		printf("open file testinput.txt failed!\n");fflush(stdout);
		exit(-1);
	}
	FILE *fpoutput = fopen("testoutput_align.txt","w");
	if(fpoutput==NULL){
		printf("open file testoutput.txt failed!\n");fflush(stdout);
		exit(-1);
	}

	char inputdir[100]="input_align/";
	char tmpname[500];
	while(!feof(fpinput)){
		strcpy(tmpname,"");
		fgets(tmpname,500,fpinput);
		//printf(" %d ", strlen(tmpname)); 
		if(strlen(tmpname)==0)
			break;
		tmpname[strlen(tmpname)-1]='\0';
		puts(tmpname);

		IplImage* pFrame = NULL; // Store captured frame
		IplImage* orgFace = NULL;

		//Count image numbers
		static int frameNum = 0;

#if 0
		pFrame = cvQueryFrame( capture );
#else
		char tmppath[500];
		sprintf(tmppath,"%s%s",inputdir,tmpname);
		puts(tmppath);
		pFrame = cvLoadImage(tmppath,3);
		if(pFrame == NULL)
		{
			printf("read image file error!\n");fflush(stdout);
			exit(-1);
		}
#endif

		if( faceDet->runFaceDetector(pFrame))
		{

			IplImage * clonedImg = cvCloneImage(pFrame);

			detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

			cvReleaseImage(&clonedImg);

			int UL_x = faceDet->faceInformation.LT.x;
			int UL_y = faceDet->faceInformation.LT.y;

			// face width and height
			CvPoint pt1, pt2;
			pt1.x =  faceDet->faceInformation.LT.x;
			pt1.y = faceDet->faceInformation.LT.y;
			pt2.x = pt1.x + faceDet->faceInformation.Width;
			pt2.y = pt1.y + faceDet->faceInformation.Height;

			cvRectangle(pFrame, pt1, pt2, cvScalar(0,0,255),2, 8, 0);

			cvCircle(pFrame, leftEye[0],  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, leftEye[1],  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rightEye[0], 2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rightEye[1], 2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, *leftMouth,  2, cvScalar(255,0,0), -1);
			cvCircle(pFrame, *rightMouth,  2, cvScalar(255,0,0), -1);
			fprintf(fpoutput,"%s: (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)"
				" (%d,%d) (%d,%d) (%d,%d)\n",tmpname,\
				pt1.x, pt1.y, pt2.x, pt2.y,
				pointPos[0].x,pointPos[0].y,
				pointPos[1].x,pointPos[1].y,
				pointPos[2].x,pointPos[2].y,
				pointPos[3].x,pointPos[3].y,
				pointPos[4].x,pointPos[4].y,
				pointPos[5].x,pointPos[5].y);

//			cvReleaseImage(&normalizedface);
//			cvReleaseImage(&grayFace);
		}
		else
		{
			cvPutText(pFrame, "NO Face Found", cvPoint(500,30), &font, cvScalar(255,255,255));
		}

		cvShowImage("show", pFrame);

		sprintf(tmppath,"output_align/%s",tmpname);
		cvSaveImage(tmppath,pFrame);
	

		frameNum++;

		if (cvWaitKey(1) == 'q')
			exit(0);
	}
	fclose(fpinput);
	fclose(fpoutput);
	return NULL;
}


void testCamera2()
{
	int c;

          // allocate memory for an image

          IplImage *img;

          // capture from video device #1

          CvCapture* capture = cvCaptureFromCAM(-1);

          // create a window to display the images

          cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);

          // position the window

          cvMoveWindow("mainWin", 5, 5);

          while(1)

          {

                    // retrieve the captured frame

                    img=cvQueryFrame(capture);

                    // show the image in the window

                    cvShowImage("mainWin", img );

                    // wait 10 ms for a key to be pressed

                    c=cvWaitKey(10);

                    // escape key terminates program

                    if(c == 'q')

                    break;

          }


}
int main(int argc, char** argv)
{
	//-------------------
	// initializations.
	//-------------------

	initFaceWarping();	
	initFaceFeature( &gf, 80, 80);
#ifdef DO_MATCH
	loadFaceData( &gf );
#endif

	//-------------------
	// data access.
	//-------------------

	testVideoData2();	// find the face coordinates and eye, mouse position
	//testCamera2();
	//videoAnalysis();	// extract feature given the face coordinates

	//-------------------
	// closing.
	//-------------------
	//closeFaceWarping();
	system("pause");
	
}

void showResults(IplImage * frame, FACE3D_Type * gf)
{
	unsigned char * ptr, *mask;
	int r, c, widthStep;
	int FRAME_WIDTH, FRAME_HEIGHT;

	FRAME_HEIGHT = gf->FRAME_HEIGHT;
	FRAME_WIDTH = gf->FRAME_WIDTH;

	widthStep = frame->widthStep;
	mask = gf->mask;

	for(r=0; r<FRAME_HEIGHT; r++)
	for(c=0; c<FRAME_WIDTH; c++)
	{
		ptr = (unsigned char *)(frame->imageData + r * widthStep + c * 3);

		if(*(mask + r * FRAME_WIDTH + c) == 1)
		{
			ptr[0] = 255; ptr[1] = 255;
		}
	
	}


}


//-------------------------------------------------------------------

void testCamera()
{
      // allocate memory for an image
      // capture from video device #1
      CvCapture* capture = cvCaptureFromCAM(-1);
      // create a window to display the images
      cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
      // position the window
      //cvMoveWindow("mainWin", 5, 5);
      while(1)
      {

                // retrieve the captured frame

                //cameraImg=cvQueryFrame(capture);


				CvFont font;
	double hScale=0.5;
	double vScale=0.5;
	int    lineWidth=2;
	
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, hScale,vScale,0,lineWidth);

	eyesDetector * detectEye = new eyesDetector;

	printf("Write out face images with feature points marked\n=============\n");

	faceDetector * faceDet =  new faceDetector();
//	CvMat*  face_data = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_32FC1 );
//	CvMat*  face_data_int = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_8UC1 );
	IplImage*  gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);

	// Feature points array
	CvPoint pointPos[6];
	CvPoint *leftEye    = &pointPos[0];
	CvPoint *rightEye   = &pointPos[2];
	CvPoint *leftMouth  = &pointPos[4];
	CvPoint *rightMouth = &pointPos[5];

	char inputdir[100]= STR_INPUT_IMAGE_DIR;
	char tmpname[500];

	char tagImgdir[100]= IMAGE_TAG_DIR;
	char tagImgName[500];
	int tmpFaceID;
	int matchedFaceID;
	int	i;

	// demo display.

	IplImage *	imgFaceIDTag[MAX_NUM_FACE_ID_TAG];
	IplImage *	inputQueryImgResized;
	int			numTaggedFaces = 0;

	for (i=0; i<MAX_NUM_FACE_ID_TAG; i++)
	{
		sprintf( tagImgName, "%s%d.JPG",tagImgdir, i+1);
		imgFaceIDTag[i]		= cvLoadImage(tagImgName,3);

		if (imgFaceIDTag[i] == NULL)
		{
			printf("\nNum of tagged faces: %d\n", i);fflush(stdout);
			//exit(-1);
			numTaggedFaces = i;
			break;
		}
	
	}

	

	// Face feature archive.
	bufferSingleFeatureID	= (unitFaceFeatClass *)malloc( sizeof(unitFaceFeatClass) );

	/************************************************************************/
	/* look over face image samples.                                        */
	/************************************************************************/
		//fscanf(fpinput, "%s	%d", tmpname, &tmpFaceID);

		// Store captured frame
		IplImage* pFrame = NULL; 
		IplImage* orgFace = NULL;

		//Count image numbers
		static int frameNum = 0;

		//Image ID
		//fscanf(fpinput, "%d", &tmpFaceID);

#if 1
		pFrame = cvQueryFrame( capture );
#else
		char tmppath[500];
		//sprintf(tmppath,"%s%s",inputdir,tmpname);
		sprintf(tmppath,"%s",fileinfo.name);
		puts(tmppath);
		sprintf(tmppath,"%s%d/%s",MATCH_IMAGE_DIR, ii+1,fileinfo.name);
		pFrame = cvLoadImage(tmppath,3);
		if(pFrame == NULL)
		{
			printf("read image file error!\n");fflush(stdout);
			exit(-1);
		}
#endif

		if( faceDet->runFaceDetector(pFrame))
		{	
			/* there was a face detected by OpenCV. */
			
			IplImage * clonedImg = cvCloneImage(pFrame);

			detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

			cvReleaseImage(&clonedImg);

			int UL_x = faceDet->faceInformation.LT.x;
			int UL_y = faceDet->faceInformation.LT.y;

			// face width and height
			CvPoint pt1, pt2;
			pt1.x =  faceDet->faceInformation.LT.x;
			pt1.y = faceDet->faceInformation.LT.y;
			pt2.x = pt1.x + faceDet->faceInformation.Width;
			pt2.y = pt1.y + faceDet->faceInformation.Height;

			// face warping.

			IplImage * tarImg = cvCreateImage( cvSize(warpedImgW, warpedImgH), IPL_DEPTH_8U, warpedImgChNum );
			

			//2013.2.11 face rotation
			faceRotate(leftEye, rightEye, pFrame, tarImg, faceDet->faceInformation.Width, faceDet->faceInformation.Height);
			
			//downsampleing twice
			grayDownsample(tarImg, &gf, frameNum, TRUE);


			// feature extraction.
			gf.featureLength = 0;
#if USE_GBP
			extractGBPFaceFeatures( (unsigned char*)(tarImg->imageData), (tarImg->widthStep), &gf);
#endif
#if USE_LBP
			extractLBPFaceFeatures( (unsigned char*)(tarImg->imageData), (tarImg->widthStep), &gf, FLIP_MATCH);
#endif
#if USE_GABOR
			extractGaborFeatures( &gf);
#endif


#ifdef DO_MATCH

			matchedFaceID	= matchFace(&gf );
#endif


			// plot graphic results.
			cvRectangle(pFrame, pt1, pt2, cvScalar(0,0,255),2, 8, 0);

			int lEyeCenterY = ( leftEye[0].y + leftEye[1].y )/2, lEyeCenterX = ( leftEye[0].x + leftEye[1].x )/2;
			int rEyeCenterY = ( rightEye[0].y + rightEye[1].y )/2, rEyeCenterX = ( rightEye[0].x + rightEye[1].x )/2;
			CvPoint lEyeball = cvPoint(lEyeCenterX, lEyeCenterY);
			CvPoint rEyeball = cvPoint(rEyeCenterX, rEyeCenterY);

			//modified: only show eyeballs position
			cvCircle(pFrame, lEyeball,  5, cvScalar(255,0,0), -1);
			cvCircle(pFrame, rEyeball,  5, cvScalar(255,0,0), -1);

			cvReleaseImage(&tarImg);

#ifdef DO_MATCH
			// debug:
			printf("\nMatched ID: %d \n------------------\n", matchedFaceID);
			
			if (pFrame->width > 800)
			{
				int tmpW = 800;
				int tmpH = (1.0 * pFrame->height / pFrame->width)* tmpW;
				inputQueryImgResized	= cvCreateImage(cvSize(tmpW, tmpH), IPL_DEPTH_8U, 3);
				cvResize(pFrame, inputQueryImgResized, 1);
				cvNamedWindow("Input Image");
				cvShowImage("Input Image", inputQueryImgResized);
				cvReleaseImage(&inputQueryImgResized);
			}
			else
			{
				cvNamedWindow("Input Image");
				cvShowImage("Input Image", pFrame);
			}

			cvNamedWindow("Matched Face");
			cvShowImage("Matched Face", imgFaceIDTag[matchedFaceID-1]);
			cvWaitKey(100);
#endif

		}
		else
		{
			cvPutText(pFrame, "NO Face Found", cvPoint(500,30), &font, cvScalar(255,255,255));
		}

		/************************************************************************/
		/* Display                                                              */
		/************************************************************************/
#if 0
		cvResize(pFrame, inputQueryImgResized, 1);
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", inputQueryImgResized);

		cvNamedWindow("Matched Face");
		cvShowImage("Matched Face", imgFaceIDTag[matchedFaceID-1]);
		cvWaitKey(100);

#endif
		// save original image.
		//sprintf(tmppath,"output_align/%s",tmpname);
		//cvSaveImage(tmppath,pFrame);
		//system("pause");
	
		

                // show the image in the window

                cvShowImage("mainWin", pFrame );
				cvReleaseImage(&pFrame);

				frameNum++;

				if (cvWaitKey(10) == 'q')
					exit;

			free(faceDet);

      }


}