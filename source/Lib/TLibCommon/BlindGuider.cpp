

#include <cv.h>
#include <time.h>
#include <highgui.h>
#include <math.h>

#include "iVAS_define.h"
#include "iAVS_imagetools.h"
#include "iVAS_global.h"
#include "iAVS_initialization.h"
#include "iVAS_analyse.h"

CameraPara					g_stCapCamera;
BKSegment_Para				g_BKSegmentPara;


int pcTest()
{

	IplImage *frame = 0;
	IplImage *frameCopy = 0;
	CvCapture *capture  = 0;
	int keyPressed = 0;

	clock_t startTime, endTime;
	double elapseTime;


	
	/*open camera*/
	capture = cvCaptureFromCAM(-1);
	if ( !capture )
	{
		fprintf(stderr,"Open camera failed\n");
		return -1;
	}

	frame = cvQueryFrame( capture );
	if ( !frameCopy )
	{
		frameCopy = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, frame->nChannels);
	}


	cvNamedWindow("frame in");
	cvNamedWindow("frame out");


	while(1)
	{
		frame = cvQueryFrame( capture );
		if ( !frame )
		{
			break;
		}

		if ( frame->origin == IPL_ORIGIN_TL )
		{
			cvCopy(frame, frameCopy, 0);
		}
		else
		{
			cvFlip(frame, frameCopy, 0);
		}

		
		/*-----------Image processing code here-------------------*/

		/* Convert RGB to YUV and downsample, to extract Y*/
//		getLumFromRGB(frameCopy,systemHandle);
	
		startTime = clock ();

//		visionAnalysis(systemHandle);
		nv_analysis(frameCopy->imageData, &g_BKSegmentPara);

		endTime = clock();

		/*--------------------------------------------------------*/

		elapseTime = 1.0 * (endTime - startTime) / CLOCKS_PER_SEC * 1000;
		printf("\nTime=%6.2fms", elapseTime);


		/*show the results in frameCopy*/
		cvShowImage("frame in",frame);
		cvShowImage("frame out", frameCopy);

		keyPressed = cvWaitKey(1);

	}

	cvReleaseCapture( &capture );
	cvDestroyWindow("frame in");
	cvDestroyWindow("frame out");

	return 0;
}



int main(int argc, char *argv[])
{
//	SystemHandle* systemHandle = (SystemHandle*) malloc(sizeof(SystemHandle));

//	configureSystem(systemHandle);
	
//	initSystem(systemHandle);
//	setCamera(&g_stCapCamera, FRAME_HEIGHT, FRAME_WIDTH );
	g_stCapCamera.Width =FRAME_WIDTH;
	g_stCapCamera.Height = FRAME_HEIGHT;
	g_stCapCamera.Width_blk = FRAME_WIDTH>>1;
	g_stCapCamera.Height_blk = FRAME_HEIGHT>>1;
	g_stCapCamera.Captured = 1;
	g_stCapCamera.bQuit = 0;/**/

    InitBKSegmentPara(&g_stCapCamera, &g_BKSegmentPara);

	pcTest();

//	closeSystem(systemHandle);
//	ReleaseBKSegmentPara(&g_BKSegmentPara);


	return 0;
}


