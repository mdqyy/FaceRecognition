/** @file */
#pragma once

#include "cv.h"
#include "global.h"

#include "ConvNN.h"

#include "faceDetector.h"


/**	
* Eye Detector Class. This Class Runs the OpenCV Haar Detect Functions for finding Eyes.
*/
class eyesDetector
{
public:
    /**
    * Eye Information that can describe the detected Eye on the Face Image
    @see struct eyes
    */
    eyes eyesInfo;

    /**
    *The Constuctor
    *Initializes internal variables
    */
    eyesDetector(void);
	~eyesDetector(void);

    /**
    *function to run the Detection Algorithm on param image
    *@param input The IplImage on which the Algorithm should be run on
    *@param pos pos[0-1]:left eye, pos[2-3]:right eye,pos[4-5]:mouth
    *@return 1 if success , 0 for failure
    */
	void runEyeDetector(IplImage * pFrame,IplImage * gray_face, faceDetector * faceDet, CvPoint pos[]);

    void runRightEyesDetector(IplImage *orgFace, CvPoint eyeCenter, int maxRad, CNNIO *rightCNNIO);

	void runLeftEyesDetector(IplImage *orgFace, CvPoint eyeCenter, int maxRad, CNNIO *leftCNNIO);

//	void runRightEyesDetector(IplImage * input,IplImage * fullImage,CvPoint LE);

	void randomLeftEye(IplImage *orgFace, CvPoint leftEyeCenter, CNNIO *leftCNNIO, CvPoint avgLeftEye[2], int maxRadL, int UL_x, int UL_y);

	void randomRightEye(IplImage *orgFace, CvPoint rightEyeCenter, CNNIO *rightCNNIO, CvPoint avgRightEye[2],int maxRadR, int UL_x, int UL_y);
    /**
    *Returns 1 or 0 , depending on Success or Failure of the detection algorithm
    *@result 1 on success , 0 on failure
    */
    int checkEyeDetected();


	CNNIO * faceCNNIO;
	CNNIO * leftCNNIO;
	CNNIO * rightCNNIO;




private:
    /**
    *Eye Cascade Structure 1
    */
    ConvNN * faceCNN;
	
	ConvNN * leftEyeCNN_cnr;
	ConvNN * leftEyeCNN_ball;

	ConvNN * rightEyeCNN_cnr;
	ConvNN * rightEyeCNN_ball;





    /**
    *Internal Variable to Track if Both eyes were detected
    *@see checkEyeDetected
    */
    int bothEyesDetected;

	void copyInputData(IplImage *grayImg);

	CvMat*  face_data ;//= cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_32FC1 );
	CvMat*  face_data_int;// = cvCreateMat( 1, CNNFACECLIPHEIGHT*CNNFACECLIPWIDTH, CV_8UC1 );
};
