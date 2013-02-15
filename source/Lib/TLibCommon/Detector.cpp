/** @file */

/*
    Detector Class - Inherits FACE DETECTOR AND EYE DETECTOR CLASS
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
#include <stdio.h>
#include <windows.h>
#include "Detector.h"
#include "cv.h"

#include "tracker.h"
#include "utils.h"



IplImage * cropFaceImg(IplImage * img,CvPoint plefteye,CvPoint prighteye, IplImage *mask)
{

    IplImage *face= cvCreateImage( cvSize(FACECLIPWIDTH, FACECLIPHEIGHT),8,img->nChannels);
    IplImage *imgDest = cvCreateImage( cvSize(img->width,img->height),8,img->nChannels);
    cvZero(face);
    double yvalue=prighteye.y-plefteye.y;
    double xvalue=prighteye.x-plefteye.x;
    double ang= atan(yvalue/xvalue)*(180/CV_PI);


    double width= sqrt(pow(xvalue,2)+pow(yvalue,2));
    double ratio=width/NORMALEYELENGTH;
    
	double sidePad=eyeSidePad*ratio;
    double topPad=eyeTopPad*ratio;
    double bottomPad=eyeBottomPad*ratio;

    CvPoint p1LeftTop,p1RightBottom;
    
	p1LeftTop.x=plefteye.x-(int)sidePad;
    p1LeftTop.y=(plefteye.y)-(int)topPad;
    p1RightBottom.x=plefteye.x+(int)(width+sidePad);
    p1RightBottom.y=(plefteye.y)  + (int)bottomPad;
    
	rotate(ang,(float)plefteye.x ,(float)plefteye.y,img,imgDest);

	//cvShowImage("eyeFitting", imgDest);

    cvSetImageROI(imgDest,cvRect(p1LeftTop.x,p1LeftTop.y,p1RightBottom.x-p1LeftTop.x,p1RightBottom.y-p1LeftTop.y));	

    cvResize(imgDest,face,CV_INTER_LINEAR);
    cvResetImageROI(imgDest);

	cvReleaseImage(&imgDest);

	if(mask == NULL)
		return face;
	else
	{
		IplImage *EclipseFace= cvCreateImage( cvSize(FACECLIPWIDTH, FACECLIPHEIGHT),8,img->nChannels);

		cvOr(face, mask, EclipseFace, NULL);

		//cvReleaseImage(&face);
		

		return EclipseFace;

	}
}


detector::detector()
{
    messageIndex=-1;

    clippedFace=0;
    boolClipFace=0;
    totalFaceClipNum=0;
    clipFaceCounter=0;
    prevlengthEye=0;
    inAngle=0;
    lengthEye=0;
    widthEyeWindow=0;
    heightEyeWindow=0;


}

IplImage * detector::clipFace(IplImage * inputImage)
{
    /*if (inputImage==0)
        return 0;
    if (eyesInformation.LE.x>0 && eyesInformation.LE.y>0 &&eyesInformation.RE.x>0 && eyesInformation.RE.y>0 )
    {
        IplImage *face=cropFaceImg(inputImage,eyesInformation.LE,eyesInformation.RE, NULL);
        return face;
    }
    else
        return 0;*/

	return NULL;

}

void detector::preprocessing(IplImage *inputImage)
{
	cvEqualizeHist(inputImage, inputImage);
}

int detector::runDetector(IplImage * input)
{
    messageIndex=-1;
    static int flag;
    if (input==0)
        return -1;


     runFaceDetector(input);

	 if(checkFaceDetected()==1)
	 {
		 IplImage * clipFaceImage=clipDetectedFace(input);
		 cvReleaseImage(&clipFaceImage);
	 }

    return 0;
}
int detector::finishedClipFace()
{
    if (totalFaceClipNum>0 && finishedClipFaceFlag==1)
    {
        finishedClipFaceFlag=0;
        return 1;

    }
    else
        return 0;
}

IplImage ** detector::returnClipedFace()
{
    IplImage**temp =clippedFace;
    clippedFace=0;

    return temp;
}
void detector::startClipFace(int num)
{
    clippedFace =new IplImage * [num];
    totalFaceClipNum=num;
    clipFaceCounter=num;
    boolClipFace=1;
}

void detector::stopClipFace()
{
    totalFaceClipNum=0;
    clipFaceCounter=0;
    boolClipFace=0;
    finishedClipFaceFlag=0;
    int i=0;
    for (i=0;i<totalFaceClipNum;i++)
    {
        if (clippedFace[i]!=0)
            cvReleaseImage(&clippedFace[i]);
    }
    if (clippedFace!=0)
        delete [] clippedFace;


}
int detector::detectorSuccessful()
{
    if (messageIndex==4)
        return 1;

    return 0;
}
char * detector::queryMessage()
{
    char *message0="Please come closer to the camera.";
    char *message1="Please go little far from the camera.";
    char *message2="Unable to Detect Your Face.";
    char *message3="Tracker lost, trying to reinitialize.";
    char *message4="Tracking in progress.";
    char *message6="Capturing Image Finished.";

    if (messageIndex==-1)
        return 0;
    else if (messageIndex==0)
        return message0;
    else if (messageIndex==1)
        return message1;
    else if (messageIndex==2)
        return message2;
    else if (messageIndex==3)
        return message3;
    else if (messageIndex==4)
        return message4;
    else if (messageIndex==5)
        return messageCaptureMessage;
    else if (messageIndex==6)
        return message6;
    return 0;

}


