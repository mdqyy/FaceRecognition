#ifndef AFFINE_WARPING_H_INCLUDED
#define AFFINE_WARPING_H_INCLUDED

//#define BILINEAR_WARPING
#define AVERAGING_WARPING

#define NUM_CTRL_PTS		4

#define warpedImgW			80
#define warpedImgH			80


#define FIXED_LEFT_EYE_X	25	//2013.2.11 20->25
#define FIXED_LEFT_EYE_Y	22	//2013.2.11 20->25->22
#define FIXED_RIGHT_EYE_X	58	//2013.2.11 60->55->58
#define FIXED_RIGHT_EYE_Y	25	//2013.2.11 20->25
#define FIXED_LEFT_MOUTH_X	20
#define FIXED_LEFT_MOUTH_Y	60
#define FIXED_RIGHT_MOUTH_X	60
#define FIXED_RIGHT_MOUTH_Y	60

#define warpedImgChNum		3

#define warpedEyeCenY		25
#define warpedEyeCenLtX		30
#define warpedEyeCenRtX		50

#define warpedMouthY		55
#define warpedMouthCnrLtX	30
#define warpedMouthCnrRtX	50

/*include*/

#include "Eigen/Core"
#include "Eigen/Array"
#include "Eigen/LU"
#include "Eigen/Cholesky"
#include "global.h"

USING_PART_OF_NAMESPACE_EIGEN

/*class*/

typedef struct frameMap{
	double *x;
	double *y;
}frameMap;


typedef struct tagFaceWarpingStruct
{
	unsigned char	*	imgDataWarpedFace;
	unsigned char	* imgDataSrcFace;

	frameMap		* frameMapxy;			// ctrl pts in Reference frame.
	frameMap		* frameMapXY;			// ctrl pts in Current frame.

}FaceWarpingStruct;



/*utilities*/

void initFaceWarping();
void closeFaceWarping();

void faceWarping(	IplImage * srcImg, IplImage * tarImg,
					int srcCtlPt01row, int srcCtlPt01col,
					int srcCtlPt02row, int srcCtlPt02col,
					int srcCtlPt03row, int srcCtlPt03col,
					int srcCtlPt04row, int srcCtlPt04col,
					int tarCtlPt01row, int tarCtlPt01col,
					int tarCtlPt02row, int tarCtlPt02col,
					int tarCtlPt03row, int tarCtlPt03col,
					int tarCtlPt04row, int tarCtlPt04col	);
void faceRotate(CvPoint* leftEye, CvPoint* rightEye, IplImage* src, IplImage* dst, int faceW, int faceH);
void grayDownsample(IplImage* src, FACE3D_Type * gf);

#endif