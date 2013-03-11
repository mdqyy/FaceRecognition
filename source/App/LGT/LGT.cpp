/*
* Local Gabor Textons(LGT) Construction
* "Face Recognition with Local Gabor Textons"
* Paper author Zhen Lei, Stan Z. Li
*
* Code: Zhi
* 3/8/2013
*/

#include <stdio.h>
#include <io.h>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>

#include "TLibCommon/EyesDetector.h"
#include "TLibCommon/Detector.h"
#include "TLibCommon/global.h"
#include <fstream>
#include "TLibCommon/ConvNN.h"
#include "TLibCommon/faceFeature.h"
#include "TLibCommon/affineWarping.h"
#include "kmean.h"


#define WRITE_FEATURE_DATUM_2_FILE
#define TRAIN_IMAGE_DIR			"../../image/train/"
#define	STR_INPUT_IMAGE_DIR		"../../image/train/input_align_2/"
#define TRAIN_INPUT_TXT_FILE    "../../image/trainIn.txt"
#define TRAIN_OUTPUT_TXT_FILE	"../../image/trainOut.txt"
#define BIN_FILE				"../../image/faces.bin"
#define IMAGE_TAG_DIR			"../../image/ImgTag/"

typedef struct LGTStruct
{
	float *gaborResponse;
	int	  LGTRegionH;
	int   LGTRegionW;
	int   numLGTRegionH;
	int   numLGTRegionW;
	int   stepImage;
	int   stepWidth;
	int   stepPixel;
	int   numImages;
	int   numInGroup;		//num of images in each group
	int	  numGroups;
	int   k1;				//num of centers in first k-means
	int	  k2;				//num of final centers
	int   validFaces;		//num of faces that have been detected
}LGTClass;


// demo only
#define MAX_NUM_FACE_ID_TAG			MAX_FACE_ID

using namespace std;
// ConvNN 
int    NSAMPLES = 1;
int    MAX_ITER = 1;
int    NTESTSAMPLES = 1;

#define BUFSIZE 20

FACE3D_Type			gf;
LGTClass			gLGT;
void getGaborResponse(LGTStruct *gLGT, FACE3D_Type * gf);
void convolution2D(unsigned char *src, float *dst, double *kernel, int size, int height, int width);



void initLGT( LGTStruct *gLGT, FACE3D_Type * gf)
{
	gLGT->numInGroup	= 10;
	gLGT->k1			= 10;				//num of centers in first k-means
	gLGT->k2			= 64;				//num of final centers
	gLGT->LGTRegionH	= 10;
	gLGT->LGTRegionW	= 10;
	gLGT->numLGTRegionH	= 8;
	gLGT->numLGTRegionW	= 8;
	gLGT->numImages		= gf->fileList.listLength;
	gLGT->numGroups		= (int)(gLGT->numImages / gLGT->numInGroup) + 1;
	//
	gLGT->stepPixel		= gf->nGabors * 2;
	gLGT->stepWidth		= gLGT->stepPixel * gf->tWidth;
	gLGT->stepImage		= gLGT->stepWidth * gf->tHeight;
	//allocate gabor response storage
	gLGT->gaborResponse = (float *)malloc( gLGT->stepImage * gLGT->numImages * sizeof(float));
	gLGT->validFaces	= 0;
}

void processFileList()
{
	char inputdir[100]= STR_INPUT_IMAGE_DIR;

	char tagImgdir[100]= IMAGE_TAG_DIR;
	char tagImgName[500];
	int	i;
	

	// count how many face IDs in training list

	IplImage *	imgFaceIDTag[MAX_NUM_FACE_ID_TAG];
	int			numTaggedFaces = 0;

	for (i=0; i<MAX_NUM_FACE_ID_TAG; i++)
	{
		sprintf( tagImgName, "%s%d.JPG",tagImgdir, i+1);
		imgFaceIDTag[i]		= cvLoadImage(tagImgName,3);

		if (imgFaceIDTag[i] == NULL)
		{
			printf("\nNum of tagged faces: %d\n", i);fflush(stdout);
			numTaggedFaces = i;
			break;
		}
	
	}
	gf.numIDtag = numTaggedFaces;

	//process image list
	char to_search[260];
	int  numImgs = 0;
	// for each face tag
	for ( int ii = 0; ii < numTaggedFaces; ii++)
	{
		sprintf(to_search, "%s%d/*.jpg",TRAIN_IMAGE_DIR, ii+1);
		long handle;                                                //search handle
		struct _finddata_t fileinfo;                          // file info struct
		handle=_findfirst(to_search,&fileinfo);         
		if(handle != -1) 
		{
			do
			{
				char *tmpPath = &gf.fileList.fileName[numImgs][0];
				sprintf(tmpPath,"%s%d/%s",TRAIN_IMAGE_DIR, ii+1, fileinfo.name);
				gf.fileList.fileID[numImgs] = ii+1;
				numImgs ++;
			}while(_findnext(handle,&fileinfo) == 0);               
	
			_findclose(handle);
		}
	}
	assert(numImgs < MAX_INPUT_IMAGES);
	gf.fileList.listLength = numImgs; // store list length
				
}

int main(int argc, char** argv)
{
	//-------------------
	// initializations.
	//-------------------

	initFaceWarping();	
	initFaceFeature( &gf, 80, 80);
	processFileList();
	initLGT(&gLGT, &gf);

	//-------------------
	// data access.
	getGaborResponse(&gLGT, &gf);
	//-------------------


	//-------------------
	// closing.
	//-------------------
	//closeFaceWarping();
	system("pause");
	
}



//-------------------------------------------------------------------
void getGaborResponse(LGTStruct *gLGT, FACE3D_Type * gf)
{
	int		validFaces = 0;
	int		numImagesInList = gf->fileList.listLength;
	int		i,j,m,n;
	char	*tmpPath;
	int		gaborWSize = gf->gaborWSize;
	//CvMat	gaborKernel[MAX_NUM_GABOR];
	float	*gaborResponse = gLGT->gaborResponse;
	float	*tmpGaborResponse;
	unsigned char *tmpImageData;
	int		ptr;
	int		stepImage = gLGT->stepImage;
	int		stepPixel = gLGT->stepPixel;
	int		stepWidth = gLGT->stepWidth;

	//init
	
	IplImage *tarFrame = cvCreateImage( cvSize(warpedImgW, warpedImgH), IPL_DEPTH_8U, warpedImgChNum );
	IplImage *grayFrame = cvCreateImage(cvSize(warpedImgW, warpedImgH), IPL_DEPTH_8U, 1 );
	tmpGaborResponse = (float*)malloc( gf->tHeight * gf->tWidth * sizeof(float));
	tmpImageData = (unsigned char*)malloc(gf->tHeight * gf->tWidth * sizeof(unsigned char));
	eyesDetector * detectEye = new eyesDetector;
	faceDetector * faceDet =  new faceDetector();
	IplImage*  gray_face_CNN = cvCreateImage(cvSize(CNNFACECLIPHEIGHT,CNNFACECLIPWIDTH), 8, 1);

	//Establish gabor kernels from coefficients
	//for ( i = 0; i < gf->nGabors * 2; i++)
	//{
	//	gaborKernel[i] = cvMat(gaborWSize, gaborWSize, CV_32FC1, gf->gaborCoefficients[i]);
	//}


	// Feature points array for eyes detection
	CvPoint pointPos[6];
	CvPoint *leftEye    = &pointPos[0];
	CvPoint *rightEye   = &pointPos[2];
	CvPoint *leftMouth  = &pointPos[4];
	CvPoint *rightMouth = &pointPos[5];

	for ( i = 0; i < numImagesInList; i++)
	{
		tmpPath = &gf->fileList.fileName[i][0];
		puts(tmpPath);
		IplImage *pFrame;
		pFrame = cvLoadImage(tmpPath, 3);
		if(pFrame == NULL)
		{
			printf("read image file error!\n");fflush(stdout);
			exit(-1);
		}
		if( faceDet->runFaceDetector(pFrame))
		{	
			/* there was a face detected by OpenCV. */
			
			IplImage * clonedImg = cvCloneImage(pFrame);

			detectEye->runEyeDetector(clonedImg, gray_face_CNN, faceDet, pointPos);

			cvReleaseImage(&clonedImg);

			int UL_x = faceDet->faceInformation.LT.x;
			int UL_y = faceDet->faceInformation.LT.y;

			
			//align face
			faceRotate(leftEye, rightEye, pFrame, tarFrame, faceDet->faceInformation.Width, faceDet->faceInformation.Height);
			cvCvtColor(tarFrame, grayFrame, CV_RGB2GRAY);
			//get unsigned char image data from IplImage
			for ( m = 0; m < gf->tHeight; m++)
			{
				for ( n =0; n < gf->tWidth; n++)
					{
						tmpImageData[ m * gf->tWidth + n] = CV_IMAGE_ELEM( grayFrame, unsigned char, m, n );
				}
			}


			for ( j = 0; j < gf->nGabors *2; j++)
			{
				//Apply gabor kernels
				//cvFilter2D(tmpFrame,grFrame, &gaborKernel[j], cvPoint(-1,-1));
				convolution2D( tmpImageData, tmpGaborResponse, gf->gaborCoefficients[j], gaborWSize, gf->tHeight, gf->tWidth);
				
				//store response
				for ( m = 0; m < gf->tHeight; m++)
				{
					for ( n =0; n < gf->tWidth; n++)
					{
						ptr = validFaces * stepImage + m * stepWidth + n * stepPixel + j;
						gaborResponse[ptr] = tmpGaborResponse[ gf->tWidth * m + n];
					}
				}
			}
			validFaces ++; 
		}
		cvReleaseImage(&pFrame);
	}
	gLGT->validFaces = validFaces;






}

/* 2D convolution for gray 8-bit image*/
// src : input image
// dst : output
// kernel: filter kernel
// size : kernel size n(should be odd number)
// height: image height
// width: image width
void convolution2D(unsigned char *src, float *dst, double *kernel, int size, int height, int width)
{
	assert((size % 2 == 1) && (size >= 3)); // kernel size should be odd number
	int cRow, cCol; //center row and column
	int kRow, kCol; //kernel row and column
	int posRow, posCol; //current accessing position
	int pad = (size - 1)/2;
	int twoHeight = 2 * (height-1);
	int twoWidth = 2 * (width-1);
	float sum;

	
	
	//scan image
	for (cRow = 0; cRow < height; cRow++ )
	{
		for (cCol = 0; cCol < width; cCol++)
		{
			sum = 0;
			//scan kernel
			for ( kRow = -pad; kRow < pad; kRow++ )
			{
				for ( kCol = -pad; kCol < pad; kCol++)
				{
					posRow = cRow + kRow;
					posCol = cCol + kCol;
					//out of border pixels
					if (posRow < 0) 
					{
						posRow = 0 - posRow;
					}
					else if (posRow >= height) 
					{
						posRow = twoHeight - posRow;
					}
					if (posCol < 0) 
					{
						posCol = 0 - posCol;
					}
					else if (posCol >= width) 
					{
						posCol = twoWidth - posCol;
					}

					sum += (float)src[width * posRow + posCol] * (float)kernel[ size * (kRow + pad) + kCol + pad];
				}
			}
			dst[width * cRow + cCol] = sum;
		}
	}

}//end function convulution2D







