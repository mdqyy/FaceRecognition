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
#include <time.h>

#include "TLibCommon/EyesDetector.h"
#include "TLibCommon/Detector.h"
#include "TLibCommon/global.h"
#include <fstream>
#include "TLibCommon/ConvNN.h"
#include "TLibCommon/faceFeature.h"
#include "TLibCommon/affineWarping.h"
//extern "C"
//{
#include "kmean.h"
//}



#define WRITE_FEATURE_DATUM_2_FILE
#define TRAIN_IMAGE_DIR			"../../image/train/"
#define	STR_INPUT_IMAGE_DIR		"../../image/train/input_align_2/"
#define TRAIN_INPUT_TXT_FILE    "../../image/trainIn.txt"
#define TRAIN_OUTPUT_TXT_FILE	"../../image/trainOut.txt"
#define BIN_FILE				"../../image/faces.bin"
#define IMAGE_TAG_DIR			"../../image/ImgTag/"
#define LGT_CENTERS_FILE		"../../image/centers.bin"

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

void initLGT( LGTStruct *gLGT, FACE3D_Type * gf);
void processFileList();
void getGaborResponse(LGTStruct *gLGT, FACE3D_Type * gf);
//void convolution2D(unsigned char *src, float *dst, double *kernel, int size, int height, int width);
void kMeanCenters(LGTStruct *gLGT, FACE3D_Type * gf, KMeanLGT *gk);
void shuffle(int *list, int n);

FACE3D_Type			gf;
LGTClass			gLGT;
KMeanLGT			gk;






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
	kMeanCenters(&gLGT, &gf, &gk);
	
	//-------------------


	//-------------------
	// closing.
	//-------------------
	//closeFaceWarping();
	system("pause");
	return 0;
	
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

	
	// release
	cvReleaseImage(&tarFrame);
	cvReleaseImage(&grayFrame);
	free(tmpGaborResponse); tmpGaborResponse = NULL;
	free(tmpImageData); tmpImageData = NULL;
	delete detectEye;
	delete faceDet;
	cvReleaseImage(&gray_face_CNN);
	




}







/* Train k-means centers in each region */
void kMeanCenters(LGTStruct *gLGT, FACE3D_Type * gf, KMeanLGT *gk)
{
	int i, j, k, m, n, h, w, tmpCnt;
	int nPoints, nDim;
	int curRow, curCol, ptr;
	int imageIdx;
	int regionH = gLGT->LGTRegionH;
	int regionW = gLGT->LGTRegionW;
	int numInLastGroup = gLGT->validFaces % gLGT->numInGroup + 10;  // the last group contains the remainders
	int numGroups = (gLGT->validFaces - numInLastGroup)/ gLGT->numInGroup + 1; //Actual number of groups
	float *gaborResponse = gLGT->gaborResponse;

	FILE *fb = fopen(LGT_CENTERS_FILE, "w+b");
	if(fb==NULL){
		printf("open file centers.bin failed!\n");fflush(stdout);
		exit(-1);
	}
	fwrite(&gLGT->numLGTRegionH, sizeof(int), 1, fb);	//region number in row
	fwrite(&gLGT->numLGTRegionW, sizeof(int), 1, fb);	//region number in colum
	fwrite(&gLGT->k2, sizeof(int), 1, fb);				//number of centers
	
	//random index list for first k-mean
	int *list = (int *)malloc(gLGT->validFaces * sizeof(int));
	for ( i = 0; i < gLGT->validFaces; i++)
	{
		list[i] = i;
	}
	//initialize
	nDim = 2 * gf->nGabors;
	initKMeanLGT( gk, gLGT->k1, gLGT->k2, gLGT->numInGroup*regionH*regionW, numGroups, numInLastGroup*regionH*regionW, nDim, gLGT->validFaces * regionH * regionW);
	
	//for each region
	for ( m = 0; m < gLGT->numLGTRegionH; m++)
	{
		for (n = 0; n < gLGT->numLGTRegionW; n++)
		{
			//shuffle list to generate psudo-random sequence
			shuffle(list, gLGT->validFaces);
			for ( i = 0; i < numGroups; i++)
			{
				if ( i < (numGroups - 1)) 
				{
					// not the last group
					for ( j = 0; j < gLGT->numInGroup; j++)
					{
						//each image in group
						imageIdx = list[i * gLGT->numInGroup + j];
						
						for ( h = 0; h < regionH; h++)
						{
							for ( w = 0; w <regionW; w++)
							{
								curRow = m * regionH + h;
								curCol = n * regionW + w;
								ptr = imageIdx * gLGT->stepImage + curRow * gLGT->stepWidth + curCol * gLGT->stepPixel;
								//memcpy( &gk->firstInput[j * regionW * regionH + h * regionW + w][0], &gaborResponse[ptr], nDim * sizeof(float));
								gk->firstInput[j * regionW * regionH + h * regionW + w] = &gaborResponse[ptr];
							}
						}
					}
					//apply kmean for the first time
					nPoints = gLGT->numInGroup * regionH * regionW;
					kMeanClustering(gk->firstInput, nPoints, nDim, gLGT->k1, &gk->km1[i], FALSE);
				}
				else
				{
					// the last group
					for ( j = 0; j < numInLastGroup; j++)
					{
						//each image in group
						imageIdx = list[i * gLGT->numInGroup + j];
						
						for ( h = 0; h < regionH; h++)
						{
							for ( w = 0; w <regionW; w++)
							{
								curRow = m * regionH + h;
								curCol = n * regionW + w;
								ptr = imageIdx * gLGT->stepImage + curRow * gLGT->stepWidth + curCol * gLGT->stepPixel;
								//memcpy( &gk->firstInputLastGroup[j * regionW * regionH + h * regionW + w][0], &gaborResponse[ptr], nDim * sizeof(float));
								gk->firstInputLastGroup[j * regionW * regionH + h * regionW + w] = &gaborResponse[ptr];
							}
						}
					}
					//apply kmean for the first time
					nPoints = numInLastGroup * regionH * regionW;
					kMeanClustering(gk->firstInputLastGroup, nPoints, nDim, gLGT->k1, &gk->km1[i], FALSE);
				}

			} // end group kmean

			//second kmean: put all groups*k1 centers to train new k2 centers
			for ( i = 0; i < numGroups; i++)
			{
				for ( j = 0; j < gLGT->k1; j++)
				{

					gk->secondInput[i * gLGT->k1 + j] = gk->km1[i].kMeanClusterCenters[j];
				}
			}
			//apply second kmean
			nPoints = numGroups * gLGT->k1;
			kMeanClustering(gk->secondInput, nPoints, nDim, gLGT->k2, &gk->km2, FALSE);

			//final kmean
			nPoints = gLGT->validFaces * regionH * regionW;
			//init with previous trained centers
			gk->km3.kMeanClusterCenters = gk->km2.kMeanClusterCenters;
			tmpCnt = 0;
			for (i = 0; i < gLGT->validFaces; i++)
			{
				for ( h = 0; h < regionH; h++)
				{
					for ( w = 0; w <regionW; w++)
					{
						curRow = m * regionH + h;
						curCol = n * regionW + w;
						ptr = i * gLGT->stepImage + curRow * gLGT->stepWidth + curCol * gLGT->stepPixel;
						gk->finalInput[tmpCnt] = &gaborResponse[ptr];
						tmpCnt++;
					}
				}
			}
			kMeanClustering(gk->finalInput, nPoints, nDim, gLGT->k2, &gk->km3, TRUE);

			//write centers to binary file
			for ( i = 0; i < gLGT->k2; i++)
			{
				for ( j = 0; j < nDim; j++)
				{
					fwrite(&gk->km3.kMeanClusterCenters[i][j], sizeof(float), 1, fb);
				}
			}

			
		}
	}
	
	fclose(fb);
	free(list);
	list = NULL;
}// end kMeanCenters


// shuffle array to randomly select group members
void shuffle(int *list, int n) 
{    
    srand((unsigned) time(NULL)); 
	int t, j;


    if (n > 1) 
	{
        int i;
        for (i = n - 1; i > 0; i--) 
		{
            j = i + rand() / (RAND_MAX / (n - i) + 1);
            t = list[j];
            list[j] = list[i];
            list[i] = t;
        }
    }
}