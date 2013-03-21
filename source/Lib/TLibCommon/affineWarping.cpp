#include <stdio.h>
#include <cv.h>
#include <math.h>
#include <highgui.h>

#include "affineWarping.h"

FaceWarpingStruct	*	g_Face_warping_param;

void initFaceWarping()
{
	g_Face_warping_param				= (FaceWarpingStruct *)malloc( sizeof(FaceWarpingStruct) );

	g_Face_warping_param->frameMapxy	= (frameMap *)malloc( sizeof(frameMap) );
	g_Face_warping_param->frameMapXY	= (frameMap *)malloc( sizeof(frameMap) );

	g_Face_warping_param->frameMapxy->x = (double*)malloc( NUM_CTRL_PTS * sizeof(double) );
	g_Face_warping_param->frameMapxy->y = (double*)malloc( NUM_CTRL_PTS * sizeof(double) );

	g_Face_warping_param->frameMapXY->x = (double*)malloc( NUM_CTRL_PTS * sizeof(double) );
	g_Face_warping_param->frameMapXY->y = (double*)malloc( NUM_CTRL_PTS * sizeof(double) );

}// end: initFaceWarping()

void closeFaceWarping()
{
	free(g_Face_warping_param->frameMapXY->y);
	free(g_Face_warping_param->frameMapXY->x);
	free(g_Face_warping_param->frameMapxy->y);
	free(g_Face_warping_param->frameMapxy->x);
	free(g_Face_warping_param->frameMapXY);
	free(g_Face_warping_param->frameMapxy);
	free(g_Face_warping_param);

}// end: closeFaceWarping()


void faceWarping(		IplImage * srcImg, IplImage * tarImg,
						int srcCtlPt01row, int srcCtlPt01col,
						int srcCtlPt02row, int srcCtlPt02col,
						int srcCtlPt03row, int srcCtlPt03col,
						int srcCtlPt04row, int srcCtlPt04col,
						int tarCtlPt01row, int tarCtlPt01col,
						int tarCtlPt02row, int tarCtlPt02col,
						int tarCtlPt03row, int tarCtlPt03col,
						int tarCtlPt04row, int tarCtlPt04col	)
{
	int					r, c;			// (x, y) here is (row, col).
	int					i, ptrTarImg, ptrSrcImg;
	int					wStepSrc, wStepTar;
	int					wSrcImg, hSrcImg;
	int					tmpInt;

	double				aa, bb, cc, dd, ee, ff, gg, hh;
	double				rposproj, cposproj;
	double				lowerL, upperL, upperR, lowerR;
	double				r_weight, c_weight, weighted_L, weighted_R;
	double				tmpDouble;

	char				tmpChar;

	MatrixXd			A(NUM_CTRL_PTS*2,8);
	MatrixXd			B(NUM_CTRL_PTS*2,1);
	VectorXd			Ce(8);			// coefficients.
	VectorXd			Bb(8);			// A.transpose() * B.
	Matrix3d			C;				// coefficient matrix.
	MatrixXd			temp(8,8);

	// setup the control points.
	// note that (x, y) here is (row, col).
	// you want to warp srcImg into tarImg, thus (src = current = framemapXY), (tar = reference = frameMapxy).

	frameMap	*		frameMapxy;
	frameMap	*		frameMapXY;

	frameMapxy			= g_Face_warping_param->frameMapxy;
	frameMapXY			= g_Face_warping_param->frameMapXY;

	wSrcImg				= srcImg->width;
	hSrcImg				= srcImg->height;

	// source or current or [XY]

	*(frameMapXY->x)	= srcCtlPt01row;
	*(frameMapXY->x+1)	= srcCtlPt02row;
	*(frameMapXY->x+2)	= srcCtlPt03row;
	*(frameMapXY->x+3)	= srcCtlPt04row;

	*(frameMapXY->y)	= srcCtlPt01col;
	*(frameMapXY->y+1)	= srcCtlPt02col;
	*(frameMapXY->y+2)	= srcCtlPt03col;
	*(frameMapXY->y+3)	= srcCtlPt04col;

	// target or reference or [xy]

	*(frameMapxy->x)	= tarCtlPt01row;
	*(frameMapxy->x+1)	= tarCtlPt02row;
	*(frameMapxy->x+2)	= tarCtlPt03row;
	*(frameMapxy->x+3)	= tarCtlPt04row;

	*(frameMapxy->y)	= tarCtlPt01col;
	*(frameMapxy->y+1)	= tarCtlPt02col;
	*(frameMapxy->y+2)	= tarCtlPt03col;
	*(frameMapxy->y+3)	= tarCtlPt04col;

	// resolve transform matrix

	for(i=0; i<NUM_CTRL_PTS; i++)
	{
		A(2*i,0) = *(frameMapxy->x + i);							  A(2*i+1,0) = 0;
		A(2*i,1) = *(frameMapxy->y + i);							  A(2*i+1,1) = 0;
		A(2*i,2) = 1;												  A(2*i+1,2) = 0;
		A(2*i,3) = 0;												  A(2*i+1,3) = *(frameMapxy->x + i);
		A(2*i,4) = 0;												  A(2*i+1,4) = *(frameMapxy->y + i);
		A(2*i,5) = 0;												  A(2*i+1,5) = 1;
		A(2*i,6) = *(frameMapxy->x + i)*(-1)*( *(frameMapXY->x + i) );A(2*i+1,6) = *(frameMapxy->x + i)*(-1)*( *(frameMapXY->y + i) );
		A(2*i,7) = *(frameMapxy->y + i)*(-1)*( *(frameMapXY->x + i) );A(2*i+1,7) = *(frameMapxy->y + i)*(-1)*( *(frameMapXY->y + i) );

		B(2*i,0)   = *(frameMapXY->x + i);
		B(2*i+1,0) = *(frameMapXY->y + i);
	}	//end of creating matrices A B.

	//temp = (~A)*A;
	temp = A.transpose() * A;	//A conjugate * A.
	Bb	 = A.transpose() * B;
	temp.ldlt().solve(Bb,&Ce);	// Ce is our desired transform matrix here.


	// warping.

	aa				= Ce(0);
	bb				= Ce(1);
	cc				= Ce(2);
	dd				= Ce(3);
	ee				= Ce(4);
	ff				= Ce(5);
	gg				= Ce(6);
	hh				= Ce(7);

	wStepSrc		= srcImg->widthStep;
	wStepTar		= tarImg->widthStep;

	for ( r=0; r<warpedImgH; r++ )
	{
		for ( c=0; c<warpedImgW; c++ )
		{
			// find forward mapping.

			rposproj	= ( r * aa +
							c * bb +
							cc )		/
							( r * gg +
							c * hh +
							1 );

			cposproj	= ( r * dd +
							c * ee +
							ff )		/
							( r * gg +
							c * hh +
							1);

			// if within image, do interpolation.

			if (	( rposproj	>=0 )			&&
					( rposproj	<(hSrcImg-1) )	&&
					( cposproj	>=0 )			&&
					( cposproj	<(wSrcImg-1) )		)
			{
				// notation: L = left, R = right.

				ptrTarImg = (r*wStepTar + c*warpedImgChNum);

				//B

				ptrSrcImg = ((int)(rposproj )* wStepSrc)		+ ((int)(cposproj) * warpedImgChNum);

				upperL		= *( srcImg->imageData + ptrSrcImg );
				upperR		= *( srcImg->imageData + ptrSrcImg + warpedImgChNum );
				lowerL		= *( srcImg->imageData + ptrSrcImg + wStepSrc );
				lowerR		= *( srcImg->imageData + ptrSrcImg + wStepSrc + warpedImgChNum );

				#ifdef BILINEAR_WARPING
				r_weight  = 1 - ( rposproj - (int)(rposproj) );
				c_weight  = 1 - ( cposproj - (int)(cposproj) );

				weighted_L								= upperL * r_weight + lowerL * ( 1 - r_weight );
				weighted_R								= upperR * r_weight + lowerR * ( 1 - r_weight );

				tmpChar									= (int)(weighted_L * c_weight + weighted_R * ( 1 - c_weight));
				if(tmpChar > 127)						tmpChar = 127;
				if(tmpChar <(-128) )					tmpChar = -128;
				#endif


				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( tarImg->imageData + ptrTarImg)		= tmpChar;

				//G

				upperL		= *( srcImg->imageData + ptrSrcImg + 1 );
				upperR		= *( srcImg->imageData + ptrSrcImg + warpedImgChNum + 1 );
				lowerL		= *( srcImg->imageData + ptrSrcImg + wStepSrc + 1 );
				lowerR		= *( srcImg->imageData + ptrSrcImg + wStepSrc + warpedImgChNum + 1 );

				#ifdef BILINEAR_WARPING
				weighted_L								= upperL * r_weight + lowerL * ( 1 - r_weight );
				weighted_R								= upperR * r_weight + lowerR * ( 1 - r_weight );

				tmpChar									= (int) (weighted_L * c_weight + weighted_R * ( 1 - c_weight));
				if(tmpChar > 127)						tmpChar = 127;
				if(tmpChar <(-128) )					tmpChar = -128;
				#endif

				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( tarImg->imageData + ptrTarImg +1)	= tmpChar;

				//R

				upperL		= *( srcImg->imageData + ptrSrcImg + 2 );
				upperR		= *( srcImg->imageData + ptrSrcImg + warpedImgChNum + 2 );
				lowerL		= *( srcImg->imageData + ptrSrcImg + wStepSrc + 2 );
				lowerR		= *( srcImg->imageData + ptrSrcImg + wStepSrc + warpedImgChNum + 2 );

				#ifdef BILINEAR_WARPING
				weighted_L								= upperL * r_weight + lowerL * ( 1 - r_weight );
				weighted_R								= upperR * r_weight + lowerR * ( 1 - r_weight );

				tmpChar									= (int)(weighted_L * c_weight + weighted_R * ( 1 - c_weight));
				if(tmpChar > 127)						tmpChar = 127;
				if(tmpChar <(-128) )					tmpChar = -128;
				#endif

				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( tarImg->imageData + ptrTarImg +2)	= tmpChar;

			}// end: paint warped pixels.

		}// end: warp col.

	}//end: warp row.


	// debug

	//printf("\n\nCe: %f	%f	%f	%f	%f	%f	%f	%f	\n", Ce(0),Ce(1),Ce(2),Ce(3),Ce(4),Ce(5),Ce(6),Ce(7) );
	//system("pause");

}// end: faceWarping()

//2013.2.11
//To change the size of face, go to affineWarping.h and adjust the defs of FIXED_RIGHT_EYE_X and so on...
void faceRotate(CvPoint* leftEye, CvPoint* rightEye, IplImage* src, IplImage* dst, int faceW, int faceH)
{
	//assert( faceW == faceH);

    int		ly = ( leftEye[0].y + leftEye[1].y )/2, lx = ( leftEye[0].x + leftEye[1].x )/2;
	int		ry = ( rightEye[0].y + rightEye[1].y )/2, rx = ( rightEye[0].x + rightEye[1].x )/2;
	int		distY = ly - ry;	// y coordinates are inverse of natural images
	int		distX = rx - lx; 
	int		ptrTarImg, ptrSrcImg;
	int		wStepSrc, wStepTar;
	int		wSrcImg, hSrcImg;
	char	tmpChar;

	//double slope = sqrt((double)(distY*distY + x)));
	//double ratioY = faceH / warpedImgH;
	//double ratioX = faceW / warpedImgW;

	assert( warpedImgW == warpedImgH);
	double resAX = 1.0 *distX / (FIXED_RIGHT_EYE_X - FIXED_LEFT_EYE_X);
	double resBX = 1.0 * distY / (FIXED_RIGHT_EYE_X - FIXED_LEFT_EYE_X);
	double resAY = (-1) * resBX;
	double resBY = resAX;

	wStepSrc	= src->widthStep;
	wStepTar	= dst->widthStep;
	wSrcImg		= src->width;
	hSrcImg		= src->height;

	int r,c = 0; // row & column
	int dr, dc = 0; // distance to left eye center in row/column
	double wr, wc = 0; // warped row & column
	double	lowerL, upperL, upperR, lowerR;

	for ( r=0; r<warpedImgH; r++ )
	{
		for ( c=0; c<warpedImgW; c++ )
		{
			dr = r - FIXED_LEFT_EYE_Y;
			dc = c - FIXED_LEFT_EYE_X;
			wr = ly + resAY * dc + resBY * dr;
			wc = lx + resAX * dc + resBX * dr;
			// if within image, do interpolation.

			if (	( wr	>=0 )			&&
					( wr	<(hSrcImg-1) )	&&
					( wc	>=0 )			&&
					( wc	<(wSrcImg-1) )		)
			{
				// notation: L = left, R = right.

				ptrTarImg = (r*wStepTar + c*warpedImgChNum);

				//B channel

				ptrSrcImg = ((int)(wr )* wStepSrc)		+ ((int)(wc) * warpedImgChNum);

				upperL		= *( src->imageData + ptrSrcImg );
				upperR		= *( src->imageData + ptrSrcImg + warpedImgChNum );
				lowerL		= *( src->imageData + ptrSrcImg + wStepSrc );
				lowerR		= *( src->imageData + ptrSrcImg + wStepSrc + warpedImgChNum );

				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( dst->imageData + ptrTarImg)		= tmpChar;
				//G channel

				upperL		= *( src->imageData + ptrSrcImg + 1 );
				upperR		= *( src->imageData + ptrSrcImg + warpedImgChNum + 1 );
				lowerL		= *( src->imageData + ptrSrcImg + wStepSrc + 1 );
				lowerR		= *( src->imageData + ptrSrcImg + wStepSrc + warpedImgChNum + 1 );

				#ifdef BILINEAR_WARPING
				weighted_L								= upperL * r_weight + lowerL * ( 1 - r_weight );
				weighted_R								= upperR * r_weight + lowerR * ( 1 - r_weight );

				tmpChar									= (int) (weighted_L * c_weight + weighted_R * ( 1 - c_weight));
				if(tmpChar > 127)						tmpChar = 127;
				if(tmpChar <(-128) )					tmpChar = -128;
				#endif

				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( dst->imageData + ptrTarImg +1)	= tmpChar;

				//R channel

				upperL		= *( src->imageData + ptrSrcImg + 2 );
				upperR		= *( src->imageData + ptrSrcImg + warpedImgChNum + 2 );
				lowerL		= *( src->imageData + ptrSrcImg + wStepSrc + 2 );
				lowerR		= *( src->imageData + ptrSrcImg + wStepSrc + warpedImgChNum + 2 );

				#ifdef BILINEAR_WARPING
				weighted_L								= upperL * r_weight + lowerL * ( 1 - r_weight );
				weighted_R								= upperR * r_weight + lowerR * ( 1 - r_weight );

				tmpChar									= (int)(weighted_L * c_weight + weighted_R * ( 1 - c_weight));
				if(tmpChar > 127)						tmpChar = 127;
				if(tmpChar <(-128) )					tmpChar = -128;
				#endif

				#ifdef AVERAGING_WARPING
				tmpChar									= (lowerL + upperL + upperR + lowerR)/4;
				tmpChar									= upperL;
				tmpChar									= lowerL;

				//tmpInt									= (upperL + lowerL)/2;
				//tmpChar									= (char)tmpInt;
				#endif

				*( dst->imageData + ptrTarImg +2)	= tmpChar;

			}// end: paint warped pixels.

		}// end: warp col.

	}//end: warp row.

}


//Convert to gray and downsampling twice
void grayDownsample(IplImage* src, FACE3D_Type * gf, int frameCnt, bool isMatching)
{
	int *fImage0, *fImage1, *fImage2, *ptr;
	int tWidth, tHeight, vR, vC, W, H,tWidth1, tWidth2, tHeight1, tHeight2;
	int i;
	unsigned char tmp;
	fImage0 = gf->fImage0;
	fImage1 = gf->fImage1;
	fImage2 = gf->fImage2;
	tWidth = gf->tWidth;
	tHeight = gf->tHeight;
	tWidth1 = tWidth / 2;
	tWidth2 = tWidth / 4;
	tHeight1 = tHeight/2;
	tHeight2 = tHeight/4;
#if FLIP_MATCH
	int *fImage0flip = gf->fImage0flip;
	int *fImage1flip = gf->fImage1flip;
	int *fImage2flip = gf->fImage2flip;
#endif

	assert( tWidth == src->width);
	assert( tHeight == src->height);

	IplImage* tmpImg0 = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* tmpImg1 = cvCreateImage( cvSize(tWidth/2, tHeight/2), IPL_DEPTH_8U, 1);
	IplImage* tmpImg2 = cvCreateImage( cvSize(tWidth/4, tHeight/4), IPL_DEPTH_8U, 1);

	//convert to gray
	cvCvtColor(src,tmpImg0,CV_RGB2GRAY);
#if HISTOGRAM_EQUALIZATION
	cvEqualizeHist(tmpImg0, tmpImg0);
#endif

	for (i = 0; i < tWidth * tHeight; i++)
	{
		
		memcpy(&tmp, &(tmpImg0->imageData[i]), sizeof(char));
		fImage0[i] = tmp;
	}

	//downsample by 2
	cvResize(tmpImg0, tmpImg1, 1);

	for (i = 0; i < tWidth1 * tHeight1; i++)
	{
		memcpy(&tmp, &(tmpImg1->imageData[i]), sizeof(char));
		fImage1[i] = tmp;
	}

	//downsample by 4
	cvResize(tmpImg0, tmpImg2, 1);
	for (i = 0; i < tWidth2 * tHeight2; i++)
	{
		memcpy(&tmp, &(tmpImg2->imageData[i]), sizeof(char));
		fImage2[i] = tmp;
	}

#if FLIP_MATCH
	if (isMatching)
	{
		cvFlip(tmpImg0, NULL, 1);
		for (i = 0; i < tWidth * tHeight; i++)
		{
			fImage0flip[i] = (int) tmpImg0->imageData[i];
		}

		//downsample by 2
		cvResize(tmpImg0, tmpImg1, 1);

		for (i = 0; i < tWidth1 * tHeight1; i++)
		{
			fImage1flip[i] = (int) tmpImg1->imageData[i];
		}

		//downsample by 4
		cvResize(tmpImg0, tmpImg2, 1);
		for (i = 0; i < tWidth2 * tHeight2; i++)
		{
			fImage2flip[i] = (int) tmpImg2->imageData[i];
		}
	}
#endif

#if DEBUG_OUTPUT_ALIGNED

	//Output faces only
	char tmpPath[500];
	sprintf(tmpPath, "%s%d_0.jpg", "C:/Users/Zhi/Desktop/Debug/",frameCnt);
	cvSaveImage(tmpPath, tmpImg0);
	sprintf(tmpPath, "%s%d_1.jpg", "C:/Users/Zhi/Desktop/Debug/",frameCnt);
	cvSaveImage(tmpPath, tmpImg1);
	sprintf(tmpPath, "%s%d_2.jpg", "C:/Users/Zhi/Desktop/Debug/",frameCnt);
	cvSaveImage(tmpPath, tmpImg2);
#endif

	cvReleaseImage(&tmpImg0);
	cvReleaseImage(&tmpImg1);
	cvReleaseImage(&tmpImg2);
}