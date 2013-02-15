#pragma once

#include <iostream>


#include <cv.h>
#include <highgui.h>
#include "cvgabor.h"
#include "blkHistogram.h"

class CvGaborJet
{
public:
	//CvGaborJet(void);
	CvGaborJet(int nScale, int nAngle);
	~CvGaborJet(void);

	CvGabor * getGaborFilter(int scale, int angle);

	blkHistogram * img2LBPHist(IplImage * input, int type);

	int m_nScale;
	int m_nAngle;

	CvGabor *** gbJet;
	void Initialize(void);

	// Tan
	//CvMat ** gaborKen;
	int halfKerSize;
	int kerSize;
	CvMat ** normalizedOddKernel;
	CvMat ** normalizedEvenKernel;

	
};
