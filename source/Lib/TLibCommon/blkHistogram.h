/** \file blkHistogram.h
    \brief Compute color histogram of blocks of an image.

    No Further Details.
*/#pragma once

#include <cv.h>
#include <highgui.h>


typedef struct histogram
{
	uchar entry[256];
} HIST;

class blkHistogram
{
public:
	//blkHistogram(void);
	blkHistogram(int nImg, int nBlockX, int nBlockY);
	~blkHistogram(void);

    /** \brief Compute color histogram for the i_th block of Image input
     * Compute color histogram for the i_th block of Image input.
     * @param input The input image.
     * @param i The ith block of input image.
     */
	void extractHist(IplImage * input, int i);

	HIST * blkHist;

	int m_nImg;
	int m_nBlockX, m_nBLockY;
	int m_nInterval;
	int m_histLen;

	
};


float computeHisSim(blkHistogram * blkHist1, blkHistogram * blkHist2);
float computeHisSimLOW(blkHistogram * blkHist1, blkHistogram * blkHist2);

void DOPCA(const CvMat* allTrainingData,  CvMat ** eigenVectors, CvMat ** dataMean, int eigenNum);
