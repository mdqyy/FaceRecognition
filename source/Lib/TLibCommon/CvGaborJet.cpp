#include "CvGaborJet.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

//CvGaborJet::CvGaborJet(void)
//{
//}

CvGaborJet::~CvGaborJet(void)
{
}


CvGaborJet::CvGaborJet(int nScale, int nAngle)
{
	gbJet = new CvGabor ** [nScale];

	for(int i=0; i<nScale; i++)
	{
		gbJet[i] = new CvGabor *[nAngle];
	}

	m_nAngle = nAngle;
	m_nScale = nScale;
}

void CvGaborJet::Initialize(void)
{
	if(gbJet == NULL)
	{
		printf("Jet has not been initialized!\n");
		exit(0);
	}

	for(int i=0; i<m_nScale; i++)
	{
		for(int j=0; j<m_nAngle; j++)
		{
			gbJet[i][j] = new CvGabor;

			gbJet[i][j]->setNumAng((double)m_nAngle);

			gbJet[i][j]->Init( j, i, 2*PI, sqrt(2.0));
		}
	}


	//normalizedOddKernel = new CvMat*[8];
	//normalizedEvenKernel = new CvMat*[8];

	//GaborKernel(8, normalizedOddKernel,normalizedEvenKernel, &halfKerSize, &kerSize, 120, 120);
}


CvGabor * CvGaborJet::getGaborFilter(int scale, int angle)
{
	return gbJet[scale][angle];
}

blkHistogram * CvGaborJet::img2LBPHist(IplImage * input, int type)
{
	IplImage * normalizedImg, *cvtImg;

	string outString;

	blkHistogram * imgHist = new blkHistogram(m_nAngle*m_nScale, 12,12);

//	delete imgHist;

	if(input->nChannels != 1)
	{
		normalizedImg = cvCreateImage(cvSize(input->width,input->height), input->depth, 1);
		cvCvtColor(input, normalizedImg, CV_BGR2GRAY);
	}
	else
		normalizedImg = cvCloneImage(input);

	int prcImg = 0;

	if(type >=CV_GABOR_LBP_BP || type== CV_ORGIMG)
	{
		cvtImg = cvCreateImage(cvSize(input->width-2,input->height-2), 8, 1);
	}
	else
	{
		cvtImg = cvCreateImage(cvSize(input->width,input->height), 8, 1);
	}


	for( int scale = 0; scale < m_nScale; scale ++)
	{
		for(int ang =0; ang <m_nAngle; ang++)
		{
			CvGabor * gbFilter = getGaborFilter(scale,ang);

			gbFilter->conv_img(normalizedImg,cvtImg,type);

#ifndef     COMPLEXTEST
			imgHist->extractHist(cvtImg, prcImg);

			/*char outName[100];
			sprintf(outName, "out_%d_%d.bmp", scale,ang);
			outString = outName;
			cvSaveImage(outString.c_str(),cvtImg, 0);*/
#endif
			prcImg++;
		}

	}

//	cvSaveImage("outcurrent.jpg",input, 0);
	
	cvReleaseImage(&normalizedImg);
	cvReleaseImage(&cvtImg);

	//return NULL;
	return imgHist;
}
