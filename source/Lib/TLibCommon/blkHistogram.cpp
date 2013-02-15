#include "blkHistogram.h"
#include <stdio.h>

void DOPCA(const CvMat* allTrainingData,  CvMat ** eigenVectors, CvMat ** dataMean, int eigenNum)
{
	int nSamples = allTrainingData->rows;
	int nfeatures = allTrainingData->cols;
	int nEigenAtMost = MIN(nSamples, nfeatures);

	CvMat* tmpEigenValues = cvCreateMat(nEigenAtMost,1,  CV_32FC1);
	CvMat* tmpEigenVectors = cvCreateMat(nEigenAtMost, nfeatures, CV_32FC1);
	*dataMean = cvCreateMat(1, nfeatures, CV_32FC1 );

	cvCalcPCA(allTrainingData, *dataMean, 
		tmpEigenValues, tmpEigenVectors, CV_PCA_DATA_AS_ROW);

	CvMat * eigenValues = cvCreateMat( eigenNum,1, CV_32FC1);
	*eigenVectors = cvCreateMat(eigenNum, nfeatures, CV_32FC1);

	CvMat G;
	cvGetRows(tmpEigenValues, eigenValues, 0, eigenNum);
	//cvCopy(&G, eigenValues);

	cvGetRows(tmpEigenVectors, &G, 0, eigenNum);
	cvCopy(&G, *eigenVectors);
	

	cvReleaseMat(&tmpEigenVectors);
	cvReleaseMat(&tmpEigenValues);
	//cvReleaseMat(&dataMean);
	cvReleaseMat(&eigenValues);
	printf("Done (%d/%d)\n", eigenNum, nEigenAtMost);
}


float computeHisSim(blkHistogram * blkHist1, blkHistogram * blkHist2)
{
	int hitLen= blkHist1->m_histLen;
	float histSim =0;

	for( int i=0; i< hitLen; i++)
	{
		uchar * pEntry1 = (uchar *)(blkHist1->blkHist[i]).entry;
		uchar * pEntry2 = (uchar *)(blkHist2->blkHist[i]).entry;

		for(int j=0; j< 256; j++)
		{
			if((pEntry2[j]+pEntry1[j]) != 0)
			histSim += ((pEntry1[j]>pEntry2[j]? pEntry2[j]:pEntry1[j])/(float)(pEntry2[j]+pEntry1[j]));
		//	printf("%x %d %d %f\n", j, pEntry1[j], pEntry2[j],histSim);
		}
	}

	return histSim;
}

float computeHisSimLOW(blkHistogram * blkHist1, blkHistogram * blkHist2)
{
	int hitLen= blkHist1->m_histLen;
	float histSim =0;

	for( int i=0; i< 144; i++)
	{
		uchar * pEntry1 = (uchar *)(blkHist1->blkHist[i]).entry;
		uchar * pEntry2 = (uchar *)(blkHist2->blkHist[i]).entry;

		for(int j=0; j< 256; j++)
		{
			if((pEntry2[j]+pEntry1[j]) != 0)
				histSim += ((pEntry1[j]>pEntry2[j]? pEntry2[j]:pEntry1[j])/(float)(pEntry2[j]+pEntry1[j]));
			//	printf("%x %d %d %f\n", j, pEntry1[j], pEntry2[j],histSim);
		}
	}

	return histSim;
}


blkHistogram::blkHistogram(int nImg, int nBlockX, int nBlockY)
{
	m_histLen = nImg*nBlockY*nBlockX;
	m_nImg = nImg;
	m_nBlockX = nBlockX;
	m_nBLockY = nBlockY;
	m_nInterval = m_nBlockX*m_nBLockY;

	blkHist = new HIST [m_histLen];

	for(int i=0; i<m_histLen; i++)
	{
		//blkHist[i] = new HIST;

		memset(blkHist[i].entry, 0 , sizeof(uchar)*256);
	}
}

blkHistogram::~blkHistogram(void)
{
	if(blkHist != NULL)
	{
		delete blkHist;
		blkHist = NULL;
	}
	
}


void blkHistogram::extractHist(IplImage * input, int ithImg)
{
	int croppedWidth  = input->width+m_nBlockX  - 1;
	int croppedHeight = input->height+m_nBLockY - 1;
	int columnStep = croppedWidth / m_nBlockX;
	int rowStep    = croppedHeight / m_nBLockY;

	uchar *p;

	HIST * pHist = (HIST *)blkHist + ithImg * m_nInterval;

	for(int j=0; j< input->height; j+=rowStep)
	{
		for(int i=0; i< input->width; i+=columnStep)
		{

			for(int jj=j; jj< ((j+rowStep)>input->height? input->height:(j+rowStep)); jj++)
			{
				p = (uchar *)input->imageData + jj*input->widthStep + i;

				for(int ii=i; ii<((i+columnStep)>input->width?input->width:(i+columnStep)); ii++)
				{
					pHist->entry[*p]++;
					p++;
				}
			}
			pHist ++;
		}
	}

	pHist = (HIST *)blkHist + ithImg * m_nInterval;

	//for(int j=0;j< m_nBLockY; j++)
	//{
	//	for(int i=0; i< m_nBlockX; i++)
	//	{
	//		for( int k=0; k<256; k++)
	//		{
	//			if(pHist->entry[k]>0)
	//			printf("%d th entry %d \n", k,pHist->entry[k]);
	//		}

	//		pHist++;
	//	}
	//}

}
