#include "kmean.h"
//KMean



void kMeanInit(KMeanType * KM)
{
	int n, nKMeanClusters;

	KM->kMeanDataBuf = (int **)malloc(MAX_KMEAN_DATA_SIZE * sizeof(int *));
	for(n=0;n<MAX_KMEAN_DATA_SIZE; n++)
		KM->kMeanDataBuf[n] = (int *)malloc(MAX_VECTOR_SIZE * sizeof(int));

	KM->kMeanClusterCenters = (int **)malloc(MAX_NUM_CLUSTERS * sizeof(int *));
	for(n=0; n<MAX_NUM_CLUSTERS; n++)
		KM->kMeanClusterCenters[n] = (int *)malloc(MAX_VECTOR_SIZE * sizeof(int));

	KM->kMeanClusterCentersBuf = (int **)malloc(MAX_NUM_CLUSTERS * sizeof(int *));
	for(n=0; n<MAX_NUM_CLUSTERS; n++)
		KM->kMeanClusterCentersBuf[n] = (int *)malloc(MAX_VECTOR_SIZE * sizeof(int));

	KM->kMeanClusterRatios = (int *)malloc(MAX_NUM_CLUSTERS * sizeof(int));

	KM->dataLabel = (unsigned char *)malloc(MAX_KMEAN_DATA_SIZE);

	KM->nKMeanClusters = 3;
	if(nKMeanClusters > MAX_NUM_CLUSTERS) nKMeanClusters = MAX_NUM_CLUSTERS;

}



int kMeanClustering(int ** data, int nPoints, int nDim, int nClusters, KMeanType * KM)
{
	int stepLen, threshold;
	int n, N0, N1, k, i, nIterations;
	int avg, tmp, diff, minDist, minPtr, ct, dist, contFlag;
	int ** tmpBuf;
	int flag;
	int ** kMeanClusterCenters;
	int ** kMeanClusterCentersBuf;
	int *  kMeanClusterRatios;	
	unsigned char * dataLabel;

	if (nPoints < (4 * nClusters)) return -1;

	kMeanClusterCenters = KM->kMeanClusterCenters;
	kMeanClusterCentersBuf = KM->kMeanClusterCentersBuf;
	kMeanClusterRatios = KM->kMeanClusterRatios;
	dataLabel = KM->dataLabel;

	//initial clusters;
	stepLen = nPoints / nClusters;

	for(n=0; n<nClusters; n++)
	{
		N0 = n * stepLen;
		N1 = (n+1) * stepLen;

		if(n == (nClusters-1))	N1 = nPoints;

		if((N1 - N0) < 1) N1 = N0 + 1;

		for(i=0; i<nDim; i++)
		{	
			avg = 0;
			for(k=N0; k<N1; k++)
				avg = avg + data[k][i];

			avg = avg / (N1 - N0);
			kMeanClusterCenters[n][i] = avg;
		}
	}

	contFlag = 1;
	nIterations = 0;

	while(contFlag == 1)
	{
		for(k=0;k<nClusters; k++)
		for(i=0; i<nDim; i++)
			kMeanClusterCentersBuf[k][i] = 0;

		for(k=0; k<nClusters; k++)
			kMeanClusterRatios[k] = 0;

		for(n=0; n<nPoints; n++)
		{
			minDist = 1000000000;
			for(k=0; k<nClusters; k++)
			{
				//find the distance to each cluster center
				tmp = 0;
				for(i=0; i<nDim; i++)
				{
					diff = data[n][i] - kMeanClusterCenters[k][i];
					tmp = tmp + diff * diff;
				}

				if(tmp < minDist)
				{
					minDist = tmp;
					minPtr = k;
				}
			}

			//assign the label
			dataLabel[n] = minPtr;

			//assign this point to the minPtr cluster
			for(i=0;i<nDim; i++)
				kMeanClusterCentersBuf[minPtr][i] = kMeanClusterCentersBuf[minPtr][i] + data[n][i];

			kMeanClusterRatios[minPtr] = kMeanClusterRatios[minPtr] + 1;
		}

		//now update the new cluster center
		dist = 0;

		for(k=0; k<nClusters; k++)
		{
			ct = kMeanClusterRatios[k];

			if(ct > 2)
			{
				for(i=0;i<nDim; i++)
					kMeanClusterCentersBuf[k][i] = kMeanClusterCentersBuf[k][i] / ct;
			}
			else
			{
				//use the old one
				kMeanClusterCentersBuf[k][i] = kMeanClusterCenters[k][i];
			}

			for(i=0; i<nDim; i++)
			{
				diff = kMeanClusterCenters[k][i] - kMeanClusterCentersBuf[k][i];
				if(diff < 0) diff = -diff;

				dist = dist + diff;
			}

		}

		//decide to loop again or not
		threshold = nClusters * nDim * AVG_ERROR_THRESHOLD;
		if((nIterations > 20) || (dist < threshold)) 
			contFlag = 0;

		//switch the center buffers
		tmpBuf = kMeanClusterCenters;
		kMeanClusterCenters = kMeanClusterCentersBuf;
		kMeanClusterCentersBuf = tmpBuf;

		nIterations++;
	}

	//output the percentage
	for(k=0; k<nClusters; k++)
		kMeanClusterRatios[k] = (100 * kMeanClusterRatios[k])/nPoints;

	return 1;
}
//END: K-Mean Clustering-----------------------------------------------------------------

//Example code

#if 0
void determineObjectColor(unsigned char * frame, int widthStep, unsigned char * fgMask, int maskWidth, 
						  int X0, int Y0, int X1, int Y1, int pixelTN, KMeanType * KM)
{
	int r, c, step, ct, n, thr;
	unsigned char * ptrFrame;
	int ** kMeanClusterCenters;
	int *  kMeanClusterRatios;	
	int ** kMeanDataBuf;

	kMeanClusterCenters = KM->kMeanClusterCenters;
	kMeanClusterRatios = KM->kMeanClusterRatios;
	kMeanDataBuf = KM->kMeanDataBuf;

	thr = MAX_KMEAN_DATA_SIZE - 2;
	step = (pixelTN * 4) / CURR_KMEAN_TN + 1;
	ct = 0;
	n = 0;
	for(r=Y0; r<Y1; r++)
	for(c=X0; c<X1; c++)
	{
		if(*(fgMask + (r>>BLK_SHIFT) * maskWidth + (c>>BLK_SHIFT)) == 1)
		{
			ct++;
			if(ct == step)
			{
				//sample the pixel
				ptrFrame = frame + r * widthStep + c * 3;
				kMeanDataBuf[n][0] = ptrFrame[0];
				kMeanDataBuf[n][1] = ptrFrame[1];
				kMeanDataBuf[n][2] = ptrFrame[2];
				n++;

				if(n > thr) n = thr;			//now overflow

				ct = 0;
			}

		}
	}

	//do k-mean clustering
	kMeanClustering(kMeanDataBuf, n, 3, 3, KM);

}

#endif