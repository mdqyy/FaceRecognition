#ifndef __KMEANS_H__
#define __KMEANS_H__

//KMean

#define MAX_NUM_CLUSTERS	80
#define MAX_VECTOR_SIZE		40
#define AVG_ERROR_THRESHOLD	1
#define MAX_KMEAN_DATA_SIZE 1200

typedef struct KMeanTag
{
	int ** kMeanClusterCenters;
	int ** kMeanClusterCentersBuf;
	int *  kMeanClusterRatios;		
	int ** kMeanDataBuf;
	unsigned char * dataLabel;

	int nKMeanClusters;


}KMeanType;



void kMeanInit(KMeanType * KM);

int kMeanClustering(int ** data, int nPoints, int nDim, int nClusters, KMeanType * KM);


#endif //end header