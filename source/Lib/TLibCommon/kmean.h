#ifndef __KMEANS_H__
#define __KMEANS_H__

//#include <string>
//KMean

#define MAX_NUM_CLUSTERS	300
#define MAX_VECTOR_SIZE		10000
#define AVG_ERROR_THRESHOLD	10
#define MAX_KMEAN_DATA_SIZE 2000

typedef struct KMeanTag
{
	float ** kMeanClusterCenters;
	float ** kMeanClusterCentersBuf;
	float *  kMeanClusterRatios;		
	float ** kMeanDataBuf;
	unsigned char * dataLabel;

	int nKMeanClusters;


}KMeanType;

typedef struct KMeanLGTTag
{
	KMeanType *km1;	//first output of kmeans
	KMeanType km2;	//second output
	KMeanType km3; //final centers
	float **firstInput, **firstInputLastGroup, **secondInput, **finalInput;	//kmeans inputs

}KMeanLGT;




void kMeanInit(KMeanType * KM);
void initKMeanLGT( KMeanLGT *KMLGT, int k1, int k2, int numFirstInput, int numGroups, int numInLastGroup, int nDim, int totalNum);
void initKMeanWithParameters(KMeanType *KM, int dataSize, int vectorSize, int numClusters);
void releaseKMean( KMeanType *KM, int dataSize, int vectorSize, int numClusters);

int kMeanClustering(float ** data, int nPoints, int nDim, int nClusters, KMeanType * KM, bool initWithCenters);


#endif //end header