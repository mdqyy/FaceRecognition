#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include "Define.h"

typedef struct GST_Tag
{
	char * path;
	char * imageListFileName;

	int nClasses;
	int nSamples;
	int featureSize;

	double ** features;
	double ** featuresNew;

	int * sampleLable;
	int * classLable;	

}SVM_GST;


#endif
