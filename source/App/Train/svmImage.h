#ifndef _SVMIMAGE_H_
#define _SVMIMAGE_H_

#include "Global.h"

void initSystem(SVM_GST * gst);
void generateSVMTrainingData(SVM_GST * gst);

void svmTraining(double ** features, int nSample, int featureSize, int * sampleLable, 
				 char * modelFileName);


void trainmodel(char* docfile,char* modelfile );
void test(char *docfile,char* modelfile);

#endif
