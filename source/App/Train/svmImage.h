#ifndef _SVMIMAGE_H_
#define _SVMIMAGE_H_

#include "Global.h"
#include "svm_classifer_clean.h"

void initSystem(SVM_GST * gst, svm_classifer_clean<int,double>*svm);
void generateSVMTrainingData(SVM_GST * gst);

void svmTraining(double ** features, int nSample, int featureSize, int * sampleLable, 
				 char * modelFileName);


void trainmodel(char* docfile,char* modelfile );
void test(char *docfile,char* modelfile);
void svmTest(double * feature, int featureSize, float * scores,svm_classifer_clean<int,double> *svm);

#endif
