#ifndef _SVMIMAGE_H_
#define _SVMIMAGE_H_

#include "Global.h"
#include "svm_classifer_clean.h"

void initSystem(SVM_GST * gst,svm_classifer_clean<int,double>*svm);
void svmTest(double * feature, int featureSize, int n,float *score,svm_classifer_clean<int,double> *svm);
void svmTestSamples(SVM_GST * gst,svm_classifer_clean<int,double> *svm);

#endif
