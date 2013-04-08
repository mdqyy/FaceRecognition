#ifndef _FEATURES_H_
#define _FEATURES_H_

#include "Global.h"

void extractImageFeatures_Type1(unsigned char * image, int R0, int C0, int R1, int C1, int widthStep, 
						  double * feature, int featureSize);

#endif
