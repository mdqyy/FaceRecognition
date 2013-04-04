#ifndef SVM_CLASSIFIER_CLEAN
#define SVM_CLASSIFIER_CLEAN
extern "C"{
#include "svm_common.h"
}
template<typename label_type, typename feature_type>
class svm_classifer_clean
{
	bool binit;
public:
	MODEL * model;
	~svm_classifer_clean(){free_model(model,1);}
	void svm_init_clean(const char* modelfile,int nVerbLevel=2);
	void svm_classifier_clean(label_type *pLabel,feature_type* pFeature,float *pScore,int feature_length,int num_feature);
};

template<typename label_type,typename feature_type>
void svm_classifer_clean<label_type,feature_type>::svm_init_clean(const char * modlefile,int nVerbLevel=2 )
{
	model = read_binary_model(modlefile);
	binit = true;
}

template<typename label_type, typename feature_type>
void svm_classifer_clean<label_type,feature_type>::svm_classifier_clean(label_type *pLabel,feature_type* pFeature,float *pScore,int feature_length,int num_feature)
{
	double ClassRes;
	feature_type *pTempFeat;
	ClassRes = 0;
	pTempFeat = pFeature;
	if(!binit)
	{
		printf("\n The SVM classifier need to be initialized before use!");
	}

	if(feature_length != model->totwords)
	{
		printf("feature length does't fit\n");
		exit(0);
	}

	for (int i=0; i<feature_length;i++, pTempFeat++)
	{
		ClassRes +=model->lin_weights[i]*((double)(*pTempFeat));
	}
	*pScore =  (ClassRes - model->b);
}
#endif