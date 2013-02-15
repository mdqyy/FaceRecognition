/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __ML_H__
#define __ML_H__

#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

#include <limits.h>

#ifdef __cplusplus

/****************************************************************************************\
*                               Main struct definitions                                  *
\****************************************************************************************/

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

/* columns of <trainData> matrix are training samples */
#define CV_COL_SAMPLE 0

/* rows of <trainData> matrix are training samples */
#define CV_ROW_SAMPLE 1

#define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

struct CvVectors
{
    int type;
    int dims, count;
    CvVectors* next;
    union
    {
        uchar** ptr;
        float** fl;
        double** db;
    } data;
};

/* Variable type */
#define CV_VAR_NUMERICAL    0
#define CV_VAR_ORDERED      0
#define CV_VAR_CATEGORICAL  1

#define CV_TYPE_NAME_ML_CNN         "opencv-ml-cnn"

class CV_EXPORTS CvStatModel
{
public:
    CvStatModel();
    virtual ~CvStatModel();

    virtual void clear();

    virtual void save( const char* filename, const char* name=0 );
    virtual void load( const char* filename, const char* name=0 );

    virtual void write( CvFileStorage* storage, const char* name );
    virtual void read( CvFileStorage* storage, CvFileNode* node );

protected:
    const char* default_model_name;
};


#if 1
#define CV_STAT_MODEL_MAGIC_VAL 0x77770000 //added by lxts
#define CV_CNN_MAGIC_VAL 0x00008888 //added by lxts
/****************************************************************************************\
*                            Convolutional Neural Network                                *
\****************************************************************************************/
typedef struct CvCNNLayer CvCNNLayer;
typedef struct CvCNNetwork CvCNNetwork;

#define CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY  1
#define CV_CNN_LEARN_RATE_DECREASE_SQRT_INV        2
#define CV_CNN_LEARN_RATE_DECREASE_LOG_INV         3

#define CV_CNN_DELTA_W_INCREASE_FIRSTORDER  0
#define CV_CNN_DELTA_W_INCREASE_LM        1

#define CV_CNN_GRAD_ESTIM_RANDOM        0
#define CV_CNN_GRAD_ESTIM_BY_WORST_IMG  1

#define ICV_CNN_LAYER                0x55550000
#define ICV_CNN_CONVOLUTION_LAYER    0x00001111
#define ICV_CNN_SUBSAMPLING_LAYER    0x00002222
#define ICV_CNN_FULLCONNECT_LAYER    0x00003333

#define CV_IS_CNN(cnn)                                                     \
	( (cnn)!=NULL )

#define ICV_IS_CNN_LAYER( layer )                                          \
    ( ((layer) != NULL) && ((((CvCNNLayer*)(layer))->flags & CV_MAGIC_MASK)\
        == ICV_CNN_LAYER ))

#define ICV_IS_CNN_CONVOLUTION_LAYER( layer )                              \
    ( (ICV_IS_CNN_LAYER( layer )) && (((CvCNNLayer*) (layer))->flags       \
        & ~CV_MAGIC_MASK) == ICV_CNN_CONVOLUTION_LAYER )

#define ICV_IS_CNN_SUBSAMPLING_LAYER( layer )                              \
    ( (ICV_IS_CNN_LAYER( layer )) && (((CvCNNLayer*) (layer))->flags       \
        & ~CV_MAGIC_MASK) == ICV_CNN_SUBSAMPLING_LAYER )

#define ICV_IS_CNN_FULLCONNECT_LAYER( layer )                              \
    ( (ICV_IS_CNN_LAYER( layer )) && (((CvCNNLayer*) (layer))->flags       \
        & ~CV_MAGIC_MASK) == ICV_CNN_FULLCONNECT_LAYER )

typedef void (CV_CDECL *CvCNNLayerForward)
    ( CvCNNLayer* layer, const CvMat* input, CvMat* output );

typedef void (CV_CDECL *CvCNNLayerBackward)
    ( CvCNNLayer* layer, int t, const CvMat* X, CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2, CvMat* d2E_dX2);

typedef void (CV_CDECL *CvCNNLayerRelease)
    (CvCNNLayer** layer);

typedef void (CV_CDECL *CvCNNetworkAddLayer)
    (CvCNNetwork* network, CvCNNLayer* layer);

typedef void (CV_CDECL *CvCNNetworkRelease)
    (CvCNNetwork** network);

#define CV_CNN_LAYER_FIELDS()           \
    /* Indicator of the layer's type */ \
    int flags;                          \
                                        \
    /* Number of input images */        \
    int n_input_planes;                 \
    /* Height of each input image */    \
    int input_height;                   \
    /* Width of each input image */     \
    int input_width;                    \
                                        \
    /* Number of output images */       \
    int n_output_planes;                \
    /* Height of each output image */   \
    int output_height;                  \
    /* Width of each output image */    \
    int output_width;                   \
                                        \
    /* Learning rate at the first iteration */                      \
    float init_learn_rate;                                          \
    /* Dynamics of learning rate decreasing */                      \
    int learn_rate_decrease_type;                                   \
    /* Dynamics of DELTAw increasing */						        \
    int delta_w_increase_type;									    \
    /* samples used in training */						        \
    int nsamples;									    \
    /* max iterations in training */						        \
    int max_iter;									    \
    /* Trainable weights of the layer (including bias) */           \
    /* i-th row is a set of weights of the i-th output plane */     \
    CvMat* weights;                                                 \
	/*used the follow variable to store the last time diag hessian*/\
	CvMat* hessian_diag;											\
                                                                    \
    CvCNNLayerForward  forward;                                     \
    CvCNNLayerBackward backward;                                    \
    CvCNNLayerRelease  release;                                     \
    /* Pointers to the previous and next layers in the network */   \
    CvCNNLayer* prev_layer;                                         \
    CvCNNLayer* next_layer

typedef struct CvCNNLayer
{
    CV_CNN_LAYER_FIELDS();
}CvCNNLayer;

typedef struct CvCNNConvolutionLayer
{
    CV_CNN_LAYER_FIELDS();
    // Kernel size (height and width) for convolution.
    int K;
    // amplitude of sigmoid activation function
    float a;//for simard method
    // scale parameter of sigmoid activation function
    float s;//for simard method
    // exp2ssumWX = exp(2<s>*(bias+w*(x1+...+x4))), where x1,...x4 are some elements of X
    // - is the vector used in computing of the activation function in backward
    CvMat* exp2ssumWX;//for simard method
    // (x1+x2+x3+x4), where x1,...x4 are some elements of X
    // - is the vector used in computing of the activation function in backward
    CvMat* sumX;//for simard method
    // connections matrix, (i,j)-th element is 1 iff there is a connection between
    // i-th plane of the current layer and j-th plane of the previous layer;
    // (i,j)-th element is equal to 0 otherwise
    CvMat *connect_mask;
    // value of the learning rate for updating weights at the first iteration
}CvCNNConvolutionLayer;

typedef struct CvCNNSubSamplingLayer
{
    CV_CNN_LAYER_FIELDS();
    // ratio between the heights (or widths - ratios are supposed to be equal)
    // of the input and output planes
    int sub_samp_scale;
    // amplitude of sigmoid activation function
    float a;
    // scale parameter of sigmoid activation function
    float s;
    // exp2ssumWX = exp(2<s>*(bias+w*(x1+...+x4))), where x1,...x4 are some elements of X
    // - is the vector used in computing of the activation function in backward
    CvMat* exp2ssumWX;
    // (x1+x2+x3+x4), where x1,...x4 are some elements of X
    // - is the vector used in computing of the activation function in backward
    CvMat* sumX;
}CvCNNSubSamplingLayer;

// Structure of the last layer.
typedef struct CvCNNFullConnectLayer
{
    CV_CNN_LAYER_FIELDS();
    // amplitude of sigmoid activation function
    float a;
    // scale parameter of sigmoid activation function
    float s;
    // exp2ssumWX = exp(2*<s>*(W*X)) - is the vector used in computing of the 
    // activation function and it's derivative by the formulae
    // activ.func. = <a>(exp(2<s>WX)-1)/(exp(2<s>WX)+1) == <a> - 2<a>/(<exp2ssumWX> + 1)
    // (activ.func.)' = 4<a><s>exp(2<s>WX)/(exp(2<s>WX)+1)^2
    CvMat* exp2ssumWX;
}CvCNNFullConnectLayer;

typedef struct CvCNNetwork
{
    int n_layers;
    CvCNNLayer* layers;
    CvCNNetworkAddLayer add_layer;
    CvCNNetworkRelease release;
}CvCNNetwork;

#define CV_STAT_MODEL_PARAM_FIELDS() CvMat* cls_labels   //add by lxts on jun-22-2008

typedef struct CvCNNStatModelParams
{
    CV_STAT_MODEL_PARAM_FIELDS();
    // network must be created by the functions cvCreateCNNetwork and <add_layer>
    CvCNNetwork* network;
    CvMat* etalons;
    // termination criteria
    int max_iter;
    int start_iter;
    int grad_estim_type;
}CvCNNStatModelParams;

// this macro is added by lxts on jun/22/2008
struct CvCNNStatModel;

typedef float (CV_CDECL *CvCNNStatModelPredict)
(const CvCNNStatModel *,const CvMat *,CvMat *,CvMat** output);

typedef void (CV_CDECL *CvCNNStatModelUpdate)
    (CvCNNStatModel *,const CvMat *,int,const CvMat *,const CvCNNStatModelParams *,const CvMat *,const CvMat *,const CvMat *,const CvMat *);
typedef void (CV_CDECL *CvCNNStatModelRelease)
    (CvCNNStatModel **);

#define CV_STAT_MODEL_FIELDS() \
    int flags;                  \
    CvCNNStatModelPredict predict;     \
    CvCNNStatModelUpdate update;                         \
    CvCNNStatModelRelease release

typedef struct CvCNNStatModel
{
    CV_STAT_MODEL_FIELDS();
    CvCNNetwork* network;
    // etalons are allocated as rows, the i-th etalon has label cls_labeles[i]
    CvMat* etalons;
    // classes labels
    CvMat* cls_labels;
}CvCNNStatModel;


CVAPI(CvCNNLayer*) cvCreateCNNConvolutionLayer(
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int K,float a,float s,
    double init_learn_rate, int learn_rate_decrease_type, int delta_w_increase_type,  int nsamples, int max_iter,
    CvMat* connect_mask CV_DEFAULT(0), CvMat* weights CV_DEFAULT(0) );

CVAPI(CvCNNLayer*) cvCreateCNNSubSamplingLayer(
    int n_input_planes, int input_height, int input_width,
    int sub_samp_scale, float a, float s,
    float init_learn_rate, int learn_rate_decrease_type, int delta_w_increase_type,  int nsamples, int max_iter, CvMat* weights CV_DEFAULT(0) );

CVAPI(CvCNNLayer*) cvCreateCNNFullConnectLayer( 
    int n_inputs, int n_outputs, float a, float s,
    float init_learn_rate, int learning_type, int delta_w_increase_type,  int nsamples, int max_iter,CvMat* weights CV_DEFAULT(0) );

CVAPI(CvCNNetwork*) cvCreateCNNetwork( CvCNNLayer* first_layer );

CVAPI(CvCNNStatModel*) cvTrainCNNClassifier(
            const CvMat* train_data, int tflag,
            const CvMat* responses,
            const CvCNNStatModelParams* params,
            const CvMat* CV_DEFAULT(0),
            const CvMat* sample_idx CV_DEFAULT(0),
            const CvMat* CV_DEFAULT(0), const CvMat* CV_DEFAULT(0) );

/****************************************************************************************\
*                               Estimate classifiers algorithms                          *
\****************************************************************************************/
typedef const CvMat* (CV_CDECL *CvStatModelEstimateGetMat)
                    ( const CvStatModel* estimateModel );

typedef int (CV_CDECL *CvStatModelEstimateNextStep)
                    ( CvStatModel* estimateModel );

typedef void (CV_CDECL *CvStatModelEstimateCheckClassifier)
                    ( CvStatModel* estimateModel,
                const CvStatModel* model, 
                const CvMat*       features, 
                      int          sample_t_flag,
                const CvMat*       responses );

typedef void (CV_CDECL *CvStatModelEstimateCheckClassifierEasy)
                    ( CvStatModel* estimateModel,
                const CvStatModel* model );

typedef float (CV_CDECL *CvStatModelEstimateGetCurrentResult)
                    ( const CvStatModel* estimateModel,
                            float*       correlation );

typedef void (CV_CDECL *CvStatModelEstimateReset)
                    ( CvStatModel* estimateModel );

//added by lxts on june-28-2008
CVAPI(CvCNNStatModel*)
cvCreateCNNStatModel(
        int flag, int size,
        CvCNNStatModelRelease release,
		CvCNNStatModelPredict predict,
		CvCNNStatModelUpdate update);
#endif
#endif

CvMat * icvCNNModelPredict_face( const CvCNNStatModel* model,
											  const CvMat* _image,
											  CvMat** output);


#endif /*__ML_H__*/
/* End of file. */
