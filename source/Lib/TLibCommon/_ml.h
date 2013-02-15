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

#ifndef __ML_INTERNAL_H__
#define __ML_INTERNAL_H__

#include "ml.h"
#include "cxmisc.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#define ML_IMPL CV_IMPL

#define ICV_IS_MAT_OF_TYPE( mat, mat_type) \
    (CV_IS_MAT( mat ) && CV_MAT_TYPE( mat->type ) == (mat_type) &&   \
    (mat)->cols > 0 && (mat)->rows > 0)

/****************************************************************************************\
*                       Auxiliary functions declarations                                 *
\****************************************************************************************/

typedef struct CvSparseVecElem32f
{
    int idx;
    float val;
}
CvSparseVecElem32f;

int
cvPrepareTrainData( const char* /*funcname*/,
                    const CvMat* train_data, int tflag,
                    const CvMat* responses, int response_type,
                    const CvMat* var_idx,
                    const CvMat* sample_idx,
                    bool always_copy_data,
                    const float*** out_train_samples,
                    int* _sample_count,
                    int* _var_count,
                    int* _var_all,
                    CvMat** out_responses,
                    CvMat** out_response_map,
                    CvMat** out_var_idx,
                    CvMat** out_sample_idx=0 );

void cvPreparePredictData( const CvArr* sample, int dims_all, const CvMat* comp_idx,
                      int class_count, const CvMat* prob, float** row_sample,
                      int as_sparse CV_DEFAULT(0) );

void cvCheckTrainData( const CvMat* train_data, int tflag,
                       const CvMat* missing_mask, 
                       int* var_all, int* sample_all );

CvMat* cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates=false );

CvMat* cvPreprocessOrderedResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all );

CvMat* cvPreprocessCategoricalResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all,
                CvMat** out_response_map, CvMat** class_counts=0 );

const float** cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data=false );



#endif /* __ML_H__ */
