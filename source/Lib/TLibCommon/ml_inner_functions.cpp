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
#include "_ml.h"
#include "ml.h"

#include "highgui.h"
#include "cv.h"



CvStatModel::CvStatModel()
{
    default_model_name = "my_stat_model";
}


CvStatModel::~CvStatModel()
{
    clear();
}


void CvStatModel::clear()
{
}


void CvStatModel::save( const char* filename, const char* name )
{
    CvFileStorage* fs = 0;
    
    CV_FUNCNAME( "CvStatModel::save" );

    __CV_BEGIN__;

    CV_CALL( fs = cvOpenFileStorage( filename, 0, CV_STORAGE_WRITE ));
    if( !fs )
        CV_ERROR( CV_StsError, "Could not open the file storage. Check the path and permissions" );

    write( fs, name ? name : default_model_name );

    __CV_END__;

    cvReleaseFileStorage( &fs );
}


void CvStatModel::load( const char* filename, const char* name )
{
    CvFileStorage* fs = 0;
    
    CV_FUNCNAME( "CvStatModel::load" );

    __CV_BEGIN__;

    CvFileNode* model_node = 0;

    CV_CALL( fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ ));
    if( !fs )
        __CV_EXIT__;

    if( name )
        model_node = cvGetFileNodeByName( fs, 0, name );
    else
    {
        CvFileNode* root = cvGetRootFileNode( fs );
        if( root->data.seq->total > 0 )
            model_node = (CvFileNode*)cvGetSeqElem( root->data.seq, 0 );
    }

    read( fs, model_node );

    __CV_END__;

    cvReleaseFileStorage( &fs );
}


void CvStatModel::write( CvFileStorage*, const char* )
{
    OPENCV_ERROR( CV_StsNotImplemented, "CvStatModel::write", "" );
}


void CvStatModel::read( CvFileStorage*, CvFileNode* )
{
    OPENCV_ERROR( CV_StsNotImplemented, "CvStatModel::read", "" );
}


static int CV_CDECL
icvCmpIntegers( const void* a, const void* b )
{
    return *(const int*)a - *(const int*)b;
}


static int CV_CDECL
icvCmpIntegersPtr( const void* _a, const void* _b )
{
    int a = **(const int**)_a;
    int b = **(const int**)_b;
    return (a < b ? -1 : 0)|(a > b);
}


static int icvCmpSparseVecElems( const void* a, const void* b )
{
    return ((CvSparseVecElem32f*)a)->idx - ((CvSparseVecElem32f*)b)->idx;
}


CvMat*
cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates )
{
    CvMat* idx = 0;

    CV_FUNCNAME( "cvPreprocessIndexArray" );

    __CV_BEGIN__;

    int i, idx_total, idx_selected = 0, step, type, prev = INT_MIN, is_sorted = 1;
    uchar* srcb = 0;
    int* srci = 0;
    int* dsti;
    
    if( !CV_IS_MAT(idx_arr) )
        CV_ERROR( CV_StsBadArg, "Invalid index array" );

    if( idx_arr->rows != 1 && idx_arr->cols != 1 )
        CV_ERROR( CV_StsBadSize, "the index array must be 1-dimensional" );

    idx_total = idx_arr->rows + idx_arr->cols - 1;
    srcb = idx_arr->data.ptr;
    srci = idx_arr->data.i;

    type = CV_MAT_TYPE(idx_arr->type);
    step = CV_IS_MAT_CONT(idx_arr->type) ? 1 : idx_arr->step/CV_ELEM_SIZE(type);

    switch( type )
    {
    case CV_8UC1:
    case CV_8SC1:
        // idx_arr is array of 1's and 0's -
        // i.e. it is a mask of the selected components
        if( idx_total != data_arr_size )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Component mask should contain as many elements as the total number of input variables" );
        
        for( i = 0; i < idx_total; i++ )
            idx_selected += srcb[i*step] != 0;

        if( idx_selected == 0 )
            CV_ERROR( CV_StsOutOfRange, "No components/input_variables is selected!" );

        if( idx_selected == idx_total )
            __CV_EXIT__;
        break;
    case CV_32SC1:
        // idx_arr is array of integer indices of selected components
        if( idx_total > data_arr_size )
            CV_ERROR( CV_StsOutOfRange,
            "index array may not contain more elements than the total number of input variables" );
        idx_selected = idx_total;
        // check if sorted already
        for( i = 0; i < idx_total; i++ )
        {
            int val = srci[i*step];
            if( val >= prev )
            {
                is_sorted = 0;
                break;
            }
            prev = val;
        }
        break;
    default:
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported index array data type "
                                           "(it should be 8uC1, 8sC1 or 32sC1)" );
    }

    CV_CALL( idx = cvCreateMat( 1, idx_selected, CV_32SC1 ));
    dsti = idx->data.i;
    
    if( type < CV_32SC1 )
    {
        for( i = 0; i < idx_total; i++ )
            if( srcb[i*step] )
                *dsti++ = i;
    }
    else
    {
        for( i = 0; i < idx_total; i++ )
            dsti[i] = srci[i*step];
        
        if( !is_sorted )
            qsort( dsti, idx_total, sizeof(dsti[0]), icvCmpIntegers );
        
        if( dsti[0] < 0 || dsti[idx_total-1] >= data_arr_size )
            CV_ERROR( CV_StsOutOfRange, "the index array elements are out of range" );

        if( check_for_duplicates )
        {
            for( i = 1; i < idx_total; i++ )
                if( dsti[i] <= dsti[i-1] )
                    CV_ERROR( CV_StsBadArg, "There are duplicated index array elements" );
        }
    }

    __CV_END__;

    if( cvGetErrStatus() < 0 )
        cvReleaseMat( &idx );

    return idx;
}

CvMat*
cvPreprocessOrderedResponses( const CvMat* responses, const CvMat* sample_idx, int sample_all )
{
    CvMat* out_responses = 0;

    CV_FUNCNAME( "cvPreprocessOrderedResponses" );

    __CV_BEGIN__;

    int i, r_type, r_step;
    const int* map = 0;
    float* dst;
    int sample_count = sample_all;

    if( !CV_IS_MAT(responses) )
        CV_ERROR( CV_StsBadArg, "Invalid response array" );

    if( responses->rows != 1 && responses->cols != 1 )
        CV_ERROR( CV_StsBadSize, "Response array must be 1-dimensional" );

    if( responses->rows + responses->cols - 1 != sample_count )
        CV_ERROR( CV_StsUnmatchedSizes,
        "Response array must contain as many elements as the total number of samples" );

    r_type = CV_MAT_TYPE(responses->type);
    if( r_type != CV_32FC1 && r_type != CV_32SC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported response type" );

    r_step = responses->step ? responses->step / CV_ELEM_SIZE(responses->type) : 1;

    if( r_type == CV_32FC1 && CV_IS_MAT_CONT(responses->type) && !sample_idx )
    {
        out_responses = (CvMat*)responses;
        __CV_EXIT__;
    }

    if( sample_idx )
    {
        if( !CV_IS_MAT(sample_idx) || CV_MAT_TYPE(sample_idx->type) != CV_32SC1 ||
            sample_idx->rows != 1 && sample_idx->cols != 1 || !CV_IS_MAT_CONT(sample_idx->type) )
            CV_ERROR( CV_StsBadArg, "sample index array should be continuous 1-dimensional integer vector" );
        if( sample_idx->rows + sample_idx->cols - 1 > sample_count )
            CV_ERROR( CV_StsBadSize, "sample index array is too large" );
        map = sample_idx->data.i;
        sample_count = sample_idx->rows + sample_idx->cols - 1;
    }

    CV_CALL( out_responses = cvCreateMat( 1, sample_count, CV_32FC1 ));
    
    dst = out_responses->data.fl;
    if( r_type == CV_32FC1 )
    {
        const float* src = responses->data.fl;
        for( i = 0; i < sample_count; i++ )
        {
            int idx = map ? map[i] : i;
            assert( (unsigned)idx < (unsigned)sample_all );
            dst[i] = src[idx*r_step];
        }
    }
    else
    {
        const int* src = responses->data.i;
        for( i = 0; i < sample_count; i++ )
        {
            int idx = map ? map[i] : i;
            assert( (unsigned)idx < (unsigned)sample_all );
            dst[i] = (float)src[idx*r_step];
        }
    }

    __CV_END__;

    return out_responses;
}

CvMat*
cvPreprocessCategoricalResponses( const CvMat* responses,
    const CvMat* sample_idx, int sample_all,
    CvMat** out_response_map, CvMat** class_counts )
{
    CvMat* out_responses;// = 0;
    int** response_ptr = 0;
    CV_FUNCNAME( "cvPreprocessCategoricalResponses" );

    if( out_response_map )
        *out_response_map = 0;

    if( class_counts )
        *class_counts = 0;

    __CV_BEGIN__;


    int i, r_type, r_step;
    int cls_count = 1, prev_cls, prev_i;
    const int* map = 0;
    const int* srci;
    const float* srcfl;
    int* dst;
    int* cls_map;
    int* cls_counts = 0;
    int sample_count = sample_all;

    if( !CV_IS_MAT(responses) )
        CV_ERROR( CV_StsBadArg, "Invalid response array" );

    if( responses->rows != 1 && responses->cols != 1 )
        CV_ERROR( CV_StsBadSize, "Response array must be 1-dimensional" );

    if( responses->rows + responses->cols - 1 != sample_count )
        CV_ERROR( CV_StsUnmatchedSizes,
        "Response array must contain as many elements as the total number of samples" );

    r_type = CV_MAT_TYPE(responses->type);
    if( r_type != CV_32FC1 && r_type != CV_32SC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported response type" );

    r_step = responses->step ? responses->step / CV_ELEM_SIZE(responses->type) : 1;

    if( sample_idx )
    {
        if( !CV_IS_MAT(sample_idx) || CV_MAT_TYPE(sample_idx->type) != CV_32SC1 ||
            sample_idx->rows != 1 && sample_idx->cols != 1 || !CV_IS_MAT_CONT(sample_idx->type) )
            CV_ERROR( CV_StsBadArg, "sample index array should be continuous 1-dimensional integer vector" );
        if( sample_idx->rows + sample_idx->cols - 1 > sample_count )
            CV_ERROR( CV_StsBadSize, "sample index array is too large" );
        map = sample_idx->data.i;
        sample_count = sample_idx->rows + sample_idx->cols - 1;
    }

    CV_CALL( out_responses = cvCreateMat( 1, sample_count, CV_32SC1 ));

    if( !out_response_map )
        CV_ERROR( CV_StsNullPtr, "out_response_map pointer is NULL" );

    CV_CALL( response_ptr = (int**)cvAlloc( sample_count*sizeof(response_ptr[0])));

    srci = responses->data.i;
    srcfl = responses->data.fl;
    dst = out_responses->data.i;

    for( i = 0; i < sample_count; i++ )
    {
        int idx = map ? map[i] : i;
        assert( (unsigned)idx < (unsigned)sample_all );
        if( r_type == CV_32SC1 )
            dst[i] = srci[idx*r_step];
        else
        {
            float rf = srcfl[idx*r_step];
            int ri = cvRound(rf);
            if( ri != rf )
            {
                char buf[100];
                sprintf( buf, "response #%d is not integral", idx );
                CV_ERROR( CV_StsBadArg, buf );
            }
            dst[i] = ri;
        }
        response_ptr[i] = dst + i;
    }

    qsort( response_ptr, sample_count, sizeof(int*), icvCmpIntegersPtr );

    // count the classes
    for( i = 1; i < sample_count; i++ )
        cls_count += *response_ptr[i] != *response_ptr[i-1];

    if( cls_count < 2 )
        CV_ERROR( CV_StsBadArg, "There is only a single class" );

    CV_CALL( *out_response_map = cvCreateMat( 1, cls_count, CV_32SC1 ));

    if( class_counts )
    {
        CV_CALL( *class_counts = cvCreateMat( 1, cls_count, CV_32SC1 ));
        cls_counts = (*class_counts)->data.i;
    }

    // compact the class indices and build the map
    prev_cls = ~*response_ptr[0];
    cls_count = -1;
    cls_map = (*out_response_map)->data.i;

    for( i = 0, prev_i = -1; i < sample_count; i++ )
    {
        int cur_cls = *response_ptr[i];
        if( cur_cls != prev_cls )
        {
            if( cls_counts && cls_count >= 0 )
                cls_counts[cls_count] = i - prev_i;
            cls_map[++cls_count] = prev_cls = cur_cls;
            prev_i = i;
        }
        *response_ptr[i] = cls_count;
    }
    
    if( cls_counts )
        cls_counts[cls_count] = i - prev_i;

    __CV_END__;

    cvFree( &response_ptr );

    return out_responses;
}

void
cvCheckTrainData( const CvMat* train_data, int tflag,
                  const CvMat* missing_mask, 
                  int* var_all, int* sample_all )
{
    CV_FUNCNAME( "cvCheckTrainData" );

    if( var_all )
        *var_all = 0;

    if( sample_all )
        *sample_all = 0;

    __CV_BEGIN__;

    // check parameter types and sizes
    if( !CV_IS_MAT(train_data) || CV_MAT_TYPE(train_data->type) != CV_32FC1 )
        CV_ERROR( CV_StsBadArg, "train data must be floating-point matrix" );

    if( missing_mask )
    {
        if( !CV_IS_MAT(missing_mask) || !CV_IS_MASK_ARR(missing_mask) ||
            !CV_ARE_SIZES_EQ(train_data, missing_mask) )
            CV_ERROR( CV_StsBadArg,
            "missing value mask must be 8-bit matrix of the same size as training data" );
    }

    if( tflag != CV_ROW_SAMPLE && tflag != CV_COL_SAMPLE )
        CV_ERROR( CV_StsBadArg,
        "Unknown training data layout (must be CV_ROW_SAMPLE or CV_COL_SAMPLE)" );

    if( var_all )
        *var_all = tflag == CV_ROW_SAMPLE ? train_data->cols : train_data->rows;
    
    if( sample_all )
        *sample_all = tflag == CV_ROW_SAMPLE ? train_data->rows : train_data->cols;

    __CV_END__;
}


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
                    CvMat** out_sample_idx )
{
    int ok = 0; 
    CvMat* _var_idx = 0;
    CvMat* _sample_idx = 0;
    CvMat* _responses = 0;
    int sample_all = 0, sample_count = 0, var_all = 0, var_count = 0;
    CV_FUNCNAME( "cvPrepareTrainData" );

    // step 0. clear all the output pointers to ensure we do not try
    // to call free() with uninitialized pointers
    if( out_responses )
        *out_responses = 0;

    if( out_response_map )
        *out_response_map = 0;

    if( out_var_idx )
        *out_var_idx = 0;

    if( out_sample_idx )
        *out_sample_idx = 0;

    if( out_train_samples )
        *out_train_samples = 0;

    if( _sample_count )
        *_sample_count = 0;

    if( _var_count )
        *_var_count = 0;

    if( _var_all )
        *_var_all = 0;
    
    __CV_BEGIN__;

	int i=0, k, *i_responses;float *f_responses;//added for FILE log;

    if( !out_train_samples )
        CV_ERROR( CV_StsBadArg, "output pointer to train samples is NULL" );

    CV_CALL( cvCheckTrainData( train_data, tflag, 0, &var_all, &sample_all ));

    if( sample_idx )
        CV_CALL( _sample_idx = cvPreprocessIndexArray( sample_idx, sample_all ));
    if( var_idx )
        CV_CALL( _var_idx = cvPreprocessIndexArray( var_idx, var_all ));

    if( responses )
    {
        if( !out_responses )
            CV_ERROR( CV_StsNullPtr, "output response pointer is NULL" );
        
        if( response_type == CV_VAR_NUMERICAL )
        {
            CV_CALL( _responses = cvPreprocessOrderedResponses( responses,
                                                _sample_idx, sample_all ));
        }
        else
        {
            CV_CALL( _responses = cvPreprocessCategoricalResponses( responses,
                                _sample_idx, sample_all, out_response_map, 0 ));
        }
    }

	CV_CALL( *out_train_samples =
                cvGetTrainSamples( train_data, tflag, _var_idx, _sample_idx,
                                   &var_count, &sample_count, always_copy_data )); 

    ok = 1;

    __CV_END__;

    if( ok )
    {
        if( out_responses )
            *out_responses = _responses, _responses = 0;

        if( out_var_idx )
            *out_var_idx = _var_idx, _var_idx = 0;

        if( out_sample_idx )
            *out_sample_idx = _sample_idx, _sample_idx = 0;

        if( _sample_count )
            *_sample_count = sample_count;

        if( _var_count )
            *_var_count = var_count;

        if( _var_all )
            *_var_all = var_all;
    }
    else
    {
        if( out_response_map )
            cvReleaseMat( out_response_map );
        cvFree( out_train_samples );
    }

    if( _responses != responses )
        cvReleaseMat( &_responses );
    cvReleaseMat( &_var_idx );
    cvReleaseMat( &_sample_idx );

    return ok;
}

void
cvPreparePredictData( const CvArr* _sample, int dims_all,
                      const CvMat* comp_idx, int class_count,
                      const CvMat* prob, float** _row_sample,
                      int as_sparse )
{
    float* row_sample = 0;
    int* inverse_comp_idx = 0;
    
    CV_FUNCNAME( "cvPreparePredictData" );

    __CV_BEGIN__;

    const CvMat* sample = (const CvMat*)_sample;
    float* sample_data;
    int sample_step;
    int is_sparse = CV_IS_SPARSE_MAT(sample);
    int d, sizes[CV_MAX_DIM];
    int i, dims_selected;
    int vec_size;

    if( !is_sparse && !CV_IS_MAT(sample) )
        CV_ERROR( !sample ? CV_StsNullPtr : CV_StsBadArg, "The sample is not a valid vector" );

    if( cvGetElemType( sample ) != CV_32FC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Input sample must have 32fC1 type" );

    CV_CALL( d = cvGetDims( sample, sizes ));

    if( !(is_sparse && d == 1 || !is_sparse && d == 2 && (sample->rows == 1 || sample->cols == 1)) )
        CV_ERROR( CV_StsBadSize, "Input sample must be 1-dimensional vector" );
    
    if( d == 1 )
        sizes[1] = 1;

    if( sizes[0] + sizes[1] - 1 != dims_all )
        CV_ERROR( CV_StsUnmatchedSizes,
        "The sample size is different from what has been used for training" );

    if( !_row_sample )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: The row_sample pointer is NULL" );

    if( comp_idx && (!CV_IS_MAT(comp_idx) || comp_idx->rows != 1 ||
        CV_MAT_TYPE(comp_idx->type) != CV_32SC1) )
        CV_ERROR( CV_StsBadArg, "INTERNAL ERROR: invalid comp_idx" );

    dims_selected = comp_idx ? comp_idx->cols : dims_all;
    
    if( prob )
    {
        if( !CV_IS_MAT(prob) )
            CV_ERROR( CV_StsBadArg, "The output matrix of probabilities is invalid" );

        if( (prob->rows != 1 && prob->cols != 1) ||
            CV_MAT_TYPE(prob->type) != CV_32FC1 &&
            CV_MAT_TYPE(prob->type) != CV_64FC1 )
            CV_ERROR( CV_StsBadSize,
            "The matrix of probabilities must be 1-dimensional vector of 32fC1 type" );

        if( prob->rows + prob->cols - 1 != class_count )
            CV_ERROR( CV_StsUnmatchedSizes,
            "The vector of probabilities must contain as many elements as "
            "the number of classes in the training set" );
    }

    vec_size = !as_sparse ? dims_selected*sizeof(row_sample[0]) :
                (dims_selected + 1)*sizeof(CvSparseVecElem32f);

    if( CV_IS_MAT(sample) )
    {
        sample_data = sample->data.fl;
        sample_step = sample->step / sizeof(row_sample[0]);

        if( !comp_idx && sample_step <= 1 && !as_sparse )
            *_row_sample = sample_data;
        else
        {
            CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

            if( !comp_idx )
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*i];
            else
            {
                int* comp = comp_idx->data.i;
                if( !sample_step )
                    for( i = 0; i < dims_selected; i++ )
                        row_sample[i] = sample_data[comp[i]];
                else
                    for( i = 0; i < dims_selected; i++ )
                        row_sample[i] = sample_data[sample_step*comp[i]];
            }

            *_row_sample = row_sample;
        }

        if( as_sparse )
        {
            const float* src = (const float*)row_sample;
            CvSparseVecElem32f* dst = (CvSparseVecElem32f*)row_sample;

            dst[dims_selected].idx = -1;
            for( i = dims_selected - 1; i >= 0; i-- )
            {
                dst[i].idx = i;
                dst[i].val = src[i];
            }
        }
    }
    else
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        const CvSparseMat* sparse = (const CvSparseMat*)sample;
        assert( is_sparse );

        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

        if( comp_idx )
        {
            CV_CALL( inverse_comp_idx = (int*)cvAlloc( dims_all*sizeof(int) ));
            memset( inverse_comp_idx, -1, dims_all*sizeof(int) );
            for( i = 0; i < dims_selected; i++ )
                inverse_comp_idx[comp_idx->data.i[i]] = i;
        }
        
        if( !as_sparse )
        {
            memset( row_sample, 0, vec_size );

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                row_sample[idx] = *(float*)CV_NODE_VAL( sparse, node );
            }
        }
        else
        {
            CvSparseVecElem32f* ptr = (CvSparseVecElem32f*)row_sample;
            
            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                ptr->idx = idx;
                ptr->val = *(float*)CV_NODE_VAL( sparse, node );
                ptr++;
            }

            qsort( row_sample, ptr - (CvSparseVecElem32f*)row_sample,
                   sizeof(ptr[0]), icvCmpSparseVecElems );
            ptr->idx = -1;
        }

        *_row_sample = row_sample;
    }

    __CV_END__;

    if( inverse_comp_idx )
        cvFree( &inverse_comp_idx );

    if( cvGetErrStatus() < 0 && _row_sample )
    {
        cvFree( &row_sample );
        *_row_sample = 0;
    }
}
