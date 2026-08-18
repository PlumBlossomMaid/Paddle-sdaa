// Copyright (c) 2024 Tecorigin Co., Ltd. All rights reserved.
//
// NOTICE TO LICENSE:
// This source code and/or documentation ("Licensed Deliverables") are subject
// to TECORIGIN intellectual property rights under CHINA and
// international Copyright laws.
//
// These Licensed Deliverables contained herein is PROPRIETARY and CONFIDENTIAL
// to TECORIGIN and is being provided under the terms and conditions of a
// form of TECORIGIN software license agreement by and between TECORIGIN and
// Licensee ("License Agreement") or electronically accepted by Licensee.
//
// Notwithstanding any terms or conditions to the contrary in the License
// Agreement, reproduction or disclosure of the Licensed Deliverables to any
// third party without the express written consent of TECORIGIN is prohibited.
//
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, TECORIGIN MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE
// LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT
// EXPRESS OR IMPLIED WARRANTY OF ANY KIND. TECORIGIN DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES
// OF MERCHANTABILITY,NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, IN NO EVENT SHALL TECORIGIN BE LIABLE FOR ANY SPECIAL, INDIRECT,
// INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
// FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
// NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH
// THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.

#ifndef CUSTOM_INCLUDE_TECOCUSTOM_EXT_H_
#define CUSTOM_INCLUDE_TECOCUSTOM_EXT_H_

#include "./tecocustom.h"
tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50Backward(tecocustomHandle_t handle,
                                 const void *dev_in,                 // N*224*224*3*sizoef(half)
                                 const void *dev_dout,               // N*1000*sizoef(half)
                                 const void *dev_weight,             // 51,008,000B(half)
                                 void *dev_dweight,                  // 51,008,000B(half)
                                 const void *dev_bias_scale,         // 212,480B(float)
                                 void *dev_dbias_dscale,             // 212,480B(float)
                                 const void *dev_saved_mean_invvar,  // 212,480B(float)
                                 void *dev_reservespace, const size_t size_dev_reservespace,
                                 void *dev_workspace, const size_t size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50ForwardInference(tecocustomHandle_t handle,
                                         const void *dev_in,          // N*224*224*3*sizoef(half)
                                         void *dev_out,               // N*1000*sizoef(half)
                                         const void *dev_weight,      // 51,008,000B(half)
                                         const void *dev_bias_scale,  // 212,480B(float)
                                         const void *dev_estimated_mean_var,  // 212,480B(float)
                                         double epsilon,
                                         void *dev_workspace,  // 2.5GB
                                         const size_t size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50ForwardTraining(tecocustomHandle_t handle,
                                        const void *dev_in,          // N*224*224*3*sizoef(half)
                                        void *dev_out,               // N*1000*sizoef(half)
                                        const void *dev_weight,      // 51,008,000B(half)
                                        const void *dev_bias_scale,  // 212,480B(float)
                                        void *dev_running_mean_var,  // 212,480B(float)
                                        double epsilon, double exponentialAverageFactor,
                                        void *dev_saved_mean_invvar,  // 212,480B(float)
                                        void *dev_reservespace, const size_t size_dev_reservespace,
                                        void *dev_workspace, const size_t size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50GetBackwardWorkSpaceSize(size_t *size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50GetBackwardWorkSpaceSize(size_t *size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50GetForwardInferenceWorkSpaceSize(size_t *size_dev_workspace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50GetTrainingReserveSpaceSize(size_t *size_dev_reservespace);

tecocustomStatus_t TECOCUSTOMWINAPI
tecocustomCustomResnet50GetForwardTrainingWorkSpaceSize(size_t *size_dev_workspace);

/*
 * residual_block Start
 */

#define TECOCUSTOM_MAX_RESIDUAL_CONV_NUM 3
typedef enum {
    TECOCUSTOM_RESIDUAL_BRANCH = 0,
    TECOCUSTOM_SHORTCUT_BRANCH = 1
} tecocustomResidualBlockBranchLabel_t;

typedef enum {
    TECOCUSTOM_RESIDUAL_IDENTITY = 0,
    TECOCUSTOM_RESIDUAL_PROJECTION = 1,
    TECOCUSTOM_RESIDUAL_ADD_Z = 2
} tecocustomResidualBlockMode_t;

typedef enum {
    // data
    TECOCUSTOM_PARAM_XDESC = 0,
    TECOCUSTOM_PARAM_YDESC = 1,
    TECOCUSTOM_PARAM_ZDESC = 2,
    // conv
    TECOCUSTOM_PARAM_WDESC = 3,
    TECOCUSTOM_PARAM_CONV_DESC = 4,
    // act
    TECOCUSTOM_PARAM_ACTIVATION_DESC = 5,
    // bias
    TECOCUSTOM_PARAM_BIAS_DESC = 6,

    // DATA POINT
    TECOCUSTOM_PARAM_W_DATA_PTR = 100,
    TECOCUSTOM_PARAM_BIAS_DATA_PTR = 101,
    TECOCUSTOM_PARAM_Z_DATA_PTR = 102
} tecocustomResidualBlockParamLabel_t;

typedef struct tecocustomResidualBlockForwardParamsPackStruct
    *tecocustomResidualBlockForwardParamsPack_t;

tecocustomStatus_t TECOCUSTOMWINAPI tecocustomCreateResidualBlockForwardParamsStruct(
    tecocustomResidualBlockForwardParamsPack_t *residual_block_params,
    tecocustomResidualBlockMode_t *residual_mode, int *residual_conv_num);

tecocustomStatus_t TECOCUSTOMWINAPI tecocustomDestroyResidualBlockForwardParamsStruct(
    tecocustomResidualBlockForwardParamsPack_t *residual_block_params);

tecocustomStatus_t TECOCUSTOMWINAPI tecocustomSetResidualBlockForwardParams(
    tecocustomResidualBlockForwardParamsPack_t residual_block_params,
    const tecocustomResidualBlockBranchLabel_t branch, const int residual_conv_index,
    const tecocustomResidualBlockParamLabel_t param_label, void *ptr);

tecocustomStatus_t TECOCUSTOMWINAPI tecocustomGetResidualBlockForwardWorkspaceSize(
    const tecocustomResidualBlockForwardParamsPack_t residual_block_params, size_t *workspace_size);

tecocustomStatus_t TECOCUSTOMWINAPI tecocustomResidualBlockForward(
    tecocustomHandle_t handle, const tecocustomTensorDescriptor_t xDesc, const void *x,
    const float alpha, const float beta,
    const tecocustomResidualBlockForwardParamsPack_t residual_block_params,
    const tecocustomActivationDescriptor_t actDesc, void *workSpace, size_t workSpace_size,
    const tecocustomTensorDescriptor_t yDesc, void *y);
/*
 * residual_block End
 */

/**
 * @brief Processes a single step in sequence generation with block-based processing for encoder-decoder architectures
 *
 * @param handle Handle to the Tecocustom library context
 * @param stopFlagsDesc Descriptor for flags indicating whether generation should stop for each sequence
 * @param stopFlags Pointer to stop flags data
 * @param seqLensThisTimeDesc Descriptor for current sequence lengths in this step
 * @param seqLensThisTime Pointer to current sequence lengths data
 * @param oriSeqLensEncoderDesc Descriptor for original encoder sequence lengths
 * @param oriSeqLensEncoder Pointer to original encoder sequence lengths data
 * @param seqLensEncoderDesc Descriptor for current encoder sequence lengths
 * @param seqLensEncoder Pointer to current encoder sequence lengths data
 * @param seqLensDecoderDesc Descriptor for decoder sequence lengths
 * @param seqLensDecoder Pointer to decoder sequence lengths data
 * @param blockTablesDesc Descriptor for block allocation tables
 * @param blockTables Pointer to block tables data
 * @param encoderBlockLensDesc Descriptor for lengths of encoder blocks
 * @param encoderBlockLens Pointer to encoder block lengths data
 * @param isBlockStepDesc Descriptor for flags indicating if current step is block-based
 * @param isBlockStep Pointer to block step flags
 * @param stepBlockListDesc Descriptor for list of blocks to process in current step
 * @param stepBlockList Pointer to step block list data
 * @param stepLensDesc Descriptor for lengths of steps
 * @param stepLens Pointer to step lengths data
 * @param recoverBlockListDesc Descriptor for list of blocks to recover
 * @param recoverBlockList Pointer to recover block list data
 * @param recoverLensDesc Descriptor for lengths of recoveries
 * @param recoverLens Pointer to recovery lengths data
 * @param needBlockListDesc Descriptor for list of blocks needed
 * @param needBlockList Pointer to needed block list data
 * @param needBlockLenDesc Descriptor for length of needed blocks
 * @param needBlockLen Pointer to needed block length data
 * @param usedListLenDesc Descriptor for length of used block list
 * @param usedListLen Pointer to used list length data
 * @param freeListDesc Descriptor for list of free blocks
 * @param freeList Pointer to free list data
 * @param freeListLenDesc Descriptor for length of free block list
 * @param freeListLen Pointer to free list length data
 * @param inputIdsDesc Descriptor for input token IDs
 * @param inputIds Pointer to input token IDs data
 * @param preIdsDesc Descriptor for previous token IDs
 * @param preIds Pointer to previous token IDs data
 * @param stepIdxDesc Descriptor for current step index
 * @param stepIdx Pointer to step index data
 * @param nextTokensDesc Descriptor for next predicted tokens
 * @param nextTokens Pointer to next tokens data
 * @param block_size Size of each block in tokens
 * @param encoder_decoder_block_num Number of blocks in encoder-decoder architecture
 * @param first_token_id ID of the first token in the vocabulary
 */
tecocustomStatus_t TECOCUSTOMWINAPI tecocustomStep(
    tecocustomHandle_t handle,
    const tecocustomTensorDescriptor_t stopFlagsDesc, void *stopFlags,
    const tecocustomTensorDescriptor_t seqLensThisTimeDesc, void *seqLensThisTime,
    const tecocustomTensorDescriptor_t oriSeqLensEncoderDesc, void *oriSeqLensEncoder,
    const tecocustomTensorDescriptor_t seqLensEncoderDesc, void *seqLensEncoder,
    const tecocustomTensorDescriptor_t seqLensDecoderDesc, void *seqLensDecoder,
    const tecocustomTensorDescriptor_t blockTablesDesc, void *blockTables,
    const tecocustomTensorDescriptor_t encoderBlockLensDesc, void *encoderBlockLens,
    const tecocustomTensorDescriptor_t isBlockStepDesc, void *isBlockStep,
    const tecocustomTensorDescriptor_t stepBlockListDesc, void *stepBlockList,
    const tecocustomTensorDescriptor_t stepLensDesc, void *stepLens,
    const tecocustomTensorDescriptor_t recoverBlockListDesc, void *recoverBlockList,
    const tecocustomTensorDescriptor_t recoverLensDesc, void *recoverLens,
    const tecocustomTensorDescriptor_t needBlockListDesc, void *needBlockList,
    const tecocustomTensorDescriptor_t needBlockLenDesc, void *needBlockLen,
    const tecocustomTensorDescriptor_t usedListLenDesc, void *usedListLen,
    const tecocustomTensorDescriptor_t freeListDesc, void *freeList,
    const tecocustomTensorDescriptor_t freeListLenDesc, void *freeListLen,
    const tecocustomTensorDescriptor_t inputIdsDesc, void *inputIds,
    const tecocustomTensorDescriptor_t preIdsDesc, void *preIds,
    const tecocustomTensorDescriptor_t stepIdxDesc, void *stepIdx,
    const tecocustomTensorDescriptor_t nextTokensDesc, void *nextTokens,
    const int block_size,
    const int encoder_decoder_block_num,
    const int64_t first_token_id);
/**
 * @brief Get the save info size for the furthest points sampling.
 * @param handle Handle to the Tecocustom library, which manages resources and kernel launches.
 * @param datasetDesc Tensor descriptor for the dataset tensor, describing the shape and data type.
 * @param dataset input and  the shape is (b,n,3)
 * @param idxsDesc Tensor descriptor for the idx tensor, describing the shape and data type.
 * @param idxs output and the shape is (b,m).
 * @return tecocustomStatus_t Status of the operation.
 */
tecocustomStatus_t TECOCUSTOMWINAPI tecocustomFurthestPointSampling(
    tecocustomHandle_t handle, const tecocustomTensorDescriptor_t datasetDesc, const void *dataset,
    const tecocustomTensorDescriptor_t idxsDesc, void *idxs);
#endif  // CUSTOM_INCLUDE_TECOCUSTOM_EXT_H_
