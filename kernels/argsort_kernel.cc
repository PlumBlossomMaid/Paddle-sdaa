// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <thread>

#include "kernels/funcs/sdaa_baseop.h"
#include "tecodnn.h"  // NOLINT
namespace custom_kernel {

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& in,
                   int axis,
                   bool descending,
                   bool stable,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  VLOG(4) << "call sdaa ArgsortKernel";
  int dim_size = in.dims().size();
  if (axis < 0) {
    axis += dim_size;
  }
  PADDLE_ENFORCE_EQ(
      dim_size,
      1,
      phi::errors::InvalidArgument("tecodnn only support input is 1 D."
                                   "But recived: input dims is %d",
                                   dim_size));
  PADDLE_ENFORCE_EQ(
      axis,
      0,
      phi::errors::InvalidArgument("tecodnn only support axis = 0 for argsort."
                                   "But recived: axis is %d",
                                   axis));
  const int Len = indices->numel();
  bool flag = (Len >= 4096) && (Len == (-Len & Len));
  PADDLE_ENFORCE_EQ(
      flag,
      true,
      phi::errors::InvalidArgument("Customized for shape should be the "
                                   "integral power of 2 and not less than 4096."
                                   "But recived: shape is %d",
                                   Len));
  dev_ctx.template Alloc<T>(output);
  dev_ctx.template Alloc<int64_t>(indices);
  std::vector<int> ind_dimensions(1, Len);

  phi::DenseTensor transformed;
  const void* sort_input = in.data();
  if (!descending) {
    transformed.set_meta(in.meta());
    dev_ctx.template Alloc<T>(&transformed);
    phi::Copy(dev_ctx, in, transformed.place(), false, &transformed);
    tecodnnTensorDescriptor_t transform_desc =
        sdaa_ops::GetTecodnnTensorDesc(
            ind_dimensions, in.dtype(), TensorFormat::Undefined);
    tecodnnHandle_t transform_handle = GetHandleFromCTX(dev_ctx);
    const float negate = -1.0f;
    TECODNN_CHECK(tecodnnScaleTensor(transform_handle,
                                     transform_desc,
                                     transformed.data(),
                                     &negate));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(transform_desc));
    sort_input = transformed.data();
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, in.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t index_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, indices->dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, output->dtype(), TensorFormat::Undefined);
  size_t workspace_size = 0;
  TECODNN_CHECK(tecodnnGetTopkExWorkspaceSize(tecodnnHandle,
                                              axis,
                                              Len,
                                              true,
                                              true,
                                              x_Desc,
                                              y_Desc,
                                              index_Desc,
                                              &workspace_size));
  phi::DenseTensor dev_workspace;
  dev_workspace.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  dev_ctx.Alloc(&dev_workspace, phi::DataType::INT8);
  phi::DenseTensor sorted_transformed;
  void* sort_output = output->data();
  if (!descending) {
    sorted_transformed.set_meta(output->meta());
    dev_ctx.template Alloc<T>(&sorted_transformed);
    sort_output = sorted_transformed.data();
  }
  TECODNN_CHECK(tecodnnTopkEx(tecodnnHandle,
                              axis,
                              Len,
                              true,
                              true,
                              x_Desc,
                              sort_input,
                              y_Desc,
                              sort_output,
                              index_Desc,
                              indices->data(),
                              dev_workspace.data(),
                              workspace_size));
  if (!descending) {
    const float negate = -1.0f;
    TECODNN_CHECK(tecodnnScaleTensor(tecodnnHandle,
                                     y_Desc,
                                     sorted_transformed.data(),
                                     &negate));
    phi::Copy(dev_ctx, sorted_transformed, output->place(), false, output);
  }
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    argsort, sdaa, ALL_LAYOUT, custom_kernel::ArgsortKernel, float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
