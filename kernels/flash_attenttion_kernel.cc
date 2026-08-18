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

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/tblas_baseop.h"
#include "paddle/phi/extension.h"
#include "sdcops.h"  //NOLINT

#include <cstdlib>
#include <string>

namespace custom_kernel {

struct TensorStride {
  uint32_t lda;
  uint32_t stride;
};

void CheckInputs(const phi::DenseTensor& q,
                 const phi::DenseTensor& k,
                 const phi::DenseTensor& v,
                 float dropout) {
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[seq_len, batch_size, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      q.dtype() == k.dtype(),
      true,
      phi::errors::InvalidArgument("flash_attn q k v dtype must be the same"
                                   "but receive q:{%d} k:{%d}",
                                   q.dtype(),
                                   k.dtype()));
  PADDLE_ENFORCE_EQ(
      q.dtype() == v.dtype(),
      true,
      phi::errors::InvalidArgument("flash_attn q k v dtype must be the same"
                                   "but receive q:{%d} v:{%d}",
                                   q.dtype(),
                                   v.dtype()));
  PADDLE_ENFORCE_EQ(
      dropout <= 0.0,
      true,
      phi::errors::InvalidArgument("flash_attn not support dropout yet"
                                   "but receive dropout:{%f}",
                                   dropout));
}

void CastFP32TOFP16Raw(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       void* dst) {
  std::vector<int> src_dims(phi::vectorize<int>(src.dims()));
  tecodnnTensorDescriptor_t src_Desc =
      sdaa_ops::GetTecodnnTensorDesc(src_dims, src.dtype(), TensorFormat::NCHW);
  tecodnnTensorDescriptor_t dst_Desc = sdaa_ops::GetTecodnnTensorDesc(
      src_dims, phi::DataType::FLOAT16, TensorFormat::NCHW);
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  float alpha = 1.0, beta = 0.0;
  TECODNN_CHECK(tecodnnTransformTensor(
      tecodnnHandle, &alpha, src_Desc, src.data(), &beta, dst_Desc, dst));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(src_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dst_Desc));
}

int64_t GetFP16TensorSize(const phi::DenseTensor& t) {
  return phi::SizeOf(phi::DataType::FLOAT16) * t.numel();
}

TensorStride GenTensorStride(const phi::DenseTensor& t) {
  // t [seq_len, batch_size, num_heads, head_dim]
  auto dims = t.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      4,
      phi::errors::InvalidArgument("stride calculate only support 4D"));
  TensorStride t_stride;
  t_stride.lda = dims[1] * dims[2] * dims[3];
  t_stride.stride = dims[3];
  return t_stride;
}

bool IsEnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr &&
         (std::string(value) == "1" || std::string(value) == "ON" ||
          std::string(value) == "on" || std::string(value) == "true" ||
          std::string(value) == "TRUE");
}

bool UseMaskedFallback() {
  return IsEnvEnabled("PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK") ||
         IsEnvEnabled("PADDLE_SDAA_FLASH_ATTN_MASKED_GRAD_FALLBACK");
}

template <typename T, typename Context>
void FlashAttnMaskedForwardFallback(const Context& dev_ctx,
                                    const phi::DenseTensor& q,
                                    const phi::DenseTensor& k,
                                    const phi::DenseTensor& v,
                                    phi::DenseTensor* out,
                                    phi::DenseTensor* softmax_lse) {
  const auto dims = q.dims();
  const int64_t seq_len = dims[0];
  const int64_t batch_size = dims[1];
  const int64_t head_num = dims[2];
  const int64_t head_dim = dims[3];
  const int64_t batch_heads = batch_size * head_num;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto make_tensor = [](const phi::DDim& shape, phi::DataType dtype) {
    phi::DenseTensor tensor;
    phi::DenseTensorMeta meta = {dtype, shape};
    tensor.set_meta(meta);
    return tensor;
  };
  auto to_bhsd = [&](const phi::DenseTensor& x) {
    phi::DenseTensor tensor = make_tensor(
        phi::make_ddim({batch_size, head_num, seq_len, head_dim}), x.dtype());
    dev_ctx.template Alloc<T>(&tensor);
    sdaa_ops::doTransposeTensor(dev_ctx, x, {1, 2, 0, 3}, &tensor);
    tensor.Resize({batch_heads, seq_len, head_dim});
    return tensor;
  };
  auto restore = [&](const phi::DenseTensor& x) {
    phi::DenseTensor tensor = x;
    tensor.Resize({batch_size, head_num, seq_len, head_dim});
    return tensor;
  };

  phi::DenseTensor q_bhsd = to_bhsd(q);
  phi::DenseTensor k_bhsd = to_bhsd(k);
  phi::DenseTensor v_bhsd = to_bhsd(v);
  phi::DenseTensor scores = make_tensor(
      phi::make_ddim({batch_heads, seq_len, seq_len}), q.dtype());
  dev_ctx.template Alloc<T>(&scores);
  tblas_ops::BatchMatmul<T>(dev_ctx, q_bhsd, k_bhsd, false, true, &scores);
  sdaa_ops::doScaleTensor(dev_ctx, scores, scale, 0.0f, true, false, &scores);

  phi::DenseTensor mask = make_tensor(scores.dims(), q.dtype());
  phi::DenseTensor mask_triu = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&mask);
  dev_ctx.template Alloc<T>(&mask_triu);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), q.dtype(), &mask);
  tblas_ops::TecoBlas<T>::Triu(
      dev_ctx, seq_len, seq_len, batch_heads, 1, mask.data(), mask_triu.data());
  const float mask_value =
      std::numeric_limits<float>::lowest() / static_cast<float>(2);
  sdaa_ops::doScaleTensor(
      dev_ctx, mask_triu, mask_value, 0.0f, true, false, &mask_triu);
  phi::DenseTensor masked_scores = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&masked_scores);
  sdaa_ops::doElementAdd(dev_ctx, scores, mask_triu, -1, &masked_scores);

  phi::DenseTensor probs = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&probs);
  sdaa_ops::doSoftmaxForward(dev_ctx, masked_scores, -1, true, &probs);
  phi::DenseTensor out_bhsd = make_tensor(v_bhsd.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&out_bhsd);
  tblas_ops::BatchMatmul<T>(dev_ctx, probs, v_bhsd, false, false, &out_bhsd);
  sdaa_ops::doTransposeTensor(dev_ctx, restore(out_bhsd), {2, 0, 1, 3}, out);

  softmax_lse->Resize({1});
  dev_ctx.Alloc(softmax_lse, DataType::INT8);
}

template <typename T, typename Context>
void FlashAttnKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& q,
    const phi::DenseTensor& k,
    const phi::DenseTensor& v,
    const paddle::optional<phi::DenseTensor>& fixed_seed_offset,
    const paddle::optional<phi::DenseTensor>& attn_mask,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    phi::DenseTensor* out,
    phi::DenseTensor* softmax,
    phi::DenseTensor* softmax_lse,
    phi::DenseTensor* seed_offset) {
  VLOG(4) << "Call SDAA FlashAttnKernel";
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  CheckInputs(q, k, v, dropout);
  if (attn_mask && UseMaskedFallback()) {
    dev_ctx.template Alloc<T>(out);
    FlashAttnMaskedForwardFallback<T, Context>(dev_ctx, q, k, v, out, softmax_lse);
    return;
  }
  // prepare param
  auto q_dims = q.dims();
  uint32_t attn_seq_len = q_dims[0];
  uint32_t attn_size_per_head = q_dims[3];
  uint32_t attn_head_num = q_dims[1] * q_dims[2];
  const float qk_scalar = 1.0f / std::sqrt(attn_size_per_head);
  FLASH_ATTENTION_MODE attn_mode = is_test ? INFER_MODE : TRAIN_MODE;
  int64_t workspace_size = lmik::flash_attention_get_size(
      attn_seq_len, attn_head_num, attn_size_per_head, attn_mode);
  int64_t softmax_offset = workspace_size;

  auto attn_mask_mode = FLASH_ATTENTION_SOFTMAX_MODE::NO_MASK;
  if (attn_mask) {
    attn_mask_mode = FLASH_ATTENTION_SOFTMAX_MODE::MASK_UP_TRI;
  }

  // get qkv size
  int64_t half_q_size = ((GetFP16TensorSize(q) + 63) / 64) * 64;
  int64_t half_k_size = ((GetFP16TensorSize(k) + 63) / 64) * 64;
  int64_t half_v_size = ((GetFP16TensorSize(v) + 63) / 64) * 64;
  workspace_size += half_q_size + half_k_size + half_v_size;
  VLOG(4) << "workspace_size:" << workspace_size;
  softmax_lse->Resize({workspace_size});
  dev_ctx.Alloc(softmax_lse, DataType::INT8);
  std::uintptr_t softmax_addr =
      reinterpret_cast<uintptr_t>(softmax_lse->data());
  void* half_q_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset);
  void* half_k_addr =
      reinterpret_cast<void*>(softmax_addr + softmax_offset + half_q_size);
  void* half_v_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset +
                                              half_q_size + half_k_size);
  CastFP32TOFP16Raw(dev_ctx, q, half_q_addr);
  CastFP32TOFP16Raw(dev_ctx, k, half_k_addr);
  CastFP32TOFP16Raw(dev_ctx, v, half_v_addr);

  VLOG(4) << "scale:" << qk_scalar;
  // calculate ld && stride
  TensorStride q_stride = GenTensorStride(q);
  TensorStride k_stride = GenTensorStride(k);
  TensorStride v_stride = GenTensorStride(v);

  // float out
  dev_ctx.template Alloc<T>(out);
  TensorStride out_stride = GenTensorStride(*out);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);

  lmik::Flash_Attention_Parameter<float> para;
  para.Q = half_q_addr;
  para.ldQ = q_stride.lda;
  para.strideQ = q_stride.stride;
  para.K = half_k_addr;
  para.ldK = k_stride.lda;
  para.strideK = k_stride.stride;
  para.V = half_v_addr;
  para.ldV = v_stride.lda;
  para.strideV = v_stride.stride;
  para.seq_len = attn_seq_len;
  para.size_per_head = attn_size_per_head;
  para.head_num = attn_head_num;
  para.atten_out = out->data<T>();
  para.ld_O = out_stride.lda;
  para.strideO = out_stride.stride;
  para.qk_scalar = qk_scalar;
  para.atten_mode = attn_mode;
  para.sfm_res = static_cast<float*>(softmax_lse->data());
  para.sfm_mode = attn_mask_mode;
  TCUS_CHECK(lmik::flash_attention_ext<float>(para, custom_stream));
}

template <typename T, typename Context>
void FlashAttnNoMaskGradFallback(const Context& dev_ctx,
                                 const phi::DenseTensor& q,
                                 const phi::DenseTensor& k,
                                 const phi::DenseTensor& v,
                                 const phi::DenseTensor& dout,
                                 phi::DenseTensor* dq,
                                 phi::DenseTensor* dk,
                                 phi::DenseTensor* dv) {
  const auto dims = q.dims();
  const int64_t seq_len = dims[0];
  const int64_t batch_size = dims[1];
  const int64_t head_num = dims[2];
  const int64_t head_dim = dims[3];
  const int64_t batch_heads = batch_size * head_num;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto make_tensor = [](const phi::DDim& shape, phi::DataType dtype) {
    phi::DenseTensor out;
    phi::DenseTensorMeta meta = {dtype, shape};
    out.set_meta(meta);
    return out;
  };
  auto transpose_bhsd = [&](const phi::DenseTensor& x) {
    phi::DenseTensor out = make_tensor(
        phi::make_ddim({batch_size, head_num, seq_len, head_dim}), x.dtype());
    dev_ctx.template Alloc<T>(&out);
    sdaa_ops::doTransposeTensor(dev_ctx, x, {1, 2, 0, 3}, &out);
    out.Resize({batch_heads, seq_len, head_dim});
    return out;
  };

  phi::DenseTensor q_bhsd = transpose_bhsd(q);
  phi::DenseTensor k_bhsd = transpose_bhsd(k);
  phi::DenseTensor v_bhsd = transpose_bhsd(v);
  phi::DenseTensor do_bhsd = transpose_bhsd(dout);

  phi::DenseTensor scores = make_tensor(
      phi::make_ddim({batch_heads, seq_len, seq_len}), q.dtype());
  dev_ctx.template Alloc<T>(&scores);
  tblas_ops::BatchMatmul<T>(dev_ctx, q_bhsd, k_bhsd, false, true, &scores);
  sdaa_ops::doScaleTensor(dev_ctx, scores, scale, 0.0f, true, false, &scores);

  phi::DenseTensor probs = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&probs);
  sdaa_ops::doSoftmaxForward(dev_ctx, scores, -1, true, &probs);

  phi::DenseTensor d_v = make_tensor(v_bhsd.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_v);
  tblas_ops::BatchMatmul<T>(dev_ctx, probs, do_bhsd, true, false, &d_v);
  phi::DenseTensor d_p = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_p);
  tblas_ops::BatchMatmul<T>(dev_ctx, do_bhsd, v_bhsd, false, true, &d_p);
  phi::DenseTensor d_s = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_s);
  sdaa_ops::doSoftmaxBackward(dev_ctx, probs, d_p, -1, true, &d_s);

  phi::DenseTensor d_q = make_tensor(q_bhsd.dims(), q.dtype());
  phi::DenseTensor d_k = make_tensor(k_bhsd.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_q);
  dev_ctx.template Alloc<T>(&d_k);
  tblas_ops::BatchMatmul<T>(dev_ctx, d_s, k_bhsd, false, false, &d_q, scale);
  tblas_ops::BatchMatmul<T>(dev_ctx, d_s, q_bhsd, true, false, &d_k, scale);

  auto restore = [&](const phi::DenseTensor& x, phi::DenseTensor* out) {
    phi::DenseTensor x_bhsd = x;
    x_bhsd.Resize({batch_size, head_num, seq_len, head_dim});
    sdaa_ops::doTransposeTensor(dev_ctx, x_bhsd, {2, 0, 1, 3}, out);
  };
  restore(d_q, dq);
  restore(d_k, dk);
  restore(d_v, dv);
}

template <typename T, typename Context>
void FlashAttnMaskedGradFallback(const Context& dev_ctx,
                                 const phi::DenseTensor& q,
                                 const phi::DenseTensor& k,
                                 const phi::DenseTensor& v,
                                 const phi::DenseTensor& dout,
                                 phi::DenseTensor* dq,
                                 phi::DenseTensor* dk,
                                 phi::DenseTensor* dv) {
  const auto dims = q.dims();
  const int64_t seq_len = dims[0];
  const int64_t batch_size = dims[1];
  const int64_t head_num = dims[2];
  const int64_t head_dim = dims[3];
  const int64_t batch_heads = batch_size * head_num;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto make_tensor = [](const phi::DDim& shape, phi::DataType dtype) {
    phi::DenseTensor out;
    phi::DenseTensorMeta meta = {dtype, shape};
    out.set_meta(meta);
    return out;
  };
  auto transpose_bhsd = [&](const phi::DenseTensor& x) {
    phi::DenseTensor out = make_tensor(
        phi::make_ddim({batch_size, head_num, seq_len, head_dim}), x.dtype());
    dev_ctx.template Alloc<T>(&out);
    sdaa_ops::doTransposeTensor(dev_ctx, x, {1, 2, 0, 3}, &out);
    out.Resize({batch_heads, seq_len, head_dim});
    return out;
  };
  phi::DenseTensor q_bhsd = transpose_bhsd(q);
  phi::DenseTensor k_bhsd = transpose_bhsd(k);
  phi::DenseTensor v_bhsd = transpose_bhsd(v);
  phi::DenseTensor do_bhsd = transpose_bhsd(dout);
  phi::DenseTensor scores = make_tensor(
      phi::make_ddim({batch_heads, seq_len, seq_len}), q.dtype());
  dev_ctx.template Alloc<T>(&scores);
  tblas_ops::BatchMatmul<T>(dev_ctx, q_bhsd, k_bhsd, false, true, &scores);
  sdaa_ops::doScaleTensor(dev_ctx, scores, scale, 0.0f, true, false, &scores);

  phi::DenseTensor mask = make_tensor(scores.dims(), q.dtype());
  phi::DenseTensor mask_triu = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&mask);
  dev_ctx.template Alloc<T>(&mask_triu);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), q.dtype(), &mask);
  tblas_ops::TecoBlas<T>::Triu(
      dev_ctx, seq_len, seq_len, batch_heads, 1, mask.data(), mask_triu.data());
  const float mask_value =
      std::numeric_limits<float>::lowest() / static_cast<float>(2);
  sdaa_ops::doScaleTensor(dev_ctx,
                          mask_triu,
                          mask_value,
                          0.0f,
                          true,
                          false,
                          &mask_triu);
  phi::DenseTensor masked_scores = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&masked_scores);
  sdaa_ops::doElementAdd(dev_ctx, scores, mask_triu, -1, &masked_scores);

  phi::DenseTensor probs = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&probs);
  sdaa_ops::doSoftmaxForward(dev_ctx, masked_scores, -1, true, &probs);
  phi::DenseTensor d_v = make_tensor(v_bhsd.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_v);
  tblas_ops::BatchMatmul<T>(dev_ctx, probs, do_bhsd, true, false, &d_v);
  phi::DenseTensor d_p = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_p);
  tblas_ops::BatchMatmul<T>(dev_ctx, do_bhsd, v_bhsd, false, true, &d_p);
  phi::DenseTensor d_s = make_tensor(scores.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_s);
  sdaa_ops::doSoftmaxBackward(dev_ctx, probs, d_p, -1, true, &d_s);
  phi::DenseTensor d_q = make_tensor(q_bhsd.dims(), q.dtype());
  phi::DenseTensor d_k = make_tensor(k_bhsd.dims(), q.dtype());
  dev_ctx.template Alloc<T>(&d_q);
  dev_ctx.template Alloc<T>(&d_k);
  tblas_ops::BatchMatmul<T>(dev_ctx, d_s, k_bhsd, false, false, &d_q, scale);
  tblas_ops::BatchMatmul<T>(dev_ctx, d_s, q_bhsd, true, false, &d_k, scale);
  auto restore = [&](const phi::DenseTensor& x, phi::DenseTensor* out) {
    phi::DenseTensor x_bhsd = x;
    x_bhsd.Resize({batch_size, head_num, seq_len, head_dim});
    sdaa_ops::doTransposeTensor(dev_ctx, x_bhsd, {2, 0, 1, 3}, out);
  };
  restore(d_q, dq);
  restore(d_k, dk);
  restore(d_v, dv);
}

template <typename T, typename Context>
void FlashAttnGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& q,
    const phi::DenseTensor& k,
    const phi::DenseTensor& v,
    const phi::DenseTensor& out,
    const phi::DenseTensor& softmax_lse,
    const phi::DenseTensor& seed_offset,
    const paddle::optional<phi::DenseTensor>& attn_mask,
    const phi::DenseTensor& dout,
    float dropout,
    bool causal,
    phi::DenseTensor* dq,
    phi::DenseTensor* dk,
    phi::DenseTensor* dv) {
  VLOG(4) << "Call SDAA FlashAttnGradKernel";
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  CheckInputs(q, k, v, dropout);
  dev_ctx.template Alloc<T>(dq);
  dev_ctx.template Alloc<T>(dk);
  dev_ctx.template Alloc<T>(dv);

  if (!attn_mask) {
    FlashAttnNoMaskGradFallback<T, Context>(
        dev_ctx, q, k, v, dout, dq, dk, dv);
    return;
  }

  // The vendor masked backward ABI remains unstable on the current SDK. Keep
  // the hand-written path opt-in until deterministic masked coverage is added.
  if (UseMaskedFallback()) {
    FlashAttnMaskedGradFallback<T, Context>(
        dev_ctx, q, k, v, dout, dq, dk, dv);
    return;
  }

  // prepare param
  FLASH_ATTENTION_MODE attn_mode = TRAIN_MODE;
  TensorStride q_stride = GenTensorStride(q);
  TensorStride k_stride = GenTensorStride(k);
  TensorStride v_stride = GenTensorStride(v);
  TensorStride dout_stride = GenTensorStride(dout);

  auto q_dims{q.dims()};
  uint32_t attn_seq_len = q_dims[0];
  uint32_t attn_size_per_head = q_dims[3];
  uint32_t attn_head_num = q_dims[1] * q_dims[2];
  const float qk_scalar = 1.0f / std::sqrt(attn_size_per_head);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  bool use_float_grad = true;

  // get qkv addr
  int64_t softmax_offset = lmik::flash_attention_get_size(
      attn_seq_len, attn_head_num, attn_size_per_head, attn_mode);
  int64_t half_q_size = ((GetFP16TensorSize(q) + 63) / 64) * 64;
  int64_t half_k_size = ((GetFP16TensorSize(k) + 63) / 64) * 64;
  int64_t half_v_size = ((GetFP16TensorSize(v) + 63) / 64) * 64;
  std::uintptr_t softmax_addr = reinterpret_cast<uintptr_t>(softmax_lse.data());
  void* half_q_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset);
  void* half_k_addr =
      reinterpret_cast<void*>(softmax_addr + softmax_offset + half_q_size);
  void* half_v_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset +
                                              half_q_size + half_k_size);

  // launch kernel
  TCUS_CHECK(
      lmik::flash_attention_bkd<float>(half_q_addr,
                                       q_stride.lda,
                                       q_stride.stride,
                                       half_k_addr,
                                       k_stride.lda,
                                       k_stride.stride,
                                       half_v_addr,
                                       v_stride.lda,
                                       v_stride.stride,
                                       dq->data(),
                                       q_stride.lda,
                                       q_stride.stride,
                                       dk->data(),
                                       k_stride.lda,
                                       k_stride.stride,
                                       dv->data(),
                                       v_stride.lda,
                                       v_stride.stride,
                                       use_float_grad,
                                       attn_seq_len,
                                       attn_size_per_head,
                                       attn_head_num,
                                       const_cast<float*>(dout.data<T>()),
                                       dout_stride.lda,
                                       dout_stride.stride,
                                       qk_scalar,
                                       1.0f,
                                       const_cast<void*>(softmax_lse.data()),
                                       attn_mode,
                                       static_cast<bool>(attn_mask),
                                       custom_stream));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    flash_attn, sdaa, ALL_LAYOUT, custom_kernel::FlashAttnKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnGradKernel,
                          float) {}
