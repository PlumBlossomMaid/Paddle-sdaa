#include "sdcops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "tecodnn.h"
#include "tecocustom.h"

namespace {

sdaac::sdaacDataTypes_t ToSdaacDataType(DataTypes_t dtype) {
  switch (dtype) {
    case DATA_FLOAT:
      return sdaac::DATA_FLOAT;
    case DATA_HALF:
      return sdaac::DATA_HALF;
    case DATA_UINT8:
      return sdaac::DATA_UINT8;
    case DATA_INT8:
      return sdaac::DATA_INT8;
    case DATA_INT16:
      return sdaac::DATA_INT16;
    case DATA_INT32:
      return sdaac::DATA_INT32;
    case DATA_INT64:
      return sdaac::DATA_INT64;
    case DATA_BOOL:
      return sdaac::DATA_BOOL;
    case DATA_BFLOAT16:
      return sdaac::DATA_BFLOAT16;
    default:
      return sdaac::DATA_FLOAT;
  }
}

sdcStatus_t ToSdcStatus(sdaac::sdaacStatus_t status) {
  switch (status) {
    case sdaac::SDAAC_SUCCESS:
      return SDC_SUCCESS;
    case sdaac::SDAAC_NO_SUPPORT:
      return SDC_NO_SUPPORT;
    default:
      return SDC_FAILED;
  }
}

sdcStatus_t ToSdcStatus(tecodnnStatus_t status) {
  switch (status) {
    case TECODNN_STATUS_SUCCESS:
      return SDC_SUCCESS;
    case TECODNN_STATUS_NOT_SUPPORTED:
      return SDC_NO_SUPPORT;
    default:
      return SDC_FAILED;
  }
}

sdcStatus_t ToSdcStatus(tecocustomStatus_t status) {
  switch (status) {
    case TECOCUSTOM_STATUS_SUCCESS:
      return SDC_SUCCESS;
    case TECOCUSTOM_STATUS_NOT_SUPPORTED:
      return SDC_NO_SUPPORT;
    default:
      return SDC_FAILED;
  }
}

tecodnnDataType_t ToTecodnnDataType(DataTypes_t dtype) {
  switch (dtype) {
    case DATA_FLOAT:
      return TECODNN_DATA_FLOAT;
    case DATA_HALF:
      return TECODNN_DATA_HALF;
    case DATA_UINT8:
      return TECODNN_DATA_UINT8;
    case DATA_INT8:
      return TECODNN_DATA_INT8;
    case DATA_INT16:
      return TECODNN_DATA_INT16;
    case DATA_INT32:
      return TECODNN_DATA_INT32;
    case DATA_INT64:
      return TECODNN_DATA_INT64;
    case DATA_BOOL:
      return TECODNN_DATA_BOOL;
    case DATA_BFLOAT16:
      return TECODNN_DATA_BFLOAT16;
    default:
      return TECODNN_DATA_FLOAT;
  }
}

tecocustomDataType_t ToTecocustomDataType(DataTypes_t dtype) {
  switch (dtype) {
    case DATA_FLOAT:
      return TECOCUSTOM_DATA_FLOAT;
    case DATA_HALF:
      return TECOCUSTOM_DATA_HALF;
    case DATA_UINT8:
      return TECOCUSTOM_DATA_UINT8;
    case DATA_INT8:
      return TECOCUSTOM_DATA_INT8;
    case DATA_INT16:
      return TECOCUSTOM_DATA_INT16;
    case DATA_INT32:
      return TECOCUSTOM_DATA_INT32;
    case DATA_INT64:
      return TECOCUSTOM_DATA_INT64;
    case DATA_BOOL:
      return TECOCUSTOM_DATA_BOOL;
    case DATA_BFLOAT16:
      return TECOCUSTOM_DATA_BFLOAT16;
    default:
      return TECOCUSTOM_DATA_FLOAT;
  }
}

std::vector<int> ToIntDims(const int64_t* dims, int64_t dim_size) {
  std::vector<int> out(std::max<int64_t>(dim_size, 1), 1);
  for (int64_t i = 0; i < dim_size; ++i) {
    out[i] = static_cast<int>(dims[i]);
  }
  return out;
}

std::vector<int> ContiguousStrides(const std::vector<int>& dims) {
  std::vector<int> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

tecodnnTensorDescriptor_t CreateTensorDesc(const int64_t* dims,
                                           int64_t dim_size,
                                           DataTypes_t dtype) {
  auto int_dims = ToIntDims(dims, dim_size);
  auto strides = ContiguousStrides(int_dims);
  tecodnnTensorDescriptor_t desc = nullptr;
  if (tecodnnCreateTensorDescriptor(&desc) != TECODNN_STATUS_SUCCESS) {
    return nullptr;
  }
  if (tecodnnSetTensorNdDescriptor(desc,
                                   ToTecodnnDataType(dtype),
                                   static_cast<int>(int_dims.size()),
                                   int_dims.data(),
                                   strides.data()) != TECODNN_STATUS_SUCCESS) {
    tecodnnDestroyTensorDescriptor(desc);
    return nullptr;
  }
  return desc;
}

tecocustomTensorDescriptor_t CreateCustomTensorDesc(const int64_t* dims,
                                                    int64_t dim_size,
                                                    DataTypes_t dtype) {
  auto int_dims = ToIntDims(dims, dim_size);
  auto strides = ContiguousStrides(int_dims);
  tecocustomTensorDescriptor_t desc = nullptr;
  if (tecocustomCreateTensorDescriptor(&desc) != TECOCUSTOM_STATUS_SUCCESS) {
    return nullptr;
  }
  if (tecocustomSetTensorNdDescriptor(desc,
                                      ToTecocustomDataType(dtype),
                                      static_cast<int>(int_dims.size()),
                                      int_dims.data(),
                                      strides.data()) != TECOCUSTOM_STATUS_SUCCESS) {
    tecocustomDestroyTensorDescriptor(desc);
    return nullptr;
  }
  return desc;
}

size_t AlignUp(size_t value, size_t alignment = 64) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct FlashWorkspaceLayout {
  size_t fwd_workspace;
  size_t bwd_workspace;
  size_t workspace;
  size_t save_info;
  size_t bias;
  size_t total;
};

sdcStatus_t GetFlashWorkspaceLayout(int seq_len,
                                    int head_num,
                                    int size_per_head,
                                    FLASH_ATTENTION_MODE mode,
                                    FlashWorkspaceLayout* layout) {
  layout->fwd_workspace = 0;
  layout->bwd_workspace = 0;
  layout->workspace = 0;
  layout->save_info = 0;
  layout->bias = static_cast<size_t>(seq_len) * seq_len * sizeof(float);
  layout->total = 0;
  constexpr int batch_size = 1;
  constexpr int head_group = 1;
  if (tecocustomGetFlashAttentionForwardWorkspaceSize(&layout->fwd_workspace,
                                                      seq_len,
                                                      seq_len,
                                                      batch_size,
                                                      head_num,
                                                      size_per_head,
                                                      size_per_head) !=
      TECOCUSTOM_STATUS_SUCCESS) {
    return SDC_FAILED;
  }
  if (mode == TRAIN_MODE) {
    if (tecocustomGetFlashAttentionBackwardWorkspaceSize(&layout->bwd_workspace,
                                                         seq_len,
                                                         seq_len,
                                                         batch_size,
                                                         head_num,
                                                         head_group,
                                                         size_per_head,
                                                         size_per_head) !=
        TECOCUSTOM_STATUS_SUCCESS) {
      return SDC_FAILED;
    }
    int save_info = 0;
    if (tecocustomGetFlashAttentionForwardSaveInfoSize(&save_info,
                                                       seq_len,
                                                       seq_len,
                                                       batch_size,
                                                       head_num) !=
        TECOCUSTOM_STATUS_SUCCESS) {
      return SDC_FAILED;
    }
    layout->save_info = static_cast<size_t>(save_info);
  }
  layout->workspace = AlignUp(std::max(layout->fwd_workspace, layout->bwd_workspace));
  layout->workspace = std::max(layout->workspace, AlignUp(sizeof(float)));
  layout->save_info = AlignUp(std::max(layout->save_info, sizeof(float)));
  layout->bias = AlignUp(layout->bias);
  layout->total = AlignUp(layout->workspace + layout->save_info + layout->bias);
  return SDC_SUCCESS;
}

tecodnnTensorDescriptor_t CreateFlashTensorDesc(int seq_len,
                                                int head_num,
                                                int size_per_head,
                                                DataTypes_t dtype) {
  int64_t dims[4] = {1, seq_len, head_num, size_per_head};
  return CreateTensorDesc(dims, 4, dtype);
}

tecocustomTensorDescriptor_t CreateCustomFlashTensorDesc(int seq_len,
                                                          int head_num,
                                                          int size_per_head,
                                                          DataTypes_t dtype) {
  int64_t dims[4] = {1, seq_len, head_num, size_per_head};
  return CreateCustomTensorDesc(dims, 4, dtype);
}

tecocustomTensorDescriptor_t CreateCustomFlashBiasDesc(int seq_len) {
  int64_t dims[4] = {1, 1, seq_len, seq_len};
  return CreateCustomTensorDesc(dims, 4, DATA_FLOAT);
}

tecocustomTensorDescriptor_t CreateCustomByteTensorDesc(size_t bytes) {
  int64_t dims[1] = {
      static_cast<int64_t>(std::max<size_t>((bytes + sizeof(float) - 1) / sizeof(float), 1))};
  return CreateCustomTensorDesc(dims, 1, DATA_FLOAT);
}

tecodnnTensorDescriptor_t CreateByteTensorDesc(size_t bytes) {
  int64_t dims[1] = {static_cast<int64_t>(std::max<size_t>(bytes, 1))};
  return CreateTensorDesc(dims, 1, DATA_INT8);
}

sdcStatus_t WithWorkspace(size_t n, const std::function<sdaac::sdaacStatus_t(void*)>& fn) {
  void* workspace = nullptr;
  size_t workspace_size = sdaac::get_distributions_workspace(n);
  if (workspace_size > 0 && sdaaMalloc(&workspace, workspace_size) != sdaaSuccess) {
    return SDC_FAILED;
  }
  auto status = ToSdcStatus(fn(workspace));
  if (sdaaDeviceSynchronize() != sdaaSuccess) {
    status = SDC_FAILED;
  }
  if (workspace) {
    sdaaFree(workspace);
  }
  return status;
}

bool IsDevicePointer(const void* ptr) {
  sdaaPointerAttributes attrs;
  std::memset(&attrs, 0, sizeof(attrs));
  return sdaaPointerGetAttributes(&attrs, ptr) == sdaaSuccess &&
         attrs.type == sdaaMemoryTypeDevice;
}

sdcStatus_t CopyFloatToDevice(float value, float* out, sdaaStream_t stream) {
  if (out == nullptr) {
    return SDC_SUCCESS;
  }
  if (IsDevicePointer(out)) {
    return sdaaMemcpyAsync(out, &value, sizeof(float), sdaaMemcpyHostToDevice, stream) ==
                   sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  *out = value;
  return SDC_SUCCESS;
}

sdcStatus_t CopyBoolToDevice(bool value, bool* out, sdaaStream_t stream) {
  if (out == nullptr) {
    return SDC_SUCCESS;
  }
  if (IsDevicePointer(out)) {
    return sdaaMemcpyAsync(out, &value, sizeof(bool), sdaaMemcpyHostToDevice, stream) ==
                   sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  *out = value;
  return SDC_SUCCESS;
}

sdcStatus_t ReadFloatScalar(const float* ptr, float* out) {
  if (ptr == nullptr) {
    *out = 0.0f;
    return SDC_SUCCESS;
  }
  if (IsDevicePointer(ptr)) {
    return sdaaMemcpy(out, ptr, sizeof(float), sdaaMemcpyDeviceToHost) == sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  *out = *ptr;
  return SDC_SUCCESS;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exponent = (h & 0x7c00) >> 10;
  uint32_t mantissa = h & 0x03ff;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    exponent = exponent + (127 - 15);
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | static_cast<uint16_t>((mantissa + 0x1000) >> 13);
  }
  if (exponent >= 0x1f) {
    return sign | 0x7c00;
  }
  return sign | static_cast<uint16_t>(exponent << 10) |
         static_cast<uint16_t>((mantissa + 0x1000) >> 13);
}

sdcStatus_t CopyTypedDeviceToFloatVector(const void* src,
                                         DataTypes_t dtype,
                                         int n_total,
                                         std::vector<float>* out) {
  out->resize(n_total);
  if (dtype == DATA_FLOAT) {
    return sdaaMemcpy(out->data(), src, n_total * sizeof(float),
                      sdaaMemcpyDeviceToHost) == sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  if (dtype == DATA_HALF) {
    std::vector<uint16_t> half(n_total);
    if (sdaaMemcpy(half.data(), src, n_total * sizeof(uint16_t),
                   sdaaMemcpyDeviceToHost) != sdaaSuccess) {
      return SDC_FAILED;
    }
    for (int i = 0; i < n_total; ++i) {
      (*out)[i] = HalfToFloat(half[i]);
    }
    return SDC_SUCCESS;
  }
  return SDC_NO_SUPPORT;
}

sdcStatus_t CopyFloatVectorToTypedDevice(const std::vector<float>& in,
                                         DataTypes_t dtype,
                                         void* dst) {
  if (dtype == DATA_FLOAT) {
    return sdaaMemcpy(dst, in.data(), in.size() * sizeof(float),
                      sdaaMemcpyHostToDevice) == sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  if (dtype == DATA_HALF) {
    std::vector<uint16_t> half(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      half[i] = FloatToHalf(in[i]);
    }
    return sdaaMemcpy(dst, half.data(), half.size() * sizeof(uint16_t),
                      sdaaMemcpyHostToDevice) == sdaaSuccess
               ? SDC_SUCCESS
               : SDC_FAILED;
  }
  return SDC_NO_SUPPORT;
}

sdcStatus_t AdamFloatFallback(int n_total,
                              void* device_vec[4],
                              void* device_vec_out[3],
                              float beta1,
                              float beta2,
                              float beta1_pow,
                              float beta2_pow,
                              float epsilon,
                              float lr,
                              float* beta1_pow_out,
                              float* beta2_pow_out,
                              DataTypes_t dtypes[3],
                              float weight_decay,
                              sdaaStream_t stream) {
  if (n_total <= 0) {
    return SDC_SUCCESS;
  }

  std::vector<float> grad;
  std::vector<float> param;
  std::vector<float> moment1;
  std::vector<float> moment2;
  if (CopyTypedDeviceToFloatVector(device_vec[0], dtypes[1], n_total, &grad) !=
          SDC_SUCCESS ||
      CopyTypedDeviceToFloatVector(device_vec[1], dtypes[0], n_total, &param) !=
          SDC_SUCCESS ||
      CopyTypedDeviceToFloatVector(device_vec[2], dtypes[2], n_total, &moment1) !=
          SDC_SUCCESS ||
      CopyTypedDeviceToFloatVector(device_vec[3], dtypes[2], n_total, &moment2) !=
          SDC_SUCCESS) {
    return SDC_FAILED;
  }

  std::vector<float> param_out(n_total);
  std::vector<float> moment1_out(n_total);
  std::vector<float> moment2_out(n_total);
  const float beta1_correction = 1.0f - beta1_pow;
  const float beta2_correction = 1.0f - beta2_pow;
  for (int i = 0; i < n_total; ++i) {
    const float g = grad[i];
    const float m1 = beta1 * moment1[i] + (1.0f - beta1) * g;
    const float m2 = beta2 * moment2[i] + (1.0f - beta2) * g * g;
    const float param_base =
        weight_decay == 0.0f ? param[i] : param[i] * (1.0f - lr * weight_decay);
    moment1_out[i] = m1;
    moment2_out[i] = m2;
    const float m1_unbiased = m1 / beta1_correction;
    const float m2_unbiased = m2 / beta2_correction;
    param_out[i] = param_base - lr * m1_unbiased / (std::sqrt(m2_unbiased) + epsilon);
  }

  if (CopyFloatVectorToTypedDevice(param_out, dtypes[0], device_vec_out[0]) !=
          SDC_SUCCESS ||
      CopyFloatVectorToTypedDevice(moment1_out, dtypes[2], device_vec_out[1]) !=
          SDC_SUCCESS ||
      CopyFloatVectorToTypedDevice(moment2_out, dtypes[2], device_vec_out[2]) !=
          SDC_SUCCESS) {
    return SDC_FAILED;
  }

  if (CopyFloatToDevice(beta1 * beta1_pow, beta1_pow_out, stream) != SDC_SUCCESS ||
      CopyFloatToDevice(beta2 * beta2_pow, beta2_pow_out, stream) != SDC_SUCCESS) {
    return SDC_FAILED;
  }
  return SDC_SUCCESS;
}

sdcStatus_t CheckFiniteAndUnscaleFallback(int M,
                                          int64_t* n_total,
                                          void* device_vec[1],
                                          void* device_vec_out[1],
                                          const float* scalar,
                                          bool* device_ret,
                                          DataTypes_t dtype,
                                          sdaaStream_t stream) {
  if (dtype != DATA_FLOAT && dtype != DATA_HALF) {
    return SDC_NO_SUPPORT;
  }
  if (sdaaStreamSynchronize(stream) != sdaaSuccess) {
    return SDC_FAILED;
  }

  std::vector<int64_t> sizes(M);
  std::vector<void*> inputs(M);
  std::vector<void*> outputs(M);
  if (sdaaMemcpy(sizes.data(), n_total, M * sizeof(int64_t), sdaaMemcpyDeviceToHost) !=
          sdaaSuccess ||
      sdaaMemcpy(inputs.data(), device_vec, M * sizeof(void*), sdaaMemcpyDeviceToHost) !=
          sdaaSuccess ||
      sdaaMemcpy(outputs.data(), device_vec_out, M * sizeof(void*),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess) {
    return SDC_FAILED;
  }

  float scale = 1.0f;
  if (ReadFloatScalar(scalar, &scale) != SDC_SUCCESS || scale == 0.0f) {
    return SDC_FAILED;
  }
  bool found = false;
  if (dtype == DATA_FLOAT) {
    for (int i = 0; i < M; ++i) {
      std::vector<float> in(sizes[i]);
      std::vector<float> out(sizes[i]);
      if (sdaaMemcpy(in.data(), inputs[i], sizes[i] * sizeof(float),
                     sdaaMemcpyDeviceToHost) != sdaaSuccess) {
        return SDC_FAILED;
      }
      for (int64_t j = 0; j < sizes[i]; ++j) {
        found = found || !std::isfinite(in[j]);
        out[j] = in[j] / scale;
      }
      if (sdaaMemcpy(outputs[i], out.data(), sizes[i] * sizeof(float),
                     sdaaMemcpyHostToDevice) != sdaaSuccess) {
        return SDC_FAILED;
      }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      std::vector<uint16_t> in(sizes[i]);
      std::vector<uint16_t> out(sizes[i]);
      if (sdaaMemcpy(in.data(), inputs[i], sizes[i] * sizeof(uint16_t),
                     sdaaMemcpyDeviceToHost) != sdaaSuccess) {
        return SDC_FAILED;
      }
      for (int64_t j = 0; j < sizes[i]; ++j) {
        const float value = HalfToFloat(in[j]);
        found = found || !std::isfinite(value);
        out[j] = FloatToHalf(value / scale);
      }
      if (sdaaMemcpy(outputs[i], out.data(), sizes[i] * sizeof(uint16_t),
                     sdaaMemcpyHostToDevice) != sdaaSuccess) {
        return SDC_FAILED;
      }
    }
  }

  return CopyBoolToDevice(found, device_ret, stream);
}

}  // namespace

namespace sdcops {

sdcStatus_t pd_dropout_kernel(const void* self,
                              void* result,
                              void* mask,
                              DataTypes_t dtype,
                              size_t n,
                              float keep_prob,
                              uint64_t seed,
                              uint64_t offset,
                              uint64_t max_threads,
                              sdaaStream_t stream) {
  return ToSdcStatus(sdaac::dropout_kernel(self,
                                           result,
                                           mask,
                                           n,
                                           keep_prob,
                                           4,
                                           seed,
                                           offset,
                                           dtype == DATA_HALF,
                                           stream));
}

sdcStatus_t pd_normal_kernel(void* result,
                             size_t n,
                             float mean,
                             float std,
                             uint64_t seed,
                             uint64_t offset,
                             uint64_t max_threads,
                             sdaaStream_t stream) {
  return WithWorkspace(n, [&](void* workspace) {
    return sdaac::normal_kernel(
        result, workspace, n, mean, std, seed, offset, sdaac::DATA_FLOAT, stream);
  });
}

sdcStatus_t random_int_from_to_kernel(void* result,
                                      size_t n,
                                      uint64_t range,
                                      int64_t base,
                                      uint64_t seed,
                                      uint64_t offset,
                                      sdaaStream_t stream) {
  return WithWorkspace(n, [&](void* workspace) {
    return sdaac::random_int_from_to_kernel(
        result, workspace, n, range, base, seed, offset, stream);
  });
}

sdcStatus_t binary_ops_tt(const void* A,
                          const int64_t* a_shape,
                          const int64_t a_dims,
                          const void* B,
                          const int64_t* b_shape,
                          const int64_t b_dims,
                          void* C,
                          const int64_t* c_shape,
                          const int64_t c_dims,
                          void* scalar,
                          BinaryOps_t op,
                          DataTypes_t dtype,
                          void* workspace,
                          sdaaStream_t stream) {
  tecodnnHandle_t handle = nullptr;
  tecodnnTensorDescriptor_t a_desc = nullptr;
  tecodnnTensorDescriptor_t b_desc = nullptr;
  tecodnnTensorDescriptor_t c_desc = nullptr;
  auto ret = tecodnnCreate(&handle);
  if (ret != TECODNN_STATUS_SUCCESS) {
    return ToSdcStatus(ret);
  }
  ret = tecodnnSetStream(handle, stream);
  if (ret != TECODNN_STATUS_SUCCESS) {
    tecodnnDestroy(handle);
    return ToSdcStatus(ret);
  }

  a_desc = CreateTensorDesc(a_shape, a_dims, dtype);
  b_desc = CreateTensorDesc(b_shape, b_dims, dtype);
  c_desc = CreateTensorDesc(c_shape, c_dims, dtype);
  if (a_desc == nullptr || b_desc == nullptr || c_desc == nullptr) {
    if (a_desc) tecodnnDestroyTensorDescriptor(a_desc);
    if (b_desc) tecodnnDestroyTensorDescriptor(b_desc);
    if (c_desc) tecodnnDestroyTensorDescriptor(c_desc);
    tecodnnDestroy(handle);
    return SDC_FAILED;
  }

  switch (op) {
    case BINARY_ADD:
      ret = tecodnnAddTensorEx(handle, a_desc, A, b_desc, B, c_desc, C);
      break;
    case BINARY_MUL:
      ret = tecodnnMulTensorEx(handle, a_desc, A, b_desc, B, c_desc, C);
      break;
    case BINARY_SUB:
      ret = tecodnnSubTensorEx(handle, a_desc, A, b_desc, B, c_desc, C);
      break;
    case BINARY_DIV:
      ret = tecodnnDivTensorEx(handle, a_desc, A, b_desc, B, c_desc, C);
      break;
    default:
      ret = TECODNN_STATUS_NOT_SUPPORTED;
      break;
  }

  tecodnnDestroyTensorDescriptor(a_desc);
  tecodnnDestroyTensorDescriptor(b_desc);
  tecodnnDestroyTensorDescriptor(c_desc);
  tecodnnDestroy(handle);
  return ToSdcStatus(ret);
}

sdcStatus_t multi_precision_add(const void* x,
                                const void* y,
                                void* out,
                                const int* dims,
                                int dim_size,
                                sdaaStream_t stream) {
  return SDC_NO_SUPPORT;
}

sdcStatus_t multi_t_ops_t_momentum(int n_total,
                                   void* device_vec[3],
                                   const float* lr,
                                   float mu_f,
                                   float coeff,
                                   bool use_nesterov,
                                   bool l2_decay,
                                   sdaaStream_t stream) {
  float lr_host = 0.0f;
  if (ReadFloatScalar(lr, &lr_host) != SDC_SUCCESS) {
    return SDC_FAILED;
  }
  if (sdaaStreamSynchronize(stream) != sdaaSuccess) {
    return SDC_FAILED;
  }

  std::vector<float> param(n_total);
  std::vector<float> grad(n_total);
  std::vector<float> velocity(n_total);
  if (sdaaMemcpy(param.data(), device_vec[0], n_total * sizeof(float),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(grad.data(), device_vec[1], n_total * sizeof(float),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(velocity.data(), device_vec[2], n_total * sizeof(float),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess) {
    return SDC_FAILED;
  }

  for (int i = 0; i < n_total; ++i) {
    float g = grad[i];
    if (l2_decay) {
      g += coeff * param[i];
    }
    const float v = mu_f * velocity[i] + g;
    velocity[i] = v;
    param[i] -= lr_host * (use_nesterov ? (g + mu_f * v) : v);
  }

  if (sdaaMemcpy(device_vec[0], param.data(), n_total * sizeof(float),
                 sdaaMemcpyHostToDevice) != sdaaSuccess ||
      sdaaMemcpy(device_vec[2], velocity.data(), n_total * sizeof(float),
                 sdaaMemcpyHostToDevice) != sdaaSuccess) {
    return SDC_FAILED;
  }
  return SDC_SUCCESS;
}

sdcStatus_t merged_momentum(int M,
                            int64_t* n_total,
                            void** device_vec,
                            const float* lr,
                            const float* coeff,
                            float mu_f,
                            int* l2_decay,
                            bool use_nesterov,
                            sdaaStream_t stream) {
  if (M <= 0 || n_total == nullptr || device_vec == nullptr || lr == nullptr ||
      coeff == nullptr || l2_decay == nullptr) {
    return SDC_FAILED;
  }

  std::vector<void*> param_ptrs(M);
  std::vector<void*> grad_ptrs(M);
  std::vector<void*> velocity_ptrs(M);
  std::vector<int64_t> lengths(M);
  std::vector<float> lr_host(M);
  std::vector<float> coeff_host(M);
  std::vector<int> l2_decay_host(M);
  if (sdaaMemcpy(lengths.data(),
                 n_total,
                 M * sizeof(int64_t),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(param_ptrs.data(),
                 device_vec[0],
                 M * sizeof(void*),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(grad_ptrs.data(),
                 device_vec[1],
                 M * sizeof(void*),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(velocity_ptrs.data(),
                 device_vec[2],
                 M * sizeof(void*),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(lr_host.data(),
                 lr,
                 M * sizeof(float),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(coeff_host.data(),
                 coeff,
                 M * sizeof(float),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess ||
      sdaaMemcpy(l2_decay_host.data(),
                 l2_decay,
                 M * sizeof(int),
                 sdaaMemcpyDeviceToHost) != sdaaSuccess) {
    return SDC_FAILED;
  }

  if (sdaaStreamSynchronize(stream) != sdaaSuccess) {
    return SDC_FAILED;
  }
  for (int i = 0; i < M; ++i) {
    void* tensors[3] = {
        param_ptrs[i], grad_ptrs[i], velocity_ptrs[i]};
    sdcStatus_t status = multi_t_ops_t_momentum(lengths[i],
                                                 tensors,
                                                 &lr_host[i],
                                                 mu_f,
                                                 coeff_host[i],
                                                 use_nesterov,
                                                 l2_decay_host[i] != 0,
                                                 stream);
    if (status != SDC_SUCCESS) {
      return status;
    }
  }
  return SDC_SUCCESS;
}

sdcStatus_t pd_adam_out(int n_total,
                        void* device_vec[4],
                        void* device_vec_out[3],
                        float beta1,
                        float beta2,
                        float* beta1_pow,
                        float* beta2_pow,
                        float epsilon,
                        float* lr,
                        int device_scale,
                        float* beta1_pow_out,
                        float* beta2_pow_out,
                        DataTypes_t dtypes[3],
                        sdaaStream_t stream) {
  if (!((dtypes[0] == DATA_FLOAT && dtypes[1] == DATA_FLOAT &&
         dtypes[2] == DATA_FLOAT) ||
        (dtypes[0] == DATA_HALF && dtypes[1] == DATA_HALF &&
         dtypes[2] == DATA_FLOAT) ||
        (dtypes[0] == DATA_FLOAT && dtypes[1] == DATA_HALF &&
         dtypes[2] == DATA_FLOAT))) {
    return SDC_NO_SUPPORT;
  }
  float beta1_pow_host = 0.0f;
  float beta2_pow_host = 0.0f;
  float lr_host = 0.0f;
  if (ReadFloatScalar(beta1_pow, &beta1_pow_host) != SDC_SUCCESS ||
      ReadFloatScalar(beta2_pow, &beta2_pow_host) != SDC_SUCCESS ||
      ReadFloatScalar(lr, &lr_host) != SDC_SUCCESS) {
    return SDC_FAILED;
  }
  return AdamFloatFallback(n_total,
                           device_vec,
                           device_vec_out,
                           beta1,
                           beta2,
                           beta1_pow_host,
                           beta2_pow_host,
                           epsilon,
                           lr_host,
                           beta1_pow_out,
                           beta2_pow_out,
                           dtypes,
                           0.0f,
                           stream);
}

sdcStatus_t pd_adamw_out(int n_total,
                         void* device_vec[4],
                         void* device_vec_out[3],
                         float beta1,
                         float beta2,
                         float* beta1_pow,
                         float* beta2_pow,
                         float epsilon,
                         float* lr,
                         int device_scale,
                         float lr_ratio,
                         float coeff,
                         float* beta1_pow_out,
                         float* beta2_pow_out,
                         DataTypes_t dtypes[3],
                         sdaaStream_t stream) {
  if (!((dtypes[0] == DATA_FLOAT && dtypes[1] == DATA_FLOAT &&
         dtypes[2] == DATA_FLOAT) ||
        (dtypes[0] == DATA_HALF && dtypes[1] == DATA_HALF &&
         dtypes[2] == DATA_FLOAT) ||
        (dtypes[0] == DATA_FLOAT && dtypes[1] == DATA_HALF &&
         dtypes[2] == DATA_FLOAT))) {
    return SDC_NO_SUPPORT;
  }
  float beta1_pow_host = 0.0f;
  float beta2_pow_host = 0.0f;
  float lr_host = 0.0f;
  if (ReadFloatScalar(beta1_pow, &beta1_pow_host) != SDC_SUCCESS ||
      ReadFloatScalar(beta2_pow, &beta2_pow_host) != SDC_SUCCESS ||
      ReadFloatScalar(lr, &lr_host) != SDC_SUCCESS) {
    return SDC_FAILED;
  }
  return AdamFloatFallback(n_total,
                           device_vec,
                           device_vec_out,
                           beta1,
                           beta2,
                           beta1_pow_host,
                           beta2_pow_host,
                           epsilon,
                           lr_host * lr_ratio,
                           beta1_pow_out,
                           beta2_pow_out,
                           dtypes,
                           coeff,
                           stream);
}

template <typename T, typename IndexT>
sdcStatus_t scatter_nd_add(const void* updates,
                           const void* index,
                           void* out,
                           int* updates_dims,
                           int* index_dims,
                           int* out_dims,
                           int updates_dim_size,
                           int index_dim_size,
                           int out_dim_size,
                           sdaaStream_t stream) {
  return SDC_NO_SUPPORT;
}

template sdcStatus_t scatter_nd_add<float, int>(const void*,
                                                const void*,
                                                void*,
                                                int*,
                                                int*,
                                                int*,
                                                int,
                                                int,
                                                int,
                                                sdaaStream_t);
template sdcStatus_t scatter_nd_add<float, int64_t>(const void*,
                                                    const void*,
                                                    void*,
                                                    int*,
                                                    int*,
                                                    int*,
                                                    int,
                                                    int,
                                                    int,
                                                    sdaaStream_t);

sdcStatus_t multi_t_group_ops_t_fusedVSCheckInvalid_out(
    int M,
    int64_t* n_total,
    void* device_vec[1],
    void* device_vec_out[1],
    const float* scalar,
    bool* device_ret,
    DataTypes_t dtype,
    sdaaStream_t stream) {
  return CheckFiniteAndUnscaleFallback(
      M, n_total, device_vec, device_vec_out, scalar, device_ret, dtype, stream);
}

}  // namespace sdcops

namespace lmik {

size_t flash_attention_get_size(int seq_len,
                                int head_num,
                                int size_per_head,
                                FLASH_ATTENTION_MODE mode) {
  FlashWorkspaceLayout layout;
  return GetFlashWorkspaceLayout(seq_len, head_num, size_per_head, mode, &layout) ==
                 SDC_SUCCESS
             ? layout.total
             : 0;
}

template <typename T>
sdcStatus_t flash_attention_ext(Flash_Attention_Parameter<T> para, sdaaStream_t stream) {
  FlashWorkspaceLayout layout;
  auto status = GetFlashWorkspaceLayout(
      para.seq_len, para.head_num, para.size_per_head, para.atten_mode, &layout);
  if (status != SDC_SUCCESS) {
    return status;
  }

  const bool is_training = para.atten_mode == TRAIN_MODE;

  tecocustomHandle_t handle = nullptr;
  tecocustomTensorDescriptor_t q_desc = nullptr;
  tecocustomTensorDescriptor_t k_desc = nullptr;
  tecocustomTensorDescriptor_t v_desc = nullptr;
  tecocustomTensorDescriptor_t o_desc = nullptr;
  tecocustomTensorDescriptor_t workspace_desc = nullptr;
  tecocustomTensorDescriptor_t train_info_desc = nullptr;
  tecocustomTensorDescriptor_t attn_bias_desc = nullptr;

  if (tecocustomCreate(&handle) != TECOCUSTOM_STATUS_SUCCESS ||
      tecocustomSetStream(handle, stream) != TECOCUSTOM_STATUS_SUCCESS ||
      tecocustomSetModelMode(handle,
                             is_training ? TECOCUSTOM_MODEL_TRAINING
                                         : TECOCUSTOM_MODEL_INFER) !=
          TECOCUSTOM_STATUS_SUCCESS) {
    if (handle) tecocustomDestroy(handle);
    return SDC_FAILED;
  }

  q_desc = CreateCustomFlashTensorDesc(para.seq_len, para.head_num, para.size_per_head, DATA_HALF);
  k_desc = CreateCustomFlashTensorDesc(para.seq_len, para.head_num, para.size_per_head, DATA_HALF);
  v_desc = CreateCustomFlashTensorDesc(para.seq_len, para.head_num, para.size_per_head, DATA_HALF);
  o_desc = CreateCustomFlashTensorDesc(para.seq_len, para.head_num, para.size_per_head, DATA_FLOAT);
  workspace_desc = CreateCustomByteTensorDesc(layout.workspace);
  train_info_desc = CreateCustomByteTensorDesc(layout.save_info);
  attn_bias_desc = CreateCustomFlashBiasDesc(para.seq_len);
  if (!q_desc || !k_desc || !v_desc || !o_desc || !workspace_desc ||
      !train_info_desc || !attn_bias_desc) {
    if (q_desc) tecocustomDestroyTensorDescriptor(q_desc);
    if (k_desc) tecocustomDestroyTensorDescriptor(k_desc);
    if (v_desc) tecocustomDestroyTensorDescriptor(v_desc);
    if (o_desc) tecocustomDestroyTensorDescriptor(o_desc);
    if (workspace_desc) tecocustomDestroyTensorDescriptor(workspace_desc);
    if (train_info_desc) tecocustomDestroyTensorDescriptor(train_info_desc);
    if (attn_bias_desc) tecocustomDestroyTensorDescriptor(attn_bias_desc);
    tecocustomDestroy(handle);
    return SDC_FAILED;
  }

  void* workspace = para.sfm_res;
  void* train_info = reinterpret_cast<void*>(reinterpret_cast<char*>(para.sfm_res) +
                                             layout.workspace);
  void* attn_bias = reinterpret_cast<void*>(reinterpret_cast<char*>(para.sfm_res) +
                                            layout.workspace + layout.save_info);
  sdaaMemsetAsync(attn_bias, 0, layout.bias, stream);
  const bool is_mask_up_triangle = para.sfm_mode == MASK_UP_TRI;
  const void* attn_bias_ptr = attn_bias;
  constexpr int batch_size = 1;
  constexpr int head_groups = 1;
  constexpr int head_begin = 0;
  const int total_head_num = para.head_num;
  constexpr int q_offset_logic = 0;
  constexpr float clip_value = 0.0f;
  constexpr bool pvc_k_flag = false;
  constexpr bool pvc_v_flag = false;

  auto ret = tecocustomFlashAttentionForward(handle,
                                             q_desc,
                                             para.Q,
                                             k_desc,
                                             para.K,
                                             v_desc,
                                             para.V,
                                             o_desc,
                                             para.atten_out,
                                             workspace_desc,
                                             workspace,
                                             train_info_desc,
                                             train_info,
                                             attn_bias_desc,
                                             attn_bias_ptr,
                                             para.ldQ,
                                             para.strideQ,
                                             para.seq_len * para.ldQ,
                                             para.ldK,
                                             para.strideK,
                                             para.seq_len * para.ldK,
                                             para.ldV,
                                             para.strideV,
                                             para.seq_len * para.ldV,
                                             batch_size,
                                             para.seq_len,
                                             para.seq_len,
                                             para.size_per_head,
                                             para.size_per_head,
                                             para.head_num,
                                             para.ld_O,
                                             para.strideO,
                                             para.seq_len * para.ld_O,
                                             para.qk_scalar,
                                             head_groups,
                                             is_training,
                                             is_mask_up_triangle ? para.seq_len * para.seq_len : 0,
                                             is_mask_up_triangle,
                                             false,
                                             head_begin,
                                             total_head_num,
                                             q_offset_logic,
                                             clip_value,
                                             pvc_k_flag,
                                             pvc_v_flag);

  tecocustomDestroyTensorDescriptor(q_desc);
  tecocustomDestroyTensorDescriptor(k_desc);
  tecocustomDestroyTensorDescriptor(v_desc);
  tecocustomDestroyTensorDescriptor(o_desc);
  tecocustomDestroyTensorDescriptor(workspace_desc);
  tecocustomDestroyTensorDescriptor(train_info_desc);
  tecocustomDestroyTensorDescriptor(attn_bias_desc);
  tecocustomDestroy(handle);
  return ToSdcStatus(ret);
}

template <typename T>
sdcStatus_t flash_attention_bkd(void* Q,
                                uint32_t ldQ,
                                uint32_t strideQ,
                                void* K,
                                uint32_t ldK,
                                uint32_t strideK,
                                void* V,
                                uint32_t ldV,
                                uint32_t strideV,
                                void* gQ,
                                uint32_t ldgQ,
                                uint32_t stridegQ,
                                void* gK,
                                uint32_t ldgK,
                                uint32_t stridegK,
                                void* gV,
                                uint32_t ldgV,
                                uint32_t stridegV,
                                bool use_float_grad,
                                uint32_t seq_len,
                                uint32_t size_per_head,
                                uint32_t head_num,
                                T* atten_out,
                                uint32_t ld_O,
                                uint32_t strideO,
                                float qk_scalar,
                                float grad_scalar,
                                void* work_space,
                                FLASH_ATTENTION_MODE mode,
                                bool is_mask_up_triangle,
                                sdaaStream_t stream) {
  FlashWorkspaceLayout layout;
  auto status = GetFlashWorkspaceLayout(seq_len, head_num, size_per_head, TRAIN_MODE, &layout);
  if (status != SDC_SUCCESS) {
    return status;
  }

  tecocustomHandle_t handle = nullptr;
  tecocustomTensorDescriptor_t q_desc = nullptr;
  tecocustomTensorDescriptor_t k_desc = nullptr;
  tecocustomTensorDescriptor_t v_desc = nullptr;
  tecocustomTensorDescriptor_t gq_desc = nullptr;
  tecocustomTensorDescriptor_t gk_desc = nullptr;
  tecocustomTensorDescriptor_t gv_desc = nullptr;
  tecocustomTensorDescriptor_t do_desc = nullptr;
  tecocustomTensorDescriptor_t workspace_desc = nullptr;
  tecocustomTensorDescriptor_t train_info_desc = nullptr;
  tecocustomTensorDescriptor_t attn_bias_desc = nullptr;

  if (tecocustomCreate(&handle) != TECOCUSTOM_STATUS_SUCCESS ||
      tecocustomSetStream(handle, stream) != TECOCUSTOM_STATUS_SUCCESS ||
      tecocustomSetModelMode(handle, TECOCUSTOM_MODEL_TRAINING) !=
          TECOCUSTOM_STATUS_SUCCESS) {
    if (handle) tecocustomDestroy(handle);
    return SDC_FAILED;
  }

  q_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_HALF);
  k_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_HALF);
  v_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_HALF);
  gq_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_FLOAT);
  gk_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_FLOAT);
  gv_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_FLOAT);
  do_desc = CreateCustomFlashTensorDesc(seq_len, head_num, size_per_head, DATA_FLOAT);
  workspace_desc = CreateCustomByteTensorDesc(layout.workspace);
  train_info_desc = CreateCustomByteTensorDesc(layout.save_info);
  attn_bias_desc = CreateCustomFlashBiasDesc(seq_len);
  if (!q_desc || !k_desc || !v_desc || !gq_desc || !gk_desc || !gv_desc ||
      !do_desc || !workspace_desc || !train_info_desc || !attn_bias_desc) {
    if (q_desc) tecocustomDestroyTensorDescriptor(q_desc);
    if (k_desc) tecocustomDestroyTensorDescriptor(k_desc);
    if (v_desc) tecocustomDestroyTensorDescriptor(v_desc);
    if (gq_desc) tecocustomDestroyTensorDescriptor(gq_desc);
    if (gk_desc) tecocustomDestroyTensorDescriptor(gk_desc);
    if (gv_desc) tecocustomDestroyTensorDescriptor(gv_desc);
    if (do_desc) tecocustomDestroyTensorDescriptor(do_desc);
    if (workspace_desc) tecocustomDestroyTensorDescriptor(workspace_desc);
    if (train_info_desc) tecocustomDestroyTensorDescriptor(train_info_desc);
    if (attn_bias_desc) tecocustomDestroyTensorDescriptor(attn_bias_desc);
    tecocustomDestroy(handle);
    return SDC_FAILED;
  }

  void* workspace = work_space;
  void* train_info = reinterpret_cast<void*>(reinterpret_cast<char*>(work_space) +
                                             layout.workspace);
  void* attn_bias = reinterpret_cast<void*>(reinterpret_cast<char*>(work_space) +
                                            layout.workspace + layout.save_info);
  sdaaMemsetAsync(attn_bias, 0, layout.bias, stream);
  constexpr int batch_size = 1;
  constexpr int head_groups = 1;
  constexpr int head_begin = 0;
  const int total_head_num = head_num;
  constexpr int q_offset_logic = 0;
  constexpr float clip_value = 0.0f;
  constexpr bool pvc_k_flag = false;
  constexpr bool pvc_v_flag = false;

  auto ret = tecocustomFlashAttentionBackward(handle,
                                              q_desc,
                                              Q,
                                              k_desc,
                                              K,
                                              v_desc,
                                              V,
                                              gq_desc,
                                              gQ,
                                              gk_desc,
                                              gK,
                                              gv_desc,
                                              gV,
                                              do_desc,
                                              atten_out,
                                              workspace_desc,
                                              workspace,
                                              train_info_desc,
                                              train_info,
                                              attn_bias_desc,
                                              is_mask_up_triangle ? attn_bias : nullptr,
                                              ldQ,
                                              strideQ,
                                              seq_len * ldQ,
                                              ldK,
                                              strideK,
                                              seq_len * ldK,
                                              ldV,
                                              strideV,
                                              seq_len * ldV,
                                              ldgQ,
                                              stridegQ,
                                              seq_len * ldgQ,
                                              ldgK,
                                              stridegK,
                                              seq_len * ldgK,
                                              ldgV,
                                              stridegV,
                                              seq_len * ldgV,
                                              batch_size,
                                              seq_len,
                                              seq_len,
                                              size_per_head,
                                              size_per_head,
                                              head_num,
                                              ld_O,
                                              strideO,
                                              seq_len * ld_O,
                                              qk_scalar,
                                              head_groups,
                                              true,
                                              is_mask_up_triangle ? seq_len * seq_len : 0,
                                              is_mask_up_triangle,
                                              false,
                                              head_begin,
                                              total_head_num,
                                              q_offset_logic,
                                              clip_value,
                                              pvc_k_flag,
                                              pvc_v_flag);

  tecocustomDestroyTensorDescriptor(q_desc);
  tecocustomDestroyTensorDescriptor(k_desc);
  tecocustomDestroyTensorDescriptor(v_desc);
  tecocustomDestroyTensorDescriptor(gq_desc);
  tecocustomDestroyTensorDescriptor(gk_desc);
  tecocustomDestroyTensorDescriptor(gv_desc);
  tecocustomDestroyTensorDescriptor(do_desc);
  tecocustomDestroyTensorDescriptor(workspace_desc);
  tecocustomDestroyTensorDescriptor(train_info_desc);
  tecocustomDestroyTensorDescriptor(attn_bias_desc);
  tecocustomDestroy(handle);
  return ToSdcStatus(ret);
}

template sdcStatus_t flash_attention_ext<float>(Flash_Attention_Parameter<float>, sdaaStream_t);
template sdcStatus_t flash_attention_bkd<float>(void*,
                                                uint32_t,
                                                uint32_t,
                                                void*,
                                                uint32_t,
                                                uint32_t,
                                                void*,
                                                uint32_t,
                                                uint32_t,
                                                void*,
                                                uint32_t,
                                                uint32_t,
                                                void*,
                                                uint32_t,
                                                uint32_t,
                                                void*,
                                                uint32_t,
                                                uint32_t,
                                                bool,
                                                uint32_t,
                                                uint32_t,
                                                uint32_t,
                                                float*,
                                                uint32_t,
                                                uint32_t,
                                                float,
                                                float,
                                                void*,
                                                FLASH_ATTENTION_MODE,
                                                bool,
                                                sdaaStream_t);

}  // namespace lmik
