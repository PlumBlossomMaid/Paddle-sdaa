#include "/opt/tecoai/extend/include/sdcops.h"
#include "/softwares/tecopytorch/torch_sdaa/sdaac_ops/include/sdaac_ops.h"

namespace lmik {
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
                                sdaaStream_t stream);
}
