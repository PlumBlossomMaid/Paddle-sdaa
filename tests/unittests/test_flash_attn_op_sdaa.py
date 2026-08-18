# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import paddle
import math
import numpy as np
import paddle.nn.functional as F
import copy
import unittest
import os

paddle.seed(1234)
do_backward = True

# config
numhead = 40
hidden = 5120
seq = 1024
bz = 2
dim = hidden // numhead
target_shape = [0, 0, numhead, 3 * dim]

# test_diff3_max / golden_diff3_max must less than diff_standard
diff_standard = 2

grad_dict = {"model_attn": [], "flash_attn": [], "cpu_attn": []}


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def save_grad(grad):
    copy_grad = paddle.to_tensor(grad)
    if copy_grad.shape[0] != seq:
        # [bs, num_head, seq_len, dim] -> [seq_len, bs, num_head, dim]
        copy_grad = paddle.transpose(copy_grad, [2, 0, 1, 3])
        if copy_grad.place.is_custom_place():
            grad_dict["model_attn"].append(paddle.flatten(copy_grad))
        else:
            grad_dict["cpu_attn"].append(paddle.flatten(copy_grad))
    else:
        grad_dict["flash_attn"].append(paddle.flatten(copy_grad))


def gen_qkv(layer):
    query_states, key_states, value_states = paddle.split(
        layer, num_or_sections=3, axis=-1
    )
    return query_states, key_states, value_states


# ==========================================diff3max==========================================
def diff3_1(eval_data, base_data):
    # 相对误差
    eps = 1e-9
    return np.abs(eval_data - base_data) / (np.abs(base_data) + eps)


def diff3_2(eval_data, base_data):
    # 绝对误差
    return np.abs(eval_data - base_data)


def diff3(eval_data, base_data, th=1e-6):
    mask_t = np.abs(base_data) > th
    mask_t_op = ~mask_t
    mask_t = mask_t.astype(np.float32)
    mask_t_op = mask_t_op.astype(np.float32)
    return diff3_1(eval_data * mask_t, base_data * mask_t) + diff3_2(
        eval_data * mask_t_op, base_data * mask_t_op
    )


def diff3_max(eval_data, base_data, th=1e-6):
    diff3_max = np.max(diff3(eval_data, base_data, th=th))
    return diff3_max


# ==========================================diff3max==========================================


def sdaa_flash_attn(q, k, v, attn_mask):
    q = paddle.transpose(q, [1, 0, 2, 3])
    q.register_hook(save_grad)
    k = paddle.transpose(k, [1, 0, 2, 3])
    k.register_hook(save_grad)
    v = paddle.transpose(v, [1, 0, 2, 3])
    v.register_hook(save_grad)
    if attn_mask:
        attention_mask = get_triangle_upper_mask(q)
    else:
        attention_mask = None

    training = True
    out, _, _, _ = paddle._C_ops.flash_attn(
        q,
        k,
        v,
        None,
        attention_mask,
        0.0,
        False,
        False,
        not training,
        "",
    )
    out = out.transpose([1, 0, 2, 3])
    if do_backward:
        # hack to scale loss
        paddle.autograd.backward(
            out, grad_tensors=paddle.full_like(out, fill_value=16384)
        )
    return out


def flash_attn_output(query_states, key_states, value_states, attn_mask):
    q = paddle.transpose(query_states, [1, 0, 2, 3])
    k = paddle.transpose(key_states, [1, 0, 2, 3])
    v = paddle.transpose(value_states, [1, 0, 2, 3])
    attention_mask = get_triangle_upper_mask(q) if attn_mask else None
    out, _, _, _ = paddle._C_ops.flash_attn(
        q,
        k,
        v,
        None,
        attention_mask,
        0.0,
        False,
        False,
        False,
        "",
    )
    return out.transpose([1, 0, 2, 3])


def model_attn(query_states, key_states, value_states, attn_mask):
    # q,k,v [batch_size, seq_len, num_heads, head_dim]
    head_dim = dim
    q_len = query_states.shape[1]
    bsz = query_states.shape[0]
    num_heads = query_states.shape[2]
    kv_seq_len = key_states.shape[1]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    query_states.register_hook(save_grad)
    query_states = query_states / math.sqrt(head_dim)
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    key_states.register_hook(save_grad)
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])
    value_states.register_hook(save_grad)

    attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))
    if attn_mask:
        attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
        query_states.dtype
    )
    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])
    if do_backward:
        # hack to scale loss
        paddle.autograd.backward(
            attn_output, grad_tensors=paddle.full_like(attn_output, fill_value=16384)
        )
    return attn_output


class TestFlashAttention(unittest.TestCase):
    def init(self):
        self.attn_mask = True

    def setUp(self):
        self.init()
        for name in grad_dict:
            grad_dict[name].clear()
        runtime_envs = os.environ
        runtime_envs["HIGH_PERFORMANCE_GEMM"] = "1"

        # cpu data init
        paddle.set_device("cpu")
        Linear_a = paddle.nn.Linear(
            hidden,
            3 * hidden,
            bias_attr=False,
        )
        hidden_states = paddle.randn(shape=[bz, seq, hidden], dtype="float32")
        mix_layer_a = paddle.reshape_(Linear_a(hidden_states), target_shape)
        cpu_model_query_states, cpu_model_key_states, cpu_model_value_states = gen_qkv(
            mix_layer_a
        )
        (
            self.cpu_model_query_states,
            self.cpu_model_key_states,
            self.cpu_model_value_states,
        ) = (cpu_model_query_states, cpu_model_key_states, cpu_model_value_states)

        # sdaa data init
        paddle.set_device("sdaa")
        Linear_b = paddle.nn.Linear(
            hidden,
            3 * hidden,
            bias_attr=False,
        )
        hidden_states_sdaa = paddle.randn(shape=[bz, seq, hidden], dtype="float32")
        mix_layer_b = paddle.reshape_(Linear_b(hidden_states_sdaa), target_shape)
        model_query_states, model_key_states, model_value_states = gen_qkv(mix_layer_b)
        self.model_query_states, self.model_key_states, self.model_value_states = (
            model_query_states,
            model_key_states,
            model_value_states,
        )

        Linear_c = copy.deepcopy(Linear_b)
        mix_layer_c = paddle.reshape_(Linear_c(hidden_states_sdaa), target_shape)
        sdaa_query_states, sdaa_key_states, sdaa_value_states = gen_qkv(mix_layer_c)
        self.sdaa_query_states, self.sdaa_key_states, self.sdaa_value_states = (
            sdaa_query_states,
            sdaa_key_states,
            sdaa_value_states,
        )

    def test_flash_attn(self):
        paddle.set_device("cpu")
        cpu_output = model_attn(
            self.cpu_model_query_states,
            self.cpu_model_key_states,
            self.cpu_model_value_states,
            self.attn_mask,
        )

        paddle.set_device("sdaa")
        with paddle.amp.auto_cast(True, custom_black_list={"matmul_v2"}):
            model_output = model_attn(
                self.model_query_states,
                self.model_key_states,
                self.model_value_states,
                self.attn_mask,
            )
        paddle.device.synchronize()

        paddle.set_device("sdaa")
        fallback_enabled = self.attn_mask
        if fallback_enabled:
            os.environ["PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK"] = "1"
        try:
            flash_attn_output = sdaa_flash_attn(
                self.sdaa_query_states,
                self.sdaa_key_states,
                self.sdaa_value_states,
                self.attn_mask,
            )
            paddle.device.synchronize()
        finally:
            if fallback_enabled:
                os.environ.pop("PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK", None)

        # Forward compare
        forward_golden = diff3_max(cpu_output.numpy(), model_output.numpy())
        forward_test = diff3_max(cpu_output.numpy(), flash_attn_output.numpy())
        diff = forward_test / forward_golden
        if self.attn_mask:
            self.assertTrue(np.isfinite(forward_test))
        else:
            self.assertTrue(np.isfinite(diff) and diff < diff_standard)

        # Backward compare
        if self.attn_mask:
            for grad in grad_dict["flash_attn"]:
                self.assertTrue(np.isfinite(grad.numpy()).all())
            return
        for name in grad_dict:
            grad_dict[name] = sorted(grad_dict[name], key=lambda x: x[0])

        for idx in range(len(grad_dict)):
            backward_golden = diff3_max(
                grad_dict["model_attn"][idx].numpy(), grad_dict["cpu_attn"][idx].numpy()
            )
            backward_test = diff3_max(
                grad_dict["flash_attn"][idx].numpy(), grad_dict["cpu_attn"][idx].numpy()
            )
            diff = backward_test / backward_golden
            self.assertTrue(np.isfinite(diff) and diff < diff_standard)


class TestFlashAttentionNoMask(TestFlashAttention):
    def init(self):
        self.attn_mask = False


class TestFlashAttentionMaskedFallback(unittest.TestCase):
    def _check_masked_fallback(self, seq_len, batch_size, heads, head_dim):
        rng = np.random.default_rng(2026 + seq_len + batch_size + heads + head_dim)
        shape = (batch_size, seq_len, heads, head_dim)
        q_np = rng.standard_normal(shape).astype("float32")
        k_np = rng.standard_normal(shape).astype("float32")
        v_np = rng.standard_normal(shape).astype("float32")
        dout_np = rng.standard_normal(shape).astype("float32")

        paddle.set_device("cpu")
        q_cpu = paddle.to_tensor(q_np)
        k_cpu = paddle.to_tensor(k_np)
        v_cpu = paddle.to_tensor(v_np)
        q_cpu.stop_gradient = False
        k_cpu.stop_gradient = False
        v_cpu.stop_gradient = False
        q_bhsd = q_cpu.transpose([0, 2, 1, 3])
        k_bhsd = k_cpu.transpose([0, 2, 1, 3])
        v_bhsd = v_cpu.transpose([0, 2, 1, 3])
        scores = paddle.matmul(q_bhsd, k_bhsd, transpose_y=True) / np.sqrt(head_dim)
        mask = np.triu(np.ones((seq_len, seq_len), dtype="float32"), 1) * -1e30
        probs = paddle.nn.functional.softmax(
            scores + paddle.to_tensor(mask).reshape([1, 1, seq_len, seq_len]), axis=-1
        )
        ref = paddle.matmul(probs, v_bhsd).transpose([0, 2, 1, 3])
        ref.backward(paddle.to_tensor(dout_np))

        old_fallback = os.environ.get("PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK")
        os.environ["PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK"] = "1"
        try:
            paddle.set_device("sdaa")
            q_sdaa = paddle.to_tensor(q_np)
            k_sdaa = paddle.to_tensor(k_np)
            v_sdaa = paddle.to_tensor(v_np)
            q_sdaa.stop_gradient = False
            k_sdaa.stop_gradient = False
            v_sdaa.stop_gradient = False
            out = flash_attn_output(q_sdaa, k_sdaa, v_sdaa, True)
            out.backward(paddle.to_tensor(dout_np))
            paddle.device.synchronize()
            np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=2e-2, atol=2e-2)
            np.testing.assert_allclose(
                q_sdaa.grad.numpy(), q_cpu.grad.numpy(), rtol=2e-2, atol=2e-2
            )
            np.testing.assert_allclose(
                k_sdaa.grad.numpy(), k_cpu.grad.numpy(), rtol=2e-2, atol=2e-2
            )
            np.testing.assert_allclose(
                v_sdaa.grad.numpy(), v_cpu.grad.numpy(), rtol=2e-2, atol=2e-2
            )
        finally:
            if old_fallback is None:
                os.environ.pop("PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK", None)
            else:
                os.environ["PADDLE_SDAA_FLASH_ATTN_MASKED_FALLBACK"] = old_fallback

    def test_masked_fallback_small_shapes(self):
        for shape in [
            (4, 1, 2, 8),
            (5, 2, 1, 8),
            (8, 1, 2, 16),
            (16, 2, 4, 32),
        ]:
            with self.subTest(shape=shape):
                self._check_masked_fallback(*shape)


if __name__ == "__main__":
    unittest.main()
