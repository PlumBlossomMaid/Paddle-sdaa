# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import os

from op_test import OpTest, convert_uint16_to_float
from unittest import mock
import paddle

paddle.enable_static()


def gaussian_wrapper(dtype_=np.uint16):
    def gauss_wrapper(shape, mean, std, seed, dtype=np.uint16, name=None):
        return paddle.tensor.random.gaussian(shape, mean, std, seed, dtype, name)

    return gauss_wrapper


class TestGaussianRandomKernel(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "gaussian_random"
        self.python_api = paddle.tensor.random.gaussian
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "use_mkldnn": False,
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype="float32")}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_customized(self.verify_output, self.place)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype(self.dtype)
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype(self.dtype)
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(hist, hist2, rtol=0, atol=0.02),
            "hist: " + str(hist) + " hist2: " + str(hist2),
        )


class TestGaussianRandomBF16Op(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.op_type = "gaussian_random"
        self.python_api = gaussian_wrapper(dtype_=np.uint16)
        self.__class__.op_type = self.op_type
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": paddle.base.core.VarDesc.VarType.BF16,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype="float32")}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def test_check_output(self):
        self.check_output_with_place_customized(self.verify_output, place=self.place)

    def verify_output(self, outs):
        outs = convert_uint16_to_float(outs)
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.05)


class TestGaussianRandomKernelFP16(TestGaussianRandomKernel):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "gaussian_random"
        self.python_api = paddle.tensor.random.gaussian
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": paddle.float16,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype="float16")}

    def init_dtype(self):
        self.dtype = "float16"


class TestGaussianAssertError(unittest.TestCase):
    def setUp(self):
        import os

        paddle.seed(10)
        os.environ["RANDOM_ALIGN_NV_DEVICE"] = "123"

    def test_assert(self):
        paddle.disable_static()
        x = paddle.ones([10, 10], dtype=paddle.float32)
        self.assertRaises(ValueError, paddle.normal, 0, 1, x.shape)
        paddle.enable_static()


class TestGaussianGPUAlign(unittest.TestCase):
    def setUp(self):
        pass

    @mock.patch.dict(os.environ, {"RANDOM_ALIGN_NV_DEVICE": "v100"})
    def test_output(self):
        paddle.disable_static(place=paddle.CustomPlace("sdaa", 0))
        paddle.seed(10)
        x = paddle.normal(mean=1.0, std=2.0, shape=[10, 10]).numpy()
        self.assertEqual(x.shape, (10, 10))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(np.count_nonzero(x), 0)
        self.assertTrue(np.allclose(x.mean(), 1.0, atol=0.5))
        self.assertTrue(np.allclose(x.std(), 2.0, atol=0.5))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
