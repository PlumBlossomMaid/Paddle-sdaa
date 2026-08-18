# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer


class ReshapeHelp(nn.Layer):
    def forward(self, x):
        return x.reshape([-1, 256])


class SDAAAMPModel(nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(192, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.reshape = ReshapeHelp()
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = self.features(x)
        x = paddle.mean(x, axis=[2, 3], keepdim=True)
        x = self.reshape(x)
        return self.loss_fn(self.classifier(x), label)


class SDAAAMPPipe(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        layers = [
            LayerDesc(nn.Conv2D, 1, 64, kernel_size=3, padding=1),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.Conv2D, 64, 192, kernel_size=3, padding=1),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.Conv2D, 192, 256, kernel_size=3, padding=1),
            LayerDesc(nn.ReLU),
            lambda x: paddle.mean(x, axis=[2, 3], keepdim=True),
            LayerDesc(ReshapeHelp),
            LayerDesc(nn.Linear, 256, num_classes),
        ]
        super().__init__(layers=layers, loss_fn=nn.CrossEntropyLoss(), **kwargs)


def set_random_seed(seed, dp_id, rank_id):
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 4
micro_batch_size = 2


class TestDistPPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        set_random_seed(1024, dp_id, dist.get_rank())
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)

        model_a = SDAAAMPModel()
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a,
            grad_clip=grad_clip,
            parameters=model_a.parameters(),
        )
        scaler_a = paddle.amp.GradScaler(init_loss_scaling=2**5)
        parameters = [parameter.numpy() for parameter in model_a.parameters()]

        model_b = SDAAAMPPipe(num_stages=self.pipeline_parallel_size)
        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b,
            grad_clip=grad_clip,
            parameters=model_b.parameters(),
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)
        scaler_b = fleet.distributed_scaler(
            paddle.amp.GradScaler(init_loss_scaling=2**5)
        )

        part_number = len(parameters) // self.pipeline_parallel_size
        for idx, param in enumerate(model_b.parameters()):
            param.set_value(parameters[idx + pp_id * part_number])

        rng = np.random.default_rng(2026)
        for _ in range(2):
            img = paddle.to_tensor(
                rng.standard_normal((batch_size, 1, 8, 8)).astype("float32")
            )
            label = paddle.to_tensor(
                rng.integers(0, 10, size=(batch_size, 1), dtype="int64")
            )
            with paddle.amp.auto_cast():
                loss_a = model_a(img, label)
            scaled_loss = scaler_a.scale(loss_a)
            scaled_loss.backward()
            scaler_a.minimize(optimizer_a, scaled_loss)
            optimizer_a.clear_grad()
            scheduler_a.step()

            with paddle.amp.auto_cast():
                loss_b = model_b.train_batch(
                    [img, label], optimizer_b, scheduler_b, scaler=scaler_b
                )
            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=5e-4)


if __name__ == "__main__":
    unittest.main()
