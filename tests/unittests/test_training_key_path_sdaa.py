import unittest

import numpy as np
import paddle


class TestTrainingKeyPathSDAA(unittest.TestCase):
    def _run_one_step(self, device):
        paddle.set_device(device)
        paddle.seed(2022)
        np.random.seed(2022)

        class Net(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = paddle.nn.Conv2D(1, 4, 3, padding=1)
                self.bn = paddle.nn.BatchNorm2D(4)
                self.ln = paddle.nn.LayerNorm([4, 8, 8])
                self.embedding = paddle.nn.Embedding(32, 8)
                self.fc = paddle.nn.Linear(4 * 8 * 8 + 8, 5)

            def forward(self, image, token):
                x = self.conv(image)
                x = self.bn(x)
                x = paddle.nn.functional.relu(x)
                x = self.ln(x)
                x = paddle.flatten(x, start_axis=1)
                token = self.embedding(token).mean(axis=1)
                return self.fc(paddle.concat([x, token], axis=1))

        net = Net()
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=net.parameters())
        image = paddle.randn([4, 1, 8, 8])
        token = paddle.to_tensor(np.random.randint(0, 32, [4, 6]), dtype="int64")
        label = paddle.to_tensor(np.random.randint(0, 5, [4]), dtype="int32")
        if device == "sdaa":
            token = token.to("sdaa")
            label = label.to("sdaa")
        logits = net(image, token)
        loss = paddle.nn.functional.cross_entropy(logits, label).mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize() if device == "sdaa" else None
        self.assertTrue(bool(paddle.isfinite(loss).item()))
        self.assertTrue(any(param.grad is not None for param in net.parameters()))
        return float(loss)

    def test_sdaa_forward_backward_optimizer(self):
        loss = self._run_one_step("sdaa")
        self.assertTrue(np.isfinite(loss))

    def test_cpu_reference_forward_backward_optimizer(self):
        loss = self._run_one_step("cpu")
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
