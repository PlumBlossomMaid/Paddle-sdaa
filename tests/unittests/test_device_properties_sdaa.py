# BSD 3-Clause License Copyright (c) 2026, Tecorigin Co., Ltd.

import unittest

import paddle
import paddle_sdaa


class TestDevicePropertiesSDAA(unittest.TestCase):
    def test_device_properties(self):
        paddle.set_device("sdaa")
        properties = paddle.device.get_device_properties()
        self.assertEqual(properties.name, "/dev/tcaicard0")
        self.assertEqual(properties.major, 61440)
        self.assertEqual(properties.minor, 256)
        self.assertGreater(properties.total_memory, 0)
        self.assertGreater(properties.free_memory, 0)
        self.assertEqual(paddle.device.get_device_name(), "/dev/tcaicard0")
        self.assertEqual(paddle.device.get_device_capability(), (61440, 256))
        self.assertEqual(paddle.device.get_device_properties("sdaa:0").name, "/dev/tcaicard0")


if __name__ == "__main__":
    unittest.main()
