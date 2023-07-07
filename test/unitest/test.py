import unittest
import mindspore
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastreid.modeling.ops import MetaConv2d  # 请替换为您的主干网络模型
# from fastreid.modeling.backbones.meta_dynamic_router_resnet import MainNetwork  # 请替换为您的主干网络模型

class TestBackbones(unittest.TestCase):
    def setUp(self):
        print("=>", self._testMethodName[5:])

    def tearDown(self):
        pass

    def test_MetaConv2d(self):
        # 创建一个示例输入张量
        input_tensor = mindspore.ops.randn(1, 3, 32, 32)  # 替换为您的输入张量形状
        model = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_tensor = model(input_tensor)
        expected_shape = (1, 64, 16, 16)
        self.assertEqual(output_tensor.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()