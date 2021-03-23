from secml.testing import CUnitTest

try:
    import foolbox
    import torch
except ImportError:
    CUnitTest.importskip("foolbox")

import torch
from secml import model_zoo
from secml.adv.attacks.evasion.foolbox.secml_autograd import SecmlLayer, as_carray


class TestSecmlAutograd(CUnitTest):
    def setUp(self):
        self.secml_model = model_zoo.load_model("mnist-svm")
        self.secml_net = model_zoo.load_model("mnist159-cnn")
        self.secml_layer_svm = SecmlLayer(self.secml_model)
        self.secml_layer_torch = SecmlLayer(self.secml_net)
        self.N, self.D_in, self.D_out = 3, 784, 10

    def test_grads_svm(self):
        x = torch.randn(self.N, self.D_in, requires_grad=True)
        logits = self.secml_layer_svm(x)

        random_op = logits.sum()
        random_op.backward()
        torch_grad = x.grad

        secml_grad = self.secml_model.backward(as_carray(torch.ones(size=(self.D_out,)))) * self.N

        self.assertAlmostEqual(torch_grad.sum().item(), secml_grad.sum(), places=3)

    def test_grads_torch(self):
        x = torch.randn(self.N, 1, 28, 28, requires_grad=True)
        logits = self.secml_layer_torch(x)

        random_op = logits.sum()
        random_op.backward()
        torch_grad = x.grad

        self.assertTrue(torch_grad.sum().item())
