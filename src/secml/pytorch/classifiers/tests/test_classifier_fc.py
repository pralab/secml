from secml.utils import CUnitTest

import random
import torch

from secml.pytorch.classifiers import CTorchClassifierFullyConnected

use_cuda = torch.cuda.is_available()
print "Using CUDA: ", use_cuda

# Random seed
random.seed(999)
torch.manual_seed(999)
if use_cuda:
    torch.cuda.manual_seed_all(999)


class TestCTorchClassifierDenseNetCifar(CUnitTest):

    def setUp(self):

        self.clf = CTorchClassifierFullyConnected()
        self.clf.verbose = 2

    # TODO: ADD TEST FOR TRAINING
    # TODO: ADD TEST FOR CLASSIFICATION

    def test_model_params(self):
        """Test for model parameters shape."""
        for name, param in self.clf._model.named_parameters():
            if name.endswith(".weight"):
                # We expect weights to be stored as 2D tensors
                self.assertEqual(2, len(param.shape))
            elif name.endswith(".bias"):
                # We expect biases to be stored as 1D tensors
                self.assertEqual(1, len(param.shape))

    def test_optimizer_params(self):
        """Testing optimizer parameters setting."""
        self.logger.info("Testing parameter `weight_decay`")
        clf = CTorchClassifierFullyConnected(weight_decay=1e-2)

        self.assertEqual(1e-2, clf._optimizer.defaults['weight_decay'])

        clf.weight_decay = 1e-4
        self.assertEqual(1e-4, clf._optimizer.defaults['weight_decay'])

    def test_save_load_state(self):
        """Test for load_state using state_dict."""
        lr_default = 1e-2
        lr = 30

        # Initializing a CLF with an unusual parameter value
        self.clf = CTorchClassifierFullyConnected(learning_rate=lr)
        self.clf.verbose = 2

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])

        state = self.clf.state_dict()

        # Initializing the clf again using default parameters
        self.clf = CTorchClassifierFullyConnected()
        self.clf.verbose = 2

        self.assertEqual(lr_default, self.clf.learning_rate)
        self.assertEqual(lr_default, self.clf._optimizer.defaults['lr'])

        self.clf.load_state(state)

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])


if __name__ == '__main__':
    CUnitTest.main()
