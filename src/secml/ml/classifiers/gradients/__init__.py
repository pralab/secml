from .mixin_classifier_gradient import CClassifierGradientMixin
from .mixin_classifier_gradient_linear import CClassifierGradientLinearMixin
from .mixin_classifier_gradient_logistic import \
    CClassifierGradientLogisticMixin
from .mixin_classifier_gradient_ridge import CClassifierGradientRidgeMixin
from .mixin_classifier_gradient_svm import CClassifierGradientSVMMixin
from .mixin_classifier_gradient_sgd import CClassifierGradientSGDMixin
from .mixin_classifier_gradient_dnn import CClassifierGradientDNNMixin

try:
    import torch
except ImportError:
    pass  # pytorch is an extra component
else:
    from .mixin_classifier_gradient_pytorch import CClassifierGradientPyTorchMixin
