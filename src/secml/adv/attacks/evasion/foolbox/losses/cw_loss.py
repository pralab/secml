import torch

from secml.adv.attacks.evasion.foolbox.losses.logits_loss import LogitsLoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor


class CWLoss(LogitsLoss):

    def _adv_objective_function(self, x):
        if self._x0 is None:
            raise Exception('Attack not run yet')
        l2dist = torch.norm(self._x0 - x.flatten(start_dim=1), dim=1, p=2) ** 2
        
        loss = super(CWLoss, self)._adv_objective_function(x)
        if x.shape[0] == self._consts.shape[0]:
            c = as_tensor(self._consts)
        else:
            c = self._consts[-1].item()
        total_loss = c * loss + l2dist
        return total_loss