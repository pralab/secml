import torch

from secml.adv.attacks.evasion.foolbox.losses.cw_loss import CWLoss


class EADLoss(CWLoss):
    def _adv_objective_function(self, x):
        cw_loss = super(EADLoss, self)._adv_objective_function(x)
        l1_norm = torch.norm(self._x0 - x.flatten(start_dim=1), dim=1, p=1)
        return cw_loss + self.regularization * l1_norm
