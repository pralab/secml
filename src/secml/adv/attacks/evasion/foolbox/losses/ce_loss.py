from torch.nn import CrossEntropyLoss
import torch


class CELoss:
    def _adv_objective_function(self, x):
        loss = CrossEntropyLoss(reduction='none')
        scores = self._pytorch_model_wrapper(x)
        target = torch.empty(scores.shape[0], dtype=torch.long)

        if self.y_target is not None:
            target[:] = self.y_target
        else:  # indiscriminate attack
            target[:] = self._y0

        total_loss = loss(scores, target)
        return total_loss if self.y_target is not None else -total_loss

