import torch

class LogitsLoss:

    def _adv_objective_function(self, x):
        if self._x0 is None:
            raise Exception('Attack not run yet')
        z_predicted = self._pytorch_model_wrapper(x)
        y_target = torch.zeros((z_predicted.shape))

        # destination point label
        if self.y_target is not None:
            y_target[:, self.y_target] = 1
        else:  # indiscriminate attack
            y_target[:, self._y0.long()] = 1

        z_target = (z_predicted * y_target).sum(dim=1)
        second_best_labels = torch.argmax(y_target * torch.min(z_predicted) + z_predicted * (1 - y_target), dim=1)
        z_other = z_predicted[torch.arange(z_predicted.size(0)).long(), second_best_labels]

        if self.y_target is not None:
            # if targeted, optimize for making the target class most likely
            loss = torch.max(z_other - z_target + self.confidence, torch.zeros(x.shape[0], dtype=x.dtype))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = torch.max(z_target - z_other + self.confidence, torch.zeros(x.shape[0], dtype=x.dtype))

        return loss