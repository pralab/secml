import torch
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor, as_carray
from torch.nn import CrossEntropyLoss


class DeepfoolLoss:
    def _adv_objective_function(self, x):
        if self.loss == "logits":
            loss_fn = self.logits_diff
        elif self.loss == "crossentropy":
            loss_fn = self.ce_diff

        losses_all_pts = torch.empty(x.shape[0])
        grads_all_pts = torch.empty(x.shape)
        for pt in range(x.shape[0]):
            x_pt = x[pt, :].detach()
            diffs_and_grads = [loss_fn(x_pt, k) for k in range(1, self.candidates)]
            diffs = [l[0] for l in diffs_and_grads]
            grads = [l[1] for l in diffs_and_grads]
            losses = torch.stack(diffs, dim=1)
            grads = torch.stack(grads, dim=0)
            assert losses.shape == (1, self.candidates - 1)
            assert grads.shape == (self.candidates - 1, x.shape[1])

            # calculate the distances
            distances = self.get_distances(losses, grads)
            assert distances.shape == (1, self.candidates - 1)
            # determine the best directions
            best = distances.argmin(dim=-1)
            losses = losses[0, best]
            grads = grads[0, best]
            losses_all_pts[pt] = losses
            grads_all_pts[pt, ...] = grads
        return losses_all_pts, grads_all_pts

    def logits_diff(self, x, k):
        x.requires_grad = True
        logits = self._pytorch_model_wrapper(x)
        classes = logits.argsort(dim=-1).flip(dims=(-1,))
        i0 = classes[:, 0]
        ik = classes[:, k]
        l0 = logits[0, i0]
        lk = logits[0, ik]
        loss = lk - l0
        loss.backward()
        grad = x.grad
        return loss, grad

    def ce_diff(self, x, k):
        x.requires_grad = True
        logits = self._pytorch_model_wrapper(x)
        classes = logits.argsort(dim=-1).flip(dims=(-1,))
        i0 = classes[:, 0]
        ik = classes[:, k]
        l0 = -CrossEntropyLoss(reduction='none')(logits, i0)
        lk = -CrossEntropyLoss(reduction='none')(logits, ik)
        loss = lk - l0
        loss.backward()
        grad = x.grad
        return loss, grad


    def get_distances(self, losses, grads):
        if self.distance == 'l2':
            return abs(losses) / ((grads.view(grads.shape[0], -1)).norm(p=2, dim=-1) + 1e-8)
        elif self.distance == 'linf':
            return abs(losses) / ((grads.view(grads.shape[0], -1)).abs().sum(dim=-1) + 1e-8)
        else:
            raise NotImplementedError



    def objective_function_gradient(self, x):
        """
        Deepfool uses the gradient to find the closest class.
        For this reason, if we need the gradient, the function
        attempts to run the backward twice, which can be avoided
        if we take care of saving the gradient at the first
        pass.
        """
        x_t = as_tensor(x).detach()
        x_t.requires_grad_()
        loss, gradient = self._adv_objective_function(x_t)
        return as_carray(gradient)

    def objective_function(self, x):
        """
        Accordingly, we should also return only the first
        returned value of the function.
        """
        return as_carray(self._adv_objective_function(as_tensor(x))[0])