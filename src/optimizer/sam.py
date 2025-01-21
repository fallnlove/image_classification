import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, params, base_optim, rho=0.05, eps=1e-9, **kwargs):
        super(SAM, self).__init__(params, kwargs)

        self.base_optim = base_optim(self.param_groups, **kwargs)
        self.cnt = 0
        self.rho = rho
        self.eps = eps

    @torch.no_grad()
    def step(self):
        if self.cnt % 2 == 0:
            gradient_norm = self._get_grad_norm()
            self.history = [
                p.data.clone()
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            for group in self.param_groups:
                l = self.rho / (gradient_norm + self.eps)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    p.add_(p.grad * l)
        else:
            i = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data = self.history[i]
                    i += 1

            self.base_optim.step()

        self.cnt += 1

    @torch.no_grad()
    def _get_grad_norm(self):
        """
        Get gradient norm of the model.
        """

        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        return total_norm
