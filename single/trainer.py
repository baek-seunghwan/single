import math
import torch


class Optim:
    """Simple optimizer wrapper exposing `.optimizer` and `.step()`.

    - `optimizer` is the underlying torch optimizer (used by schedulers).
    - `step()` computes gradient norm, applies optional clipping, and steps.
    """

    def __init__(self, params, optim_method, lr, clip, lr_decay=0.0):
        self.clip = clip
        method = (optim_method or 'adam').lower()
        if method == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, weight_decay=lr_decay)
        elif method == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=lr_decay)
        else:
            # default to Adam
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=lr_decay)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        total_norm_sq = 0.0
        found = False
        for group in self.optimizer.param_groups:
            for p in group.get('params', []):
                if p.grad is None:
                    continue
                found = True
                param_norm = p.grad.data.norm(2)
                total_norm_sq += (param_norm.item() ** 2)

        total_norm = math.sqrt(total_norm_sq) if found else 0.0

        if self.clip and self.clip > 0:
            params_with_grad = [p for g in self.optimizer.param_groups for p in g.get('params', []) if p.grad is not None]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, self.clip)

        self.optimizer.step()
        return total_norm
