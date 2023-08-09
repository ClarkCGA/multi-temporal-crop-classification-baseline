import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    
    This optimizer is designed to improve upon traditional gradient descent methods by 
    considering the sharpness of the loss landscape in addition to the loss value. 
    This is achieved by performing two steps for each update: 
    1. A "lookahead" step, where the parameters are updated as per usual, but with a larger step size.
    2. A "sharpness-aware" step, where the parameters are updated to minimize the loss at the "lookahead" position.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        base_optimizer (torch.optim.Optimizer): The base optimizer to be used, such as SGD or Adam.
        rho (float, optional): The "lookahead" distance. Defaults to 0.05.
        adaptive (bool, optional): If True, the "lookahead" distance is scaled by the parameters. 
                                    Defaults to False.
        **kwargs: Keyword arguments for the base optimizer.

    Methods:
        first_step(zero_grad=False): Performs the "lookahead" step and saves the current parameters.
        second_step(zero_grad=False): Moves the parameters back to their original values and performs 
                                      the "sharpness-aware" update.
        step(closure): Combines first_step and second_step. Closure should be a function that performs 
                       a model update, computes the loss, and returns the loss.
        _grad_norm(): Helper method to compute the norm of the gradients.
        load_state_dict(state_dict): Loads the state dictionary for this optimizer and the base optimizer.

    Example usage:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups