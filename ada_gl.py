import torch
from torch.optim import Optimizer
from math import gamma

class AdaGL(Optimizer):
    def __init__(self, params, lr=0.001, alpha=1.5, beta1=0.9, beta2=0.999, epsilon=1e-8, history_size=10):
        """
        Initializes the AdaGL optimizer.
        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate (default: 0.001).
            alpha (float): Fractional order (default: 1.5).
            beta1 (float): Decay rate for first moment (default: 0.9).
            beta2 (float): Decay rate for second moment (default: 0.999).
            epsilon (float): Numerical stability constant (default: 1e-8).
            history_size (int): Maximum size of the gradient history (default: 10).
        """
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("Invalid alpha: must be in [0, 2]")
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: must be > 0")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1: must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: must be in [0, 1)")

        defaults = dict(lr=lr, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon, history_size=history_size)
        super(AdaGL, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            history_size = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment
                    state['grad_history'] = torch.zeros((history_size,) + grad.shape, device=grad.device)  # Preallocate

                m, v, grad_history = state['m'], state['v'], state['grad_history']
                state['step'] += 1
                step = state['step']

                # Update gradient history
                grad_history = torch.roll(grad_history, shifts=1, dims=0)
                grad_history[0] = grad.clone()

                # Compute G–L fractional gradient
                fractional_grad = torch.zeros_like(grad)
                for j in range(min(step, history_size)):
                    coeff = ((-1)**j) * gamma(alpha + 1) / (gamma(j + 1) * gamma(alpha - j + 1))
                    fractional_grad += coeff * grad_history[j]

                # Update biased first and second moments
                m.mul_(beta1).add_(1 - beta1, fractional_grad)
                v.mul_(beta2).addcmul_(1 - beta2, fractional_grad, fractional_grad)

                # Bias corrections
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                # Step size control coefficient
                if step > 1:
                    gradient_change = grad - grad_history[1]
                else:
                    gradient_change = grad
                step_size_control = 1.1 - 0.5 / (1 + gradient_change.abs().mean())

                # Parameter update
                p.data.addcdiv_(-lr * step_size_control, m_hat, (v_hat.sqrt() + epsilon))

        return loss
