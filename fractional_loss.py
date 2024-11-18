import torch
import torch.nn.functional as F

class FractionalSmoothLoss:
    def __init__(self, base_loss='mse', alpha=1.5, smooth_coeff=0.1):
        """
        Initializes the FractionalSmoothLoss.
        Args:
            base_loss (str): The base loss type ('mse', 'cross_entropy').
            alpha (float): Fractional order for gradient smoothing.
            smooth_coeff (float): Coefficient for the gradient regularization term.
        """
        if base_loss not in ['mse', 'cross_entropy']:
            raise ValueError("Invalid base_loss. Choose 'mse' or 'cross_entropy'.")
        self.base_loss = base_loss
        self.alpha = alpha
        self.smooth_coeff = smooth_coeff

    def __call__(self, predictions, targets, model):
        """
        Computes the custom loss.
        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
            model (torch.nn.Module): The model being trained.
        Returns:
            torch.Tensor: Computed loss.
        """
        # Base loss
        if self.base_loss == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.base_loss == 'cross_entropy':
            loss = F.cross_entropy(predictions, targets)

        # Gradient regularization
        grad_norms = 0
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = (grad ** self.alpha).abs().mean()  # Fractional gradient penalty
                grad_norms += grad_norm
        
        # Combine losses
        total_loss = loss + self.smooth_coeff * grad_norms
        return total_loss
