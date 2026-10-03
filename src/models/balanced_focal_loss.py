import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedFocalLoss(nn.Module):

    def __init__(
        self,
        class_weights=None,
        gamma=2.0
    ):
        super().__init__()

        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(
                    class_weights,
                    dtype=torch.float32
                )
            )
        else:
            self.class_weights = None

    def forward(
        self,
        inputs,
        targets
    ):

        # Standard cross-entropy WITHOUT class weighting.
        # This gives the correct probability of the true class.
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none"
        )

        # Probability assigned to the correct class.
        pt = torch.exp(-ce_loss)

        # Focal modulation.
        focal_factor = (
            (1.0 - pt) ** self.gamma
        )

        # Apply focal modulation first.
        focal_loss = focal_factor * ce_loss

        # Apply class balancing separately.
        if self.class_weights is not None:

            alpha = self.class_weights[
                targets
            ]

            focal_loss = alpha * focal_loss

        return focal_loss.mean()