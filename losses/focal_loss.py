import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=0.9999, num_classes=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self._class_counts = None
        self._weights = None

    def update_class_counts(self, class_counts):
        self._class_counts = class_counts
        effective_num = 1.0 - torch.pow(self.beta, class_counts.float())
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(weights)
        self._weights = weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self._weights is not None:
            alpha_t = self._weights.to(inputs.device)[targets]
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
