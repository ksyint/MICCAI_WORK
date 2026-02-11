import torch
import torch.nn as nn
import torch.nn.functional as F
from .focal_loss import FocalLoss, ClassBalancedFocalLoss


class CombinedVQALoss(nn.Module):
    def __init__(self, focal_gamma=2.0, lambda_cls=1.0, lambda_gen=1.0,
                 lambda_reg=1e-4, num_classes=None):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_gen = lambda_gen
        self.lambda_reg = lambda_reg
        if num_classes:
            self.focal_loss = ClassBalancedFocalLoss(gamma=focal_gamma, num_classes=num_classes)
        else:
            self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.gen_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, cls_logits=None, cls_targets=None,
                gen_logits=None, gen_targets=None, lora_params=None):
        total_loss = torch.tensor(0.0, device=self._get_device(cls_logits, gen_logits))
        loss_dict = {}
        if cls_logits is not None and cls_targets is not None:
            cls_loss = self.focal_loss(cls_logits, cls_targets)
            total_loss = total_loss + self.lambda_cls * cls_loss
            loss_dict["cls_loss"] = cls_loss.item()
        if gen_logits is not None and gen_targets is not None:
            if gen_logits.dim() == 3:
                gen_logits = gen_logits.reshape(-1, gen_logits.size(-1))
                gen_targets = gen_targets.reshape(-1)
            gen_loss = self.gen_loss(gen_logits, gen_targets)
            total_loss = total_loss + self.lambda_gen * gen_loss
            loss_dict["gen_loss"] = gen_loss.item()
        if lora_params is not None and self.lambda_reg > 0:
            reg_loss = torch.tensor(0.0, device=total_loss.device)
            for p in lora_params:
                if p.requires_grad:
                    reg_loss = reg_loss + torch.norm(p, 2) ** 2
            total_loss = total_loss + self.lambda_reg * reg_loss
            loss_dict["reg_loss"] = reg_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def _get_device(self, *tensors):
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


class SigLIPContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1, temperature=0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, student_features, teacher_features):
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        logits = torch.matmul(student_features, teacher_features.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        targets = 2 * torch.eye(logits.shape[0], device=logits.device) - 1
        loss = -torch.sum(F.logsigmoid(targets * logits + self.margin)) / logits.shape[0]
        return loss
