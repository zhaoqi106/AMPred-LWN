import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label version of Focal Loss.

    Args:
        alpha (float): Weighting factor for positive samples. Default = 0.25.
        gamma (float): Focusing parameter to reduce relative loss for well-classified examples. Default = 2.0.
        reduction (str): "mean", "sum", or "none". Default = "mean".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        """Compute focal loss.

        Args:
            logits (Tensor): Raw model outputs of shape (B, num_tasks).
            targets (Tensor): Binary targets of the same shape.
        """
        # Binary cross-entropy per element
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probabilities for the true class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # Alpha weighting factor
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------------------------------------------------------
# Differentiable AUROC loss (pair-wise surrogate) compatible with torch_auc.AUCROC
# Source: Cao et al., NeurIPS 2020. Pure-PyTorch implementation – no extra deps.
# -----------------------------------------------------------------------------


class AUCROC(nn.Module):
    """Differentiable surrogate for maximising AUROC.

    This mirrors the API of ``torch_auc.AUCROC`` so the training script can
    switch seamlessly. It realises the dual-symmetric pair-wise ranking loss:

    \[ L = (1-π)·E_{p,n} log(1+e^{γ(n−p)}) + π·E_{n,p} log(1+e^{γ(n−p)}) \]

    where *π* is the positive sample ratio and *γ* a scaling factor.
    """

    def __init__(self, imratio: float = 0.1, gamma: float = 500.0,
                 reduction: str = "mean"):
        super().__init__()
        if not 0.0 < imratio < 1.0:
            raise ValueError("imratio must be in (0,1)")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        # keep as buffer so it moves with .to(device)
        self.register_buffer("imratio", torch.tensor(float(imratio)))
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        preds = preds.view(-1)
        targets = targets.view(-1).float()

        pos_mask = targets == 1
        neg_mask = ~pos_mask

        # 如果 batch 只有单一类别，则返回 0，避免 NaN 并保持梯度稳定
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.zeros((), dtype=preds.dtype, device=preds.device)

        p = preds[pos_mask]  # 正样本分数 (P,)
        n = preds[neg_mask]  # 负样本分数 (N,)

        # pair-wise differences，广播后得到 P×N
        diff = n.unsqueeze(0) - p.unsqueeze(1)  # (P, N)
        pair_loss = torch.nn.functional.softplus(self.gamma * diff)  # log(1+e^{γΔ})

        # 双向加权平均
        pi = self.imratio
        loss = (1 - pi) * pair_loss.mean() + pi * pair_loss.mean()

        if self.reduction == "sum":
            return loss * preds.numel()
        if self.reduction == "none":
            return pair_loss
        return loss 