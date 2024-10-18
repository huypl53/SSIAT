import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import logging


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type="cosface", eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface", "crossentropy"]
        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == "crossentropy":
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == "cosface":
                numerator = self.s * (
                    torch.diagonal(wf.transpose(0, 1)[labels]) - self.m
                )
            if self.loss_type == "arcface":
                numerator = self.s * torch.cos(
                    torch.acos(
                        torch.clamp(
                            torch.diagonal(wf.transpose(0, 1)[labels]),
                            -1.0 + self.eps,
                            1 - self.eps,
                        )
                    )
                    + self.m
                )
            if self.loss_type == "sphereface":
                numerator = self.s * torch.cos(
                    self.m
                    * torch.acos(
                        torch.clamp(
                            torch.diagonal(wf.transpose(0, 1)[labels]),
                            -1.0 + self.eps,
                            1 - self.eps,
                        )
                    )
                )

            excl = torch.cat(
                [
                    torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0)
                    for i, y in enumerate(labels)
                ],
                dim=0,
            )
            denominator = torch.exp(numerator) + torch.sum(
                torch.exp(self.s * excl), dim=1
            )
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


# class SupervisedContrastiveLoss(torch.nn.Module):
#     def __init__(self, temperature=0.05):
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, features, targets):
#         B = features.size(0)
#         features = F.normalize(features, p=2, dim=1)
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature
#         positive_mask = targets.unsqueeze(1) == targets.unsqueeze(0)
#         mask_self = torch.eye(B, dtype=torch.bool).to(features.device)
#         positive_mask = positive_mask & ~mask_self
#         exp_sim = torch.exp(similarity_matrix)
#         pos_sim = exp_sim * positive_mask.float()
#         # pos_sum = pos_sim.sum(dim=1)
#         # denom_sum = exp_sim.sum(dim=1) - torch.exp(similarity_matrix.diag())
#         denom_sum = exp_sim.sum(dim=1) - exp_sim.diag()
#         logits = torch.log( pos_sim[pos_sim != 0] / denom_sum )
#         loss = - logits.sum(dim=1).mean()
#         # loss = -torch.log(pos_sum[pos_sum != 0] / denom_sum[pos_sum != 0])
#         if torch.isinf(loss).any():
#             logging.warning("`inf` detected in contrative loss. Ignoring.")
#             # loss = torch.where(torch.isinf(loss), torch.tensor(1e8).to(loss.device), loss)
#             loss = torch.where(torch.isinf(loss), torch.tensor(0).to(loss.device), loss)
#         return loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        logits = torch.matmul(features, features.T)

        # fix cosine similarity function, its numerator should be the dot product of original features instead of normalized ones
        norms = torch.norm(features, dim=1, keepdim=True)
        norm_matrix = torch.matmul(norms, norms.T)

        cosine_sim = logits / norm_matrix

        scaled_sim = cosine_sim / self.temperature

        max_val, _ = scaled_sim.max(dim=1, keepdim=True)
        exp_sim = torch.exp(scaled_sim - max_val)
        sum_exp = exp_sim.sum(dim=1, keepdim=True)
        log_prob = scaled_sim - max_val - torch.log(sum_exp)

        batch_size = features.shape[0]
        labels_expanded = labels.expand(batch_size, batch_size)
        mask = labels_expanded.eq(labels_expanded.t())
        mask.fill_diagonal_(False)

        n_positives = mask.sum(1)

        valid_samples = n_positives > 0

        # filter the data points which has no same class samples
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_samples] / n_positives[
            valid_samples
        ]

        loss = -mean_log_prob_pos.mean()

        return loss
