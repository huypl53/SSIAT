import torch
import torch.nn as nn
import torch.nn.functional as F


class OPELoss(nn.Module):
    def __init__(self, temperature=0.5, only_old_proto=False):
        super(OPELoss, self).__init__()
        self.temperature = temperature
        self.only_old_proto = only_old_proto

    def cal_prototype(self, z1, z2, y):
        # start_i = 0
        # cls_num = z1.shape[0]

        uniq_classes = torch.unique(y)
        cls_num = uniq_classes.shape[0]
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((cls_num, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((cls_num, dim), device=z1.device)
        for i, cls_id in enumerate(uniq_classes):
            indices = y == cls_id
            if not any(indices):
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]

            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)

            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        return current_classes_mean_z1, current_classes_mean_z2

    def forward(self, z1, z2, labels, new_cls_num, is_new=False):
        prototype_z1, prototype_z2 = self.cal_prototype(z1, z2, labels)

        if not self.only_old_proto or is_new:
            nonZeroRows = torch.abs(prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = prototype_z2[nonZeroRows]
        else:
            old_prototype_z1 = prototype_z1[: z1.shape[0] - new_cls_num]
            old_prototype_z2 = prototype_z2[: z1.shape[0] - new_cls_num]
            nonZeroRows = torch.abs(old_prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = old_prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = old_prototype_z2[nonZeroRows]

        if not nonZero_prototype_z1.numel() or not nonZero_prototype_z2.numel():
            return None, None, None

        nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
        nonZero_prototype_z2 = F.normalize(nonZero_prototype_z2)

        device = nonZero_prototype_z1.device

        class_num = nonZero_prototype_z1.size(0)
        z = torch.cat((nonZero_prototype_z1, nonZero_prototype_z2), dim=0)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = torch.zeros(
            (2 * class_num, 2 * class_num), dtype=torch.bool, device=device
        )
        pos_mask[:, class_num:].fill_diagonal_(True)
        pos_mask[class_num:, :].fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss, prototype_z1, prototype_z2
