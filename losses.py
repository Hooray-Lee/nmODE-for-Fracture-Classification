import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MultiLabelSoftMarginLoss()
    def forward(self, input, target, **kwargs):
        return self.loss(input, target)

class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target, **kwargs):
        return self.loss(input, target)

class AsymmetricFocalMarginLoss(nn.Module):

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos 
        self.gamma_neg = gamma_neg 
        self.reduction = reduction

    def forward(self, input, target,** kwargs):
        p = torch.sigmoid(input) 
        eps = 1e-8  

        pos_loss = - (1 - p) **self.gamma_pos * target * torch.log(p + eps)
        neg_loss = - p** self.gamma_neg * (1 - target) * torch.log(1 - p + eps)
        
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    def __init__(self, base_loss='softmargin', alpha=1.0, beta=1.0, gamma=1.0,
                 gamma_pos=1.0, gamma_neg=4.0): 
        super().__init__()
        if base_loss == 'softmargin':
            self.base_loss = nn.MultiLabelSoftMarginLoss()
        elif base_loss == 'bce':
            self.base_loss = nn.BCEWithLogitsLoss()
        elif base_loss == 'asymmetric_focal': 
            self.base_loss = AsymmetricFocalMarginLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg)
        else:
            raise ValueError('Unknown base_loss, choose from ["softmargin", "bce", "asymmetric_focal"]')
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target, hs=None, query_embed=None, query_embed_nce=None, return_parts=False):

        loss1 = self.base_loss(input, target)
        if hs is not None:
            decoder_nce = torch.stack([info_nce_loss(h) for h in hs]).mean()
        else:
            decoder_nce = torch.tensor(0.0, device=loss1.device, dtype=loss1.dtype)
        if query_embed is not None:
            query_embed_nce = info_nce_loss(query_embed)
        else:
            query_embed_nce = torch.tensor(0.0, device=loss1.device, dtype=loss1.dtype)

        total_loss = self.alpha * loss1 + self.beta * query_embed_nce + self.gamma * decoder_nce

        if return_parts:
            return total_loss, loss1.detach(), query_embed_nce.detach(), decoder_nce.detach()
        else:
            return total_loss

def info_nce_loss(query, temperature=0.07):
    q = F.normalize(query, dim=-1)
    sim_matrix = torch.matmul(q, q.t()) 
    logits = sim_matrix / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = F.cross_entropy(logits, labels)
    return loss