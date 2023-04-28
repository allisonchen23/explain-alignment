import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(logits, target):
    loss = torch.nn.CrossEntropyLoss()
    return loss(logits, target)

def contrastive_cross_entropy(logits, target, margin=0.0):
    """
    A special loss that is similar to crossentropy but becomes exactly zero if
    logp(target) >= max(logp(all_excluding_target)) + margin
    Used for classification edits
    """
    logp = F.log_softmax(logits, dim=-1)
    target_one_hot = F.one_hot(target, num_classes=logp.shape[-1])
    logp_target = (logp * target_one_hot.to(logits.dtype)).sum(-1)
    logp_others = torch.where(target_one_hot.to(torch.uint8), torch.full_like(logp, -float('inf')), logp)
    return F.relu(margin + logp_others.max(dim=-1)[0] - logp_target).mean()