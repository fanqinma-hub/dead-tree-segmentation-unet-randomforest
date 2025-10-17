import torch

@torch.no_grad()
def miou(logits, target, n_classes):
    pred = logits.argmax(1)
    ious = []
    for c in range(n_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float().clamp_min(1)
        ious.append((inter / union).item())
    return sum(ious) / len(ious)

@torch.no_grad()
def precision_recall(logits, target, positive_class=1):
    pred = logits.argmax(1)
    tp = ((pred == positive_class) & (target == positive_class)).sum().float()
    fp = ((pred == positive_class) & (target != positive_class)).sum().float()
    fn = ((pred != positive_class) & (target == positive_class)).sum().float()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    return prec.item(), rec.item()
