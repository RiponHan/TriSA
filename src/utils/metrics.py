import torch
from torchmetrics import Metric


class RankingMetrics(Metric):
    """P, NDCG, Recall, F1 at each cutoff, plus full-list MRR.

    The historical DCG@K output names are retained; their values are NDCG.
    Each cutoff uses its own topk call to preserve tie handling.
    """
    def __init__(self, cutoffs=(5, 10, 15)):
        super().__init__()
        self.cutoffs = tuple(cutoffs)
        self.names = tuple(
            f"{name}@{k}" for name in ("P", "DCG", "Recall", "F1") for k in self.cutoffs
        ) + ("MRR",)
        self.add_state("sums", default=torch.zeros(len(self.names)), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        target = target.to(preds.device)
        relevant = target.sum(dim=1).clamp_min(1.0)
        values = {}
        for k in self.cutoffs:
            indices = torch.topk(preds, k, dim=1).indices
            ranked = target.gather(1, indices)
            hits = ranked.sum(dim=1)
            precision = hits / k
            recall = hits / relevant
            discounts = torch.log2(
                torch.arange(2, k + 2, device=preds.device, dtype=preds.dtype)
            ).unsqueeze(0)
            ideal = torch.topk(target, k, dim=1).values
            ndcg = (ranked / discounts).sum(dim=1) / (ideal / discounts).sum(dim=1).clamp_min(1e-8)
            values[f"P@{k}"] = precision
            values[f"DCG@{k}"] = ndcg
            values[f"Recall@{k}"] = recall
            values[f"F1@{k}"] = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        indices = torch.sort(preds, dim=1, descending=True).indices
        ranked = target.gather(1, indices)
        ranks = torch.arange(1, ranked.size(1) + 1, device=preds.device).float().unsqueeze(0)
        values["MRR"] = (ranked * (1.0 / ranks)).max(dim=1).values
        self.sums += torch.stack([values[name].sum() for name in self.names])
        self.total += preds.size(0)

    def compute(self):
        return {name: self.sums[i] / self.total for i, name in enumerate(self.names)}
