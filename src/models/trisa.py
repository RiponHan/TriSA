import json
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from src.models.trisa_backbone import TriSABackbone
from src.utils.metrics import RankingMetrics


class TriSA(pl.LightningModule):
    def __init__(
        self,
        num_mashups: int,
        num_apis: int,
        text_dim: int,
        hide_dim: int,
        layers: int,
        beta: float,
        lambda_cl: float,
        cl_temp: float,
        lr: float,
        weight_decay: float,
        use_tuple_head: bool = True,
        tuple_dim: int = 64,
        lambda_tuple: float = 0.05,
        tuple_alpha_max: float = 0.10,
        tuple_warmup_epochs: int = 3,
        topm_rerank: int = 50,
        max_eval_scenes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        dummy_h = sp.eye(num_apis, dtype=float, format="csr")
        dummy_ui = sp.csr_matrix((num_mashups + num_apis, num_mashups + num_apis))
        self.backbone = TriSABackbone(
            num_mashups,
            num_apis,
            H_co=dummy_h,
            H_cat=dummy_h,
            H_prov=dummy_h,
            uiMat=dummy_ui,
            hide_dim=hide_dim,
            Layers=layers,
        )

        self.m_align = nn.Linear(text_dim, hide_dim)
        self.a_text = nn.Linear(text_dim, hide_dim)

        self.m_struct_proj = nn.Linear(hide_dim, hide_dim)
        self.a_struct_proj = nn.Linear(hide_dim, hide_dim)


        self.text_mlp = nn.Sequential(
            nn.Linear(hide_dim * 2, hide_dim),
            nn.ReLU(),
            nn.Linear(hide_dim, 1),
        )

        # tuple-aware lightweight residual head
        self.provider_emb = nn.Embedding(num_apis + 1, tuple_dim)
        self.scene_emb = nn.Embedding(num_apis + 1, tuple_dim)
        tuple_input_dim = hide_dim * 4 + tuple_dim * 2
        self.tuple_residual = nn.Sequential(
            nn.Linear(tuple_input_dim, hide_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hide_dim, 1),
        )
        nn.init.zeros_(self.tuple_residual[-1].weight)
        nn.init.zeros_(self.tuple_residual[-1].bias)

        self.criterion = nn.BCEWithLogitsLoss()

        self.train_mapping: Dict[int, list] = {}
        self.val_mapping: Dict[int, list] = {}
        self.test_mapping: Dict[int, list] = {}

        self.mashup_text_emb = None
        self.api_text_emb = None
        self._graph_initialized = False
        self._graph_epoch = None
        self._eval_cache_enabled = False
        self._eval_embeddings = None
        self._eval_graph_revision = None

        # tuple metadata from datamodule
        self.id2provider: List[str] = []
        self.id2scene: List[str] = []
        self.api_scene_lists: List[List[int]] = []
        self.register_buffer("api_provider_ids", torch.zeros(num_apis, dtype=torch.long))

        self.val_metrics = RankingMetrics()
        self.test_metrics = RankingMetrics()


    def _info_nce(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        logits = torch.matmul(z1, z2.t()) / self.hparams.cl_temp
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    def _tuple_alpha(self) -> float:
        if not self.hparams.use_tuple_head:
            return 0.0
        warm = max(int(self.hparams.tuple_warmup_epochs), 1)
        ratio = min(1.0, float(self.current_epoch + 1) / float(warm))
        return float(self.hparams.tuple_alpha_max) * ratio

    def _initialize_runtime_data(self):
        dm = self.trainer.datamodule
        if dm is None:
            raise RuntimeError("TriSA requires a datamodule for text features and API metadata.")
        self.mashup_text_emb = dm.mashup_text_emb.to(self.device)
        self.api_text_emb = dm.api_text_emb.to(self.device)
        if tuple(self.mashup_text_emb.shape) != (self.hparams.num_mashups, self.hparams.text_dim):
            raise ValueError("Mashup text feature shape does not match the model configuration.")
        if tuple(self.api_text_emb.shape) != (self.hparams.num_apis, self.hparams.text_dim):
            raise ValueError("API text feature shape does not match the model configuration.")

        self.train_mapping = dm.train_mapping
        self.val_mapping = dm.val_mapping
        self.test_mapping = dm.test_mapping

        self.api_provider_ids = torch.as_tensor(dm.api_provider_ids, dtype=torch.long, device=self.device)
        self.api_scene_lists = [list(x) if len(x) > 0 else [0] for x in dm.api_scene_lists]
        self.id2provider = list(getattr(dm, "id2provider", []))
        self.id2scene = list(getattr(dm, "id2scene", []))

        # A loaded checkpoint owns its graphs. setup() may have sampled new ones.
        if not self._graph_initialized:
            self.backbone.update_mats(dm.H_co, dm.H_cat, dm.H_prov, dm.uiMat)
            self._graph_initialized = True
        self.val_metrics.to(self.device)
        self.test_metrics.to(self.device)

    def on_fit_start(self):
        self._clear_eval_cache()
        self._initialize_runtime_data()

    def on_validation_start(self):
        self._initialize_runtime_data()
        self._clear_eval_cache()
        self._eval_cache_enabled = True

    def on_test_start(self):
        self._initialize_runtime_data()
        self._clear_eval_cache()
        self._eval_cache_enabled = True

    def _clear_eval_cache(self):
        self._eval_cache_enabled = False
        self._eval_embeddings = None
        self._eval_graph_revision = None

    def on_validation_end(self):
        self._clear_eval_cache()

    def on_test_end(self):
        self._clear_eval_cache()

    def on_save_checkpoint(self, checkpoint):
        if not self._graph_initialized:
            raise RuntimeError("Cannot save TriSA before its training graphs are initialized.")
        graph_state = self.backbone.graph_state_dict()
        graph_state["epoch"] = self._graph_epoch
        checkpoint["trisa_graph_state"] = graph_state

    def on_load_checkpoint(self, checkpoint):
        self._clear_eval_cache()
        if "trisa_graph_state" not in checkpoint:
            raise RuntimeError(
                "This TriSA checkpoint has no graph snapshot. Its original sampled graphs "
                "cannot be recovered from weights alone. Retrain with graph checkpointing "
                "enabled to obtain a matching weights-and-graphs checkpoint."
            )
        graph_state = checkpoint["trisa_graph_state"]
        self.backbone.load_graph_state_dict(graph_state)
        self._graph_initialized = True
        self._graph_epoch = graph_state.get("epoch")

    def on_train_epoch_start(self):
        self._clear_eval_cache()
        dm = self.trainer.datamodule
        if hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)
        dm.refresh_hypergraph()
        self.backbone.update_mats(dm.H_co, dm.H_cat, dm.H_prov, dm.uiMat)
        self._graph_initialized = True
        self._graph_epoch = int(self.current_epoch)

    def _compute_embeddings(self):
        cache_allowed = self._eval_cache_enabled and not self.training and not torch.is_grad_enabled()
        graph_revision = self.backbone.core.graph_revision
        if cache_allowed and self._eval_embeddings is not None and self._eval_graph_revision == graph_revision:
            return self._eval_embeddings
        m_struct, a_struct = self.backbone()
        m_struct = m_struct.to(self.device)
        a_struct = a_struct.to(self.device)

        m_align = self.m_align(self.mashup_text_emb)
        a_text = self.a_text(self.api_text_emb)

        m_struct_proj = self.m_struct_proj(m_struct)
        a_struct_proj = self.a_struct_proj(a_struct)

        embeddings = (m_align, a_text, m_struct_proj, a_struct_proj)
        if cache_allowed:
            self._eval_embeddings = embeddings
            self._eval_graph_revision = graph_revision
        return embeddings

    def _item_scores(self, m_id, a_id, m_align, a_text, m_struct_proj, a_struct_proj):
        text = self.text_mlp(torch.cat([m_align[m_id], a_text[a_id]], dim=1)).squeeze(-1)
        struct = (m_struct_proj[m_id] * a_struct_proj[a_id]).sum(dim=1)
        return self.hparams.beta * text + (1.0 - self.hparams.beta) * struct


    def _tuple_delta(self, m_id, a_id, provider_id, scene_id, m_align, a_text, m_struct_proj, a_struct_proj):
        feat = torch.cat(
            [
                m_align[m_id],
                m_struct_proj[m_id],
                a_text[a_id],
                a_struct_proj[a_id],
                self.provider_emb(provider_id),
                self.scene_emb(scene_id),
            ],
            dim=1,
        )
        return self.tuple_residual(feat).squeeze(-1)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            m_id = batch["m_id"].long().to(self.device)
            pos_a = batch["pos_a"].long().to(self.device)
            neg_a = batch["neg_a"].long().to(self.device)
            pos_provider_id = batch["pos_provider_id"].long().to(self.device)
            pos_scene_id = batch["pos_scene_id"].long().to(self.device)
            neg_scene_same_item_id = batch["neg_scene_same_item_id"].long().to(self.device)
            neg_item_same_scene_id = batch["neg_item_same_scene_id"].long().to(self.device)
        else:
            # backward compatibility: old dataloader without tuple metadata
            m_id, pos_a, neg_a = batch
            m_id = m_id.long().to(self.device)
            pos_a = pos_a.long().to(self.device)
            neg_a = neg_a.long().to(self.device)
            pos_provider_id = self.api_provider_ids[pos_a]
            pos_scene_id = torch.zeros_like(pos_a)
            neg_scene_same_item_id = torch.zeros_like(pos_a)
            neg_item_same_scene_id = neg_a[:, 0]

        m_align, a_text, m_struct_proj, a_struct_proj = self._compute_embeddings()

        pos_score = self._item_scores(m_id, pos_a, m_align, a_text, m_struct_proj, a_struct_proj)

        neg_m = m_align[m_id].unsqueeze(1).expand(-1, neg_a.size(1), -1)
        neg_a_text = a_text[neg_a]
        neg_text = self.text_mlp(torch.cat([neg_m, neg_a_text], dim=2)).squeeze(-1)
        neg_struct = (m_struct_proj[m_id].unsqueeze(1) * a_struct_proj[neg_a]).sum(dim=2)
        neg_score = self.hparams.beta * neg_text + (1.0 - self.hparams.beta) * neg_struct


        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score).view(-1),
        ])
        scores = torch.cat([pos_score, neg_score.view(-1)])
        loss_bce = self.criterion(scores, labels)

        uniq_m = torch.unique(m_id)
        uniq_a = torch.unique(torch.cat([pos_a, neg_a.view(-1)]))
        loss_cl = self._info_nce(m_align[uniq_m], m_struct_proj[uniq_m]) + self._info_nce(a_text[uniq_a], a_struct_proj[uniq_a])

        loss = loss_bce + self.hparams.lambda_cl * loss_cl
        loss_tuple = torch.tensor(0.0, device=self.device)

        if self.hparams.use_tuple_head:
            tuple_alpha = self._tuple_alpha()
            neg_item_provider_id = self.api_provider_ids[neg_item_same_scene_id]

            delta_pos = self._tuple_delta(
                m_id, pos_a, pos_provider_id, pos_scene_id,
                m_align, a_text, m_struct_proj, a_struct_proj,
            )
            delta_neg_scene = self._tuple_delta(
                m_id, pos_a, pos_provider_id, neg_scene_same_item_id,
                m_align, a_text, m_struct_proj, a_struct_proj,
            )
            delta_neg_item = self._tuple_delta(
                m_id, neg_item_same_scene_id, neg_item_provider_id, pos_scene_id,
                m_align, a_text, m_struct_proj, a_struct_proj,
            )

            neg_item_base = self._item_scores(
                m_id, neg_item_same_scene_id, m_align, a_text, m_struct_proj, a_struct_proj
            )

            score_pos_tuple = pos_score.detach() + tuple_alpha * delta_pos
            score_neg_scene_tuple = pos_score.detach() + tuple_alpha * delta_neg_scene
            score_neg_item_tuple = neg_item_base.detach() + tuple_alpha * delta_neg_item

            loss_tuple = F.softplus(-(score_pos_tuple - score_neg_scene_tuple)).mean()
            loss_tuple = loss_tuple + F.softplus(-(score_pos_tuple - score_neg_item_tuple)).mean()
            loss = loss + self.hparams.lambda_tuple * loss_tuple

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/bce", loss_bce, on_step=True, on_epoch=True)
        self.log("train/cl", loss_cl, on_step=True, on_epoch=True)
        self.log("train/tuple", loss_tuple, on_step=True, on_epoch=True)
        self.log("train/tuple_alpha", torch.tensor(self._tuple_alpha(), device=self.device), on_step=False, on_epoch=True)
        return loss

    def _mask_seen(self, scores, user_ids):
        for row_idx, uid in enumerate(user_ids.tolist()):
            seen = self.train_mapping.get(uid, [])
            if len(seen) > 0:
                scores[row_idx, torch.tensor(seen, device=scores.device)] = -1e18
        return scores

    def _full_item_scores(self, user_ids):
        m_align, a_text, m_struct_proj, a_struct_proj = self._compute_embeddings()
        m_struct_proj_batch = m_struct_proj[user_ids]
        m_align_batch = m_align[user_ids]

        struct_scores = torch.matmul(m_struct_proj_batch, a_struct_proj.t())
        m_expand = m_align_batch.unsqueeze(1).expand(-1, self.hparams.num_apis, -1)
        a_expand = a_text.unsqueeze(0).expand(m_align_batch.size(0), -1, -1)
        text_scores = self.text_mlp(torch.cat([m_expand, a_expand], dim=2)).squeeze(-1)

        scores = self.hparams.beta * text_scores + (1.0 - self.hparams.beta) * struct_scores

        return scores, (m_align, a_text, m_struct_proj, a_struct_proj)

    def validation_step(self, batch, batch_idx):
        user_ids, labels, _ = batch
        user_ids = user_ids.long().to(self.device)
        labels = labels.to(self.device)

        scores, _ = self._full_item_scores(user_ids)
        scores = self._mask_seen(scores, user_ids)
        self.val_metrics.update(scores, labels)

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def _log_metrics(self, stage, metrics):
        values = metrics.compute()
        for name, value in values.items():
            self.log(
                f"{stage}/{name}", value, on_step=False, on_epoch=True,
                prog_bar=name in ("P@5", "DCG@5", "MRR"),
            )
        return values

    def on_validation_epoch_end(self):
        self._log_metrics("val", self.val_metrics)

    def test_step(self, batch, batch_idx):
        user_ids, labels, _ = batch
        user_ids = user_ids.long().to(self.device)
        labels = labels.to(self.device)

        scores, _ = self._full_item_scores(user_ids)
        scores = self._mask_seen(scores, user_ids)
        self.test_metrics.update(scores, labels)


    def on_test_epoch_start(self):
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return

        self.test_metrics.reset()


    def on_test_epoch_end(self):
        values = self._log_metrics("test", self.test_metrics)
        if not self.trainer.is_global_zero:
            return
        output_dir = getattr(self.logger, "log_dir", None) or self.trainer.default_root_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump({name: float(value.item()) for name, value in values.items()}, stream, indent=2)


    @torch.no_grad()
    def predict_tuple_topk(self, user_ids: torch.Tensor, top_k: int = 10, top_m: Optional[int] = None):
        """
        Return explicit Top-K <provider, api, scene> tuples for the given user ids.
        This does not change the standard API-level validation/test metrics.
        """
        user_ids = user_ids.long().to(self.device)
        if top_m is None:
            top_m = int(self.hparams.topm_rerank)

        item_scores, cached = self._full_item_scores(user_ids)
        item_scores = self._mask_seen(item_scores, user_ids)
        m_align, a_text, m_struct_proj, a_struct_proj = cached
        tuple_alpha = self._tuple_alpha()

        top_item_scores, top_item_ids = torch.topk(item_scores, k=min(top_m, item_scores.size(1)), dim=1)
        outputs = []
        for row_idx, uid in enumerate(user_ids.tolist()):
            tuple_candidates = []
            for local_rank in range(top_item_ids.size(1)):
                api_id = int(top_item_ids[row_idx, local_rank].item())
                base_score = float(top_item_scores[row_idx, local_rank].item())
                provider_id = int(self.api_provider_ids[api_id].item())
                scene_ids = self.api_scene_lists[api_id][: max(int(self.hparams.max_eval_scenes), 1)]
                for scene_id in scene_ids:
                    mid = torch.tensor([uid], device=self.device)
                    aid = torch.tensor([api_id], device=self.device)
                    pid = torch.tensor([provider_id], device=self.device)
                    sid = torch.tensor([int(scene_id)], device=self.device)
                    delta = self._tuple_delta(mid, aid, pid, sid, m_align, a_text, m_struct_proj, a_struct_proj)
                    score = base_score + tuple_alpha * float(delta.item())
                    tuple_candidates.append({
                        "provider_id": provider_id,
                        "provider": self.id2provider[provider_id] if provider_id < len(self.id2provider) else str(provider_id),
                        "api_id": api_id,
                        "scene_id": int(scene_id),
                        "scene": self.id2scene[int(scene_id)] if int(scene_id) < len(self.id2scene) else str(scene_id),
                        "score": score,
                    })
            tuple_candidates.sort(key=lambda x: x["score"], reverse=True)
            outputs.append(tuple_candidates[:top_k])
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
