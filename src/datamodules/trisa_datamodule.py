import csv
import re
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.dataset.PWDataset import PWDataset


class TriSATrainDataset(Dataset):
    def __init__(
        self,
        interactions,
        num_apis,
        user_pos,
        neg_k,
        rng,
        api_provider_ids,
        api_scene_lists,
        scene_to_api_ids,
        all_scene_ids,
    ):
        self.interactions = interactions
        self.num_apis = num_apis
        self.user_pos = user_pos
        self.neg_k = neg_k
        self.rng = rng
        self.api_provider_ids = np.asarray(api_provider_ids, dtype=np.int64)
        self.api_scene_lists = [list(x) if len(x) > 0 else [0] for x in api_scene_lists]
        self.scene_to_api_ids = {int(k): list(v) for k, v in scene_to_api_ids.items()}
        self.all_scene_ids = list(all_scene_ids) if len(all_scene_ids) > 0 else [0]

    def set_rng(self, rng):
        self.rng = rng

    def __len__(self):
        return len(self.interactions)

    def _sample_random_unseen_item(self, m_id, exclude=None):
        pos_set = self.user_pos.get(m_id, set())
        if exclude is None:
            exclude = set()
        while True:
            neg = int(self.rng.integers(0, self.num_apis))
            if neg not in pos_set and neg not in exclude:
                return neg

    def _sample_pos_scene(self, api_id):
        scenes = self.api_scene_lists[api_id]
        idx = int(self.rng.integers(0, len(scenes)))
        return int(scenes[idx])

    def _sample_neg_scene_same_item(self, api_id, pos_scene):
        pos_scene_set = set(self.api_scene_lists[api_id])
        candidates = [sid for sid in self.all_scene_ids if sid not in pos_scene_set]
        if not candidates:
            return int(pos_scene)
        idx = int(self.rng.integers(0, len(candidates)))
        return int(candidates[idx])

    def _sample_neg_item_same_scene(self, m_id, pos_a, pos_scene):
        pos_set = self.user_pos.get(m_id, set())
        candidates = [
            int(api_id)
            for api_id in self.scene_to_api_ids.get(int(pos_scene), [])
            if api_id != int(pos_a) and api_id not in pos_set
        ]
        if candidates:
            idx = int(self.rng.integers(0, len(candidates)))
            return int(candidates[idx])
        return self._sample_random_unseen_item(m_id, exclude={int(pos_a)})

    def __getitem__(self, idx):
        m_id, pos_a = self.interactions[idx]
        pos_set = self.user_pos.get(m_id, set())

        negs = []
        while len(negs) < self.neg_k:
            neg = int(self.rng.integers(0, self.num_apis))
            if neg not in pos_set:
                negs.append(neg)

        pos_provider = int(self.api_provider_ids[pos_a])
        pos_scene = self._sample_pos_scene(pos_a)
        neg_scene_same_item = self._sample_neg_scene_same_item(pos_a, pos_scene)
        neg_item_same_scene = self._sample_neg_item_same_scene(m_id, pos_a, pos_scene)

        return {
            "m_id": torch.tensor(m_id, dtype=torch.long),
            "pos_a": torch.tensor(pos_a, dtype=torch.long),
            "neg_a": torch.tensor(negs, dtype=torch.long),
            "pos_provider_id": torch.tensor(pos_provider, dtype=torch.long),
            "pos_scene_id": torch.tensor(pos_scene, dtype=torch.long),
            "neg_scene_same_item_id": torch.tensor(neg_scene_same_item, dtype=torch.long),
            "neg_item_same_scene_id": torch.tensor(neg_item_same_scene, dtype=torch.long),
        }


class TriSADataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        mashup_num,
        api_num,
        neg_k,
        train_edges_path,
        val_edges_path,
        test_edges_path,
        api_embedding_pt,
        mashup_embedding_pt,
        api_nodes_path,
        sample_k_min,
        sample_k_max,
        max_hyperedge_size,
        seed,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mashup_num = mashup_num
        self.api_num = api_num
        self.neg_k = neg_k
        self.train_edges_path = train_edges_path
        self.val_edges_path = val_edges_path
        self.test_edges_path = test_edges_path
        self.api_embedding_pt = api_embedding_pt
        self.mashup_embedding_pt = mashup_embedding_pt
        self.api_nodes_path = api_nodes_path
        self.sample_k_min = sample_k_min
        self.sample_k_max = sample_k_max
        self.max_hyperedge_size = max_hyperedge_size
        self.seed = int(seed)
        self.epoch = 0
        self.rng = np.random.default_rng(self.seed)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.uiMat = None
        self.H_co = None
        self.H_cat = None
        self.H_prov = None

        self.mashup_text_emb = None
        self.api_text_emb = None

        self.train_mapping = {}
        self.val_mapping = {}
        self.test_mapping = {}

        self.hco_edges = []
        self.hcat_edges = []
        self.hprov_edges = []
        self.api_freq = None

        # tuple-aware metadata, all indexed at api-level
        self.provider2id = {"__unk_provider__": 0}
        self.scene2id = {"__unk_scene__": 0}
        self.id2provider = ["__unk_provider__"]
        self.id2scene = ["__unk_scene__"]
        self.api_provider_ids = np.zeros(self.api_num, dtype=np.int64)
        self.api_scene_lists = [[0] for _ in range(self.api_num)]
        self.scene_to_api_ids = defaultdict(list)

    def _normalize_api_index(self, api_idx):
        return api_idx - self.mashup_num

    def _load_edges(self, path):
        edges = torch.load(path, map_location="cpu")
        if torch.is_tensor(edges):
            edges = edges.cpu().numpy()
        edges = np.asarray(edges)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"Expected an [N, 2] edge array in {path}, got {edges.shape}.")
        if not np.issubdtype(edges.dtype, np.number) or not np.isfinite(edges).all():
            raise ValueError(f"Edges must contain finite numeric IDs: {path}")
        if not np.equal(edges, np.floor(edges)).all():
            raise ValueError(f"Edges must contain integer IDs: {path}")
        valid = (
            (edges[:, 0] >= 0) & (edges[:, 0] < self.mashup_num)
            & (edges[:, 1] >= self.mashup_num)
            & (edges[:, 1] < self.mashup_num + self.api_num)
        )
        if not valid.all():
            raise ValueError(f"Out-of-range Mashup/API IDs in {path}: {edges[~valid][0].tolist()}")
        return edges.astype(np.int64, copy=False)

    def _edges_to_mapping(self, edges):
        mapping = defaultdict(list)
        for u_id, i_id in edges:
            api_idx = self._normalize_api_index(int(i_id))
            mapping[int(u_id)].append(api_idx)
        return mapping

    def _edges_to_interactions(self, mapping):
        interactions = []
        for u_id, api_list in mapping.items():
            for api_idx in api_list:
                interactions.append((u_id, api_idx))
        return interactions

    def _edges_to_eval_dataset(self, mapping):
        xs, ys, ts = [], [], []
        for u, api_list in mapping.items():
            if len(api_list) == 0:
                continue
            xs.append(u)
            ys.append(api_list)
            ts.append(0)
        return PWDataset(xs, ys, ts, self.api_num)

    def _load_embeddings(self):
        api_embeds_raw = torch.load(self.api_embedding_pt, map_location="cpu")
        mashup_embeds_raw = torch.load(self.mashup_embedding_pt, map_location="cpu")
        if isinstance(api_embeds_raw, list):
            api_embeds = torch.stack(api_embeds_raw, dim=0)
        else:
            api_embeds = torch.as_tensor(api_embeds_raw)
        if isinstance(mashup_embeds_raw, list):
            mashup_embeds = torch.stack(mashup_embeds_raw, dim=0)
        else:
            mashup_embeds = torch.as_tensor(mashup_embeds_raw)
        for name, features, count in (
            ("API", api_embeds, self.api_num), ("Mashup", mashup_embeds, self.mashup_num)
        ):
            if features.ndim != 2 or features.shape[0] != count:
                raise ValueError(f"{name} features must have shape [{count}, text_dim], got {tuple(features.shape)}.")
            if not torch.isfinite(features).all():
                raise ValueError(f"{name} features contain non-finite values.")
        if api_embeds.shape[1] != mashup_embeds.shape[1]:
            raise ValueError("API and Mashup text feature dimensions must match.")
        return mashup_embeds.float(), api_embeds.float()

    def _build_ui_mat(self, train_edges_global):
        rows = []
        cols = []
        for u_id, i_id in train_edges_global:
            rows.append(int(u_id))
            cols.append(int(i_id))
            rows.append(int(i_id))
            cols.append(int(u_id))
        data = np.ones(len(rows), dtype=np.float32)
        size = self.mashup_num + self.api_num
        return sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    def _extract_provider(self, url):
        """Use the first hyphen-separated word in the PW /api/<slug> path."""
        if url is None:
            return None
        url = str(url).strip()
        if not url:
            return None
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or parts[0].lower() != "api":
            return None
        provider = parts[1].split("-", 1)[0].strip().lower()
        return provider or None

    def _parse_categories(self, raw_category):
        if raw_category is None:
            return []
        text = str(raw_category).strip()
        if not text:
            return []
        parts = re.split(r"[|,;/]+", text)
        cats = [p.strip() for p in parts if p and p.strip()]
        return cats if cats else [text]

    def _get_provider_id(self, provider):
        if not provider:
            return 0
        provider = str(provider).strip().lower()
        if provider not in self.provider2id:
            self.provider2id[provider] = len(self.provider2id)
            self.id2provider.append(provider)
        return int(self.provider2id[provider])

    def _get_scene_id(self, scene_name):
        if not scene_name:
            return 0
        scene_name = str(scene_name).strip()
        if scene_name not in self.scene2id:
            self.scene2id[scene_name] = len(self.scene2id)
            self.id2scene.append(scene_name)
        return int(self.scene2id[scene_name])

    def _load_api_nodes(self):
        cat_groups = defaultdict(list)
        prov_groups = defaultdict(list)
        self.api_provider_ids = np.zeros(self.api_num, dtype=np.int64)
        self.api_scene_lists = [[0] for _ in range(self.api_num)]
        self.scene_to_api_ids = defaultdict(list)
        self.provider2id = {"__unk_provider__": 0}
        self.scene2id = {"__unk_scene__": 0}
        self.id2provider = ["__unk_provider__"]
        self.id2scene = ["__unk_scene__"]
        num_rows = 0

        with open(self.api_nodes_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not {"url", "c"}.issubset(reader.fieldnames or []):
                raise ValueError("API metadata must be tab-separated and contain url and c columns.")
            for idx, row in enumerate(reader):
                if idx >= self.api_num:
                    break
                num_rows += 1

                category_raw = row.get("c")
                categories = self._parse_categories(category_raw)
                if categories:
                    self.api_scene_lists[idx] = []
                    for cat in categories:
                        cat_groups[str(cat)].append(idx)
                        sid = self._get_scene_id(cat)
                        self.api_scene_lists[idx].append(sid)
                        self.scene_to_api_ids[sid].append(idx)
                else:
                    self.api_scene_lists[idx] = [0]
                    self.scene_to_api_ids[0].append(idx)

                provider = self._extract_provider(row.get("url"))
                provider_id = self._get_provider_id(provider)
                self.api_provider_ids[idx] = provider_id
                if provider:
                    prov_groups[provider].append(idx)

        if num_rows != self.api_num:
            raise ValueError(f"Expected at least {self.api_num} API metadata rows, got {num_rows}.")
        self.hcat_edges = [self._truncate_edge(v) for v in cat_groups.values() if len(v) > 1]
        self.hprov_edges = [self._truncate_edge(v) for v in prov_groups.values() if len(v) > 1]

    def _truncate_edge(self, members):
        if len(members) <= self.max_hyperedge_size:
            return list(members)
        return self.rng.choice(list(members), size=self.max_hyperedge_size, replace=False).tolist()

    def _sample_hco_edges(self):
        sampled = []
        for members in self.hco_edges:
            if len(members) == 0:
                continue
            members = self._truncate_edge(members)
            k = int(self.rng.integers(self.sample_k_min, self.sample_k_max + 1))
            k = min(k, len(members))
            if k < 2:
                continue
            weights = np.array([self.api_freq[m] for m in members], dtype=np.float64)
            weights = np.where(weights > 0, 1.0 / np.sqrt(weights), 1.0)
            weights = weights / weights.sum()
            sampled_members = self.rng.choice(members, size=k, replace=False, p=weights)
            sampled.append(sampled_members.tolist())
        return sampled


    def _build_incidence(self, hyperedges):
        clean_edges = []
        for edge in hyperedges:
            uniq = sorted({int(x) for x in edge if 0 <= int(x) < self.api_num})
            if len(uniq) >= 2:
                clean_edges.append(uniq)
        if not clean_edges:
            return sp.eye(self.api_num, dtype=np.float32, format="csr")

        rows = []
        cols = []
        data = []
        for e_idx, edge in enumerate(clean_edges):
            rows.extend(edge)
            cols.extend([e_idx] * len(edge))
            data.extend([1.0] * len(edge))
        return sp.coo_matrix((data, (rows, cols)), shape=(self.api_num, len(clean_edges)), dtype=np.float32).tocsr()

    def refresh_hypergraph(self):
        sampled_hco = self._sample_hco_edges()

        self.H_co = self._build_incidence(sampled_hco)
        self.H_cat = self._build_incidence(self.hcat_edges)
        self.H_prov = self._build_incidence(self.hprov_edges)

    def setup(self, stage=None):
        train_edges_global = self._load_edges(self.train_edges_path)
        val_edges = self._load_edges(self.val_edges_path)
        test_edges = self._load_edges(self.test_edges_path)

        self.train_mapping = self._edges_to_mapping(train_edges_global)
        self.val_mapping = self._edges_to_mapping(val_edges)
        self.test_mapping = self._edges_to_mapping(test_edges)

        self.api_freq = np.zeros(self.api_num, dtype=np.int64)
        for api_list in self.train_mapping.values():
            for api in api_list:
                if 0 <= api < self.api_num:
                    self.api_freq[api] += 1

        self.hco_edges = [apis for apis in self.train_mapping.values() if len(apis) > 0]
        self._load_api_nodes()

        self.refresh_hypergraph()
        self.uiMat = self._build_ui_mat(train_edges_global)

        interactions = self._edges_to_interactions(self.train_mapping)
        train_mapping_set = {u: set(v) for u, v in self.train_mapping.items()}
        self.train_dataset = TriSATrainDataset(
            interactions=interactions,
            num_apis=self.api_num,
            user_pos=train_mapping_set,
            neg_k=self.neg_k,
            rng=self.rng,
            api_provider_ids=self.api_provider_ids,
            api_scene_lists=self.api_scene_lists,
            scene_to_api_ids=self.scene_to_api_ids,
            all_scene_ids=sorted(self.scene2id.values()),
        )
        self.val_dataset = self._edges_to_eval_dataset(self.val_mapping)
        self.test_dataset = self._edges_to_eval_dataset(self.test_mapping)

        self.mashup_text_emb, self.api_text_emb = self._load_embeddings()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.rng = np.random.default_rng(self.seed + self.epoch)
        if self.train_dataset is not None:
            self.train_dataset.set_rng(self.rng)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
