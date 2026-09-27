import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def scipy_sparse_to_torch(mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo().astype(np.float32)
    else:
        mat = mat.astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64)).to(device)
    values = torch.from_numpy(mat.data).float().to(device)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def normalize_adj(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).tocsr()


class GraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HypergraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        x: [N, d]
        H: sparse incidence matrix [N, E]
        """
        if not H.is_sparse:
            raise ValueError("Hypergraph incidence matrix H must be sparse.")

        H = H.coalesce()
        dv = torch.sparse.sum(H, dim=1).to_dense()  # [N]
        de = torch.sparse.sum(H, dim=0).to_dense()  # [E]

        dv_inv_sqrt = torch.pow(dv + 1e-8, -0.5)
        de_inv = torch.pow(de + 1e-8, -1.0)

        x = dv_inv_sqrt.unsqueeze(1) * x
        x = torch.sparse.mm(H.transpose(0, 1).coalesce(), x)
        x = de_inv.unsqueeze(1) * x
        x = torch.sparse.mm(H, x)
        x = dv_inv_sqrt.unsqueeze(1) * x

        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MODEL(nn.Module):
    def __init__(
        self,
        mashup_num: int,
        api_num: int,
        H_co: sp.spmatrix,
        H_cat: sp.spmatrix,
        H_prov: sp.spmatrix,
        uiMat: sp.spmatrix,
        hide_dim: int,
        Layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mashup_num = mashup_num
        self.api_num = api_num
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        self.dropout = dropout

        self.H_co = H_co
        self.H_cat = H_cat
        self.H_prov = H_prov
        self.uiMat = uiMat

        self.user_emb = nn.Embedding(mashup_num, hide_dim)
        self.api_emb = nn.Embedding(api_num, hide_dim)

        self.ui_layers = nn.ModuleList([GraphLayer(hide_dim, dropout) for _ in range(Layers)])
        self.hg_co_layers = nn.ModuleList([HypergraphLayer(hide_dim, dropout) for _ in range(Layers)])
        self.hg_cat_layers = nn.ModuleList([HypergraphLayer(hide_dim, dropout) for _ in range(Layers)])
        self.hg_prov_layers = nn.ModuleList([HypergraphLayer(hide_dim, dropout) for _ in range(Layers)])


        self.W_a = nn.Linear(3 * hide_dim, 3, bias=False)
        self.W_ui = nn.Linear(2 * hide_dim, 2, bias=False)

        self._cached_device = None
        self.graph_revision = 0
        self._ui_adj_t = None
        self._H_co_t = None
        self._H_cat_t = None
        self._H_prov_t = None

    def _refresh_sparse_cache(self, device: torch.device):
        device_key = str(device)
        if self._cached_device == device_key:
            return
        self._cached_device = device_key
        self._ui_adj_t = scipy_sparse_to_torch(normalize_adj(self.uiMat, add_self_loops=True), device)
        self._H_co_t = scipy_sparse_to_torch(self.H_co, device)
        self._H_cat_t = scipy_sparse_to_torch(self.H_cat, device)
        self._H_prov_t = scipy_sparse_to_torch(self.H_prov, device)

    def update_mats(self, H_co: sp.spmatrix, H_cat: sp.spmatrix, H_prov: sp.spmatrix, uiMat: sp.spmatrix):
        self.H_co = H_co
        self.H_cat = H_cat
        self.H_prov = H_prov
        self.uiMat = uiMat
        self.graph_revision += 1
        self._cached_device = None
        self._ui_adj_t = None
        self._H_co_t = None
        self._H_cat_t = None
        self._H_prov_t = None

    def _encode_ui_view(self):
        device = self.user_emb.weight.device
        self._refresh_sparse_cache(device)

        x = torch.cat([self.user_emb.weight, self.api_emb.weight], dim=0)
        states = [F.normalize(x, p=2, dim=1)]
        for layer in self.ui_layers:
            x = layer(x, self._ui_adj_t)
            states.append(F.normalize(x, p=2, dim=1))
        x = torch.mean(torch.stack(states, dim=1), dim=1)
        mashup_ui, api_ui = torch.split(x, [self.mashup_num, self.api_num], dim=0)
        return mashup_ui, api_ui

    def _encode_single_hypergraph(self, layers: nn.ModuleList, H: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        states = [F.normalize(x0, p=2, dim=1)]
        for layer in layers:
            x = layer(x, H)
            states.append(F.normalize(x, p=2, dim=1))
        return torch.mean(torch.stack(states, dim=1), dim=1)

    def encode(self):
        device = self.user_emb.weight.device
        self._refresh_sparse_cache(device)

        mashup_ui, api_ui = self._encode_ui_view()
        api_x0 = self.api_emb.weight

        z_co = self._encode_single_hypergraph(self.hg_co_layers, self._H_co_t, api_x0)
        z_cat = self._encode_single_hypergraph(self.hg_cat_layers, self._H_cat_t, api_x0)
        z_prov = self._encode_single_hypergraph(self.hg_prov_layers, self._H_prov_t, api_x0)

        h = torch.cat([z_co, z_cat, z_prov], dim=1)
        alpha_type = torch.softmax(self.W_a(h), dim=1)
        api_hyper = (
            alpha_type[:, 0:1] * z_co
            + alpha_type[:, 1:2] * z_cat
            + alpha_type[:, 2:3] * z_prov
        )

        h_ui = torch.cat([api_ui, api_hyper], dim=1)
        alpha_view = torch.softmax(self.W_ui(h_ui), dim=1)
        api_struct = alpha_view[:, 0:1] * api_ui + alpha_view[:, 1:2] * api_hyper

        mashup_struct = mashup_ui

        aux = {
            "alpha_type": alpha_type,
            "alpha_view": alpha_view,
            "api_ui": api_ui,
            "api_hyper": api_hyper,
            "z_co": z_co,
            "z_cat": z_cat,
            "z_prov": z_prov,
        }
        return mashup_struct, api_struct, aux

    def forward(self):
        mashup_struct, api_struct, _ = self.encode()
        return mashup_struct, api_struct


class TriSABackbone(nn.Module):
    def __init__(
        self,
        userNum: int,
        itemNum: int,
        H_co: sp.spmatrix,
        H_cat: sp.spmatrix,
        H_prov: sp.spmatrix,
        uiMat: sp.spmatrix,
        hide_dim: int,
        Layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.core = MODEL(
            mashup_num=userNum,
            api_num=itemNum,
            H_co=H_co,
            H_cat=H_cat,
            H_prov=H_prov,
            uiMat=uiMat,
            hide_dim=hide_dim,
            Layers=Layers,
            dropout=dropout,
        )

    def update_mats(self, H_co: sp.spmatrix, H_cat: sp.spmatrix, H_prov: sp.spmatrix, uiMat: sp.spmatrix):
        self.core.update_mats(H_co, H_cat, H_prov, uiMat)

    def graph_state_dict(self):
        """Snapshot the actual graphs used by the encoder, including sampled edges."""
        matrices = {}
        for name in ("H_co", "H_cat", "H_prov", "uiMat"):
            mat = getattr(self.core, name).tocsr()
            # Owned CPU tensors avoid pickling SciPy objects or aliasing live graphs.
            matrices[name] = {
                "shape": tuple(mat.shape),
                "data": torch.from_numpy(mat.data.copy()),
                "indices": torch.from_numpy(mat.indices.astype(np.int64, copy=True)),
                "indptr": torch.from_numpy(mat.indptr.astype(np.int64, copy=True)),
            }
        return {
            "format_version": 1,
            "num_mashups": self.core.mashup_num,
            "num_apis": self.core.api_num,
            "matrices": matrices,
        }

    def load_graph_state_dict(self, state):
        """Restore variable-sized CSR graphs and invalidate device-side caches."""
        if state.get("format_version") != 1:
            raise ValueError("Unsupported TriSA graph checkpoint format.")
        if (state.get("num_mashups"), state.get("num_apis")) != (
            self.core.mashup_num, self.core.api_num
        ):
            raise ValueError("Checkpoint graph node counts do not match this TriSA model.")

        matrices = {}
        total_nodes = self.core.mashup_num + self.core.api_num
        for name in ("H_co", "H_cat", "H_prov", "uiMat"):
            saved = state["matrices"][name]
            shape = tuple(saved["shape"])
            if len(shape) != 2 or min(shape) < 0:
                raise ValueError(f"Invalid checkpoint graph shape for {name}: {shape}")
            if name == "uiMat":
                valid_shape = shape == (total_nodes, total_nodes)
            else:
                valid_shape = shape[0] == self.core.api_num
            if not valid_shape:
                raise ValueError(f"Checkpoint graph shape does not match {name}: {shape}")
            mat = sp.csr_matrix(
                (
                    saved["data"].detach().cpu().numpy(),
                    saved["indices"].detach().cpu().numpy(),
                    saved["indptr"].detach().cpu().numpy(),
                ),
                shape=shape,
                copy=True,
            )
            mat.check_format(full_check=True)
            matrices[name] = mat
        self.update_mats(**matrices)

    def forward(self):
        return self.core()
