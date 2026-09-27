# TriSA

## Tri-Type Hypergraph Semantic Alignment for Metaverse Service Paradigm and Recommendation

**Ruipeng Han, Dunlei Rong, Yeqi Zhu, Zihang Su, Xiao Wang, and Hanchuan Xu**  
Faculty of Computing, Harbin Institute of Technology, Harbin, China

**IEEE International Conference on Web Services (ICWS), 2026**  
[Conference Program](https://services.conferences.computer.org/2026/icws-program/) · [Dataset](https://github.com/HIT-ICES/Correted-ProgrammableWeb-dataset) · [Getting Started](#getting-started) · [Citation](#citation)

This repository provides the PyTorch implementation of **TriSA** for the ProgrammableWeb (PW) experiments. It contains the model, data loader, training and evaluation pipeline, and experiment configurations. Data files, preprocessing scripts, and local regression tests are excluded from this release.

## Overview

Metaverse service recommendation requires modeling how **providers, customers, services, and scenes** jointly determine service relevance. TriSA addresses this problem through typed high-order structure modeling and structural-textual semantic alignment. It combines three components:

- **Tri-type hypergraph modeling:** co-invocation, provider, and scene hypergraphs capture complementary high-order service dependencies alongside the customer-service interaction graph.
- **Frequency-aware hyperedge debiasing:** frequency-aware sampling reduces the dominance of frequently invoked services during structural learning.
- **Cross-tower semantic alignment:** contrastive learning aligns structural representations with textual semantics to support recommendation under sparse customer intent.

A lightweight residual scoring module incorporates provider and scene factors into explicit service-pattern ranking. The joint training objective combines recommendation, semantic alignment, and residual ranking losses:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \lambda_{\mathrm{cl}}\mathcal{L}_{\mathrm{cl}} + \lambda_{\mathrm{r}}\mathcal{L}_{\mathrm{r}}.
$$

The default PW pipeline evaluates **API-level recommendation**. The model also exposes `predict_tuple_topk()` for explicit provider-service-scene prediction. The synthetic metaverse case studies are outside the scope of this release.

## Getting Started

### Environment

Use Python 3.9. The main dependencies are PyTorch 2.0.1, PyTorch Lightning 2.0.0, and Hydra 1.3.2; the complete list of direct dependencies is in [requirements.txt](requirements.txt).

From the repository root, install the PyTorch build appropriate for your machine, followed by the project dependencies.

**CUDA 11.8:**

```bash
python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

**CPU:**

```bash
python -m pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The `torch==2.0.1` requirement accepts either of these builds. Local release checks used Python 3.9.23 and the CPU build; CUDA execution has not been revalidated during release preparation.

### Download the Dataset

Download ProgrammableWeb from the **official dataset repository cited in Section VI-A.1, footnote 1 of the paper**:

**[HIT-ICES / Correted-ProgrammableWeb-dataset](https://github.com/HIT-ICES/Correted-ProgrammableWeb-dataset)**

Use the repository's **Code → Download ZIP** option, or clone it into a directory beside TriSA:

```bash
git clone https://github.com/HIT-ICES/Correted-ProgrammableWeb-dataset.git ../Correted-ProgrammableWeb-dataset
```

Please follow the official repository's data description and citation instructions. Prepare the data according to the experimental protocol in the paper before training.

This release excludes datasets, experiment-specific splits, precomputed text embeddings, and preprocessing scripts. The official download provides the source dataset; compatible training inputs must be prepared separately.

## Training and Evaluation

Run commands from the repository root after preparing the input files.

**Train on GPU 0 and evaluate the best checkpoint:**

```bash
python run.py
```

**Train and evaluate on CPU:**

```bash
python run.py trainer.accelerator=cpu trainer.devices=1
```

**Check one batch with the prepared data:**

```bash
python run.py debug=true
```

The debug command uses Lightning's fast development mode and does not perform the normal checkpoint-based test run.

### Configuration

The main configuration is [configs/config.yaml](configs/config.yaml). It composes the model, data, data module, optimizer settings, trainer, callbacks, logging, and Hydra output settings.

| Setting | Default | Override |
| --- | ---: | --- |
| Random seed | 2021 | `seed=2021` |
| Batch size | 256 | `train.batch_size=256` |
| Negative APIs per positive instance | 6 | `train.neg_k=6` |
| Learning rate | 0.001 | `train.lr=0.001` |
| Weight decay | 0.0001 | `train.weight_decay=0.0001` |
| Maximum epochs | 50 | `train.max_epochs=50` |
| Semantic/structural fusion coefficient | 0.7 | `train.beta=0.7` |
| Contrastive loss weight | 0.02 | `train.lambda_cl=0.02` |
| Contrastive temperature | 0.2 | `train.cl_temp=0.2` |
| Sampled co-invocation hyperedge size | 3-8 | `train.sample_k_min=3 train.sample_k_max=8` |
| Maximum hyperedge size | 200 | `train.max_hyperedge_size=200` |

For example, run with another random seed:

```bash
python run.py seed=2022
```

The optimizer is Adam. The default trainer runs for at least 10 epochs and at most 50, with early stopping based on validation `P@5` and patience 10. The data loader uses one worker by default; `datamodule.num_workers=0` is also supported. More than one worker requires separate initialization of each worker's internal NumPy random generator.

### Outputs

Each run writes to `logs/runs/<date>/<time>_seed<seed>/`. The pipeline selects `best.ckpt` by validation **P@5**, then restores it for testing. A checkpoint contains both model weights and the graph state used at that epoch.

Evaluation reports Precision, NDCG, Recall, and F1 at `K = 5, 10, 15`, together with MRR. Results are saved in `metrics.json` under the TensorBoard logger directory. Metric values are fractions. Output keys named `DCG@K` contain **NDCG@K**.

TensorBoard logs are enabled by default. To disable them:

```bash
python run.py '~logger'
```

Without a logger, `metrics.json` is saved directly in the run directory.

## Repository Structure

```text
TriSA/
├── configs/                       # Experiment configuration groups
│   ├── config.yaml                # Main configuration
│   ├── data/mashup_api.yaml        # Input paths and node counts
│   ├── datamodule/trisa.yaml       # Data loading and graph sampling
│   ├── model/trisa.yaml            # Model dimensions and parameters
│   ├── train/default.yaml          # Optimization and sampling settings
│   ├── trainer/default.yaml        # Device and training control
│   ├── callbacks/default.yaml      # Checkpoint selection and early stopping
│   ├── logger/tensorboard.yaml     # TensorBoard logging
│   └── hydra/default.yaml          # Run directories
├── src/
│   ├── datamodules/
│   │   ├── trisa_datamodule.py     # Data loading, graph construction, sampling
│   │   └── dataset/PWDataset.py    # Validation/test sample representation
│   ├── models/
│   │   ├── trisa.py                # TriSA, losses, ranking, and checkpointing
│   │   └── trisa_backbone.py       # Interaction graph and hypergraph encoder
│   ├── utils/
│   │   ├── metrics.py              # Ranking metrics
│   │   └── utils.py                # Configuration and logging helpers
│   └── train.py                    # Training and evaluation orchestration
├── run.py                         # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

Python package initialization files are omitted from the overview above. Local `data/` and `tests/` directories are excluded from version control.

## Citation

If you use TriSA in your research, please cite:

```bibtex
@inproceedings{han2026trisa,
  title     = {{TriSA}: Tri-Type Hypergraph Semantic Alignment for Metaverse Service Paradigm and Recommendation},
  author    = {Han, Ruipeng and Rong, Dunlei and Zhu, Yeqi and Su, Zihang and Wang, Xiao and Xu, Hanchuan},
  booktitle = {2026 IEEE International Conference on Web Services (ICWS)},
  year      = {2026}
}
```

We thank the maintainers of the [Corrected ProgrammableWeb dataset](https://github.com/HIT-ICES/Correted-ProgrammableWeb-dataset). If you use PW, please also cite the dataset paper following the instructions in its official repository.
