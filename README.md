# hji-pi-pinn (repro)

Reproducible codebase for policy-iteration PINNs on viscous HJ/HJI PDEs.

Problems covered:
- Path planning (2D + time, moving obstacle): `hji_pi_pinn/problems/path_planning/`
- Publisher-subscriber game (odd dimensions): `hji_pi_pinn/problems/publisher_subscriber/`

## Environment
Install:
```bash
pip install -r requirements.txt
```

Tested baseline:
- Python `3.9-3.11`
- `jax==0.4.30`
- `jaxlib==0.4.30`
- `optax==0.2.4`
- `numpy==1.26.4`
- `matplotlib==3.9.4`
- `tqdm==4.67.1`
- `PyYAML==6.0.2`

Notes:
- `requirements.txt` is a CPU-safe pinned baseline.
- For GPU/CUDA, install JAX/JAXLIB using the official JAX installation guide for your CUDA/driver version, then install the remaining dependencies.
- On CPU-only installs, a message like "CUDA-enabled jaxlib is not installed, falling back to cpu" is expected.
- If install fails on an old pip, run:
  `pip install --upgrade pip setuptools wheel`

## Project Layout
- `configs/`: YAML configs
- `scripts/`: training entrypoints
- `hji_pi_pinn/`: core/problem modules
- `plots/`: figure scripts (`Figure_1_2.py`, `Figure_3.py`, `Figure_4_5.py`, `Figure_6.py`, `Figure_7.py`, `Figure_8.py`)
- `data/`: external reference arrays (folder tracked, files excluded)
- `outputs/`: curated checkpoint artifacts (`.pkl`) tracked for reproducibility

## Data Policy
Large data should be hosted externally and linked (Google Drive, Zenodo, Hugging Face Datasets, etc.).

Download source:
`https://drive.google.com/drive/folders/1Ccjkm_PXA_smmYGG9z9tAyHpXScJ2BFc?usp=drive_link`

How to use:
1. Download data files from the Google Drive folder above.
2. Follow `docs/DATA.md` for exact filenames/target paths/checksum values.
3. Validate placement and integrity:
   `python scripts/check_data.py`

Optional CLI download (server-friendly):
```bash
pip install -U gdown
gdown 1s_AGJq0PfWXpZX6PRB2UCLEO7lZO76-2 -O data/PP2D_bcN_l10p1_l2100_l310_om-2_2_nx101.npy
gdown 1aAQKeJau7u-KwAI-RoH17BBvqZRIebnV -O data/PS3D_a1_b1_c0p5_al-2_be2_sig0p1I_nx201.npy
```

Repository tracking policy:
- `data/`: folder only (dataset files are not tracked)
- `outputs/`: curated checkpoint `.pkl` files are tracked
- `plots/figures/`: final `.png`/`.pdf` figures are tracked

Quick data validation:
```bash
python scripts/check_data.py
```

## Training
YAML format:
- `problem`: PDE/dynamics params
- `train`: optimization params (`Ni`, `num_iters`, `num_epochs`, `lr`, `seed`, `unit`, ...)

Run from repo root.

Path planning:
```bash
python scripts/train_path_planning.py --config configs/path_planning_default.yaml --outdir outputs/path_run
```
Default output name:
`PI_N{Ni}_it{num_iters}_ep{num_epochs}_seed{seed}.pkl`

Publisher-subscriber (PI):
```bash
python scripts/train_ps.py --config configs/ps_default.yaml --mode pi --outdir outputs/ps_run/3d
```

Publisher-subscriber (Direct):
```bash
python scripts/train_ps.py --config configs/ps_default.yaml --mode direct --outdir outputs/ps_run/3d
```

PS default output names:
- PI: `PI_N{Ni}_it{num_iters}_ep{num_epochs}_epsilon{eps}_seed{seed}.pkl`
- Direct: `Direct_N{Ni}_ep{num_iters*num_epochs}_epsilon{eps}_seed{seed}.pkl`

Optional PS reference evaluation:
```bash
python scripts/train_ps.py --config configs/ps_default.yaml --mode pi --fdm_path /path/to/fdm.npy
python scripts/train_ps.py --config configs/ps_default.yaml --mode direct --fdm_path /path/to/fdm.npy
```

## Figures
After checkpoints and reference data are ready:
```bash
python plots/Figure_1_2.py
python plots/Figure_3.py
python plots/Figure_4_5.py
python plots/Figure_6.py
python plots/Figure_7.py
python plots/Figure_8.py
```

Figures are saved to `plots/figures/`.

Note:
- `Figure_8.py` is an analytical Hamiltonian visualization and does not require training checkpoints.

## Output Schema
Path planning:
- Top-level keys: `params`, `info`

Publisher-subscriber:
- Top-level keys: `params`, `l2_history`, `train_time`, `info`

`info` keys are standardized and mode-specific where needed (PI vs Direct).
