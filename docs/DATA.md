# Data Files (External Hosting)

This repository ignores large artifacts under `data/` and `outputs/`.
Host large files externally (Google Drive / Zenodo / Hugging Face Datasets / etc.) and document links here.

Google Drive folder (official):
- https://drive.google.com/drive/folders/1Ccjkm_PXA_smmYGG9z9tAyHpXScJ2BFc?usp=drive_link

## Required Files

1. Path planning reference
- Local target path: `data/PP2D_bcN_l10p1_l2100_l310_om-2_2_nx101.npy`
- Used by: `plots/Figure_1_2.py`
- Size (approx): `0.4 MB`
- Download link: Google Drive folder above
- SHA256: `b311e88a4491d3243f73a62dcfc5729df621a569de03c47004eb8fa4168e4f3a`

2. Publisher-subscriber 3D FDM reference
- Local target path: `data/PS3D_a1_b1_c0p5_al-2_be2_sig0p1I_nx201.npy`
- Used by: `plots/Figure_3.py`, `plots/Figure_4_5.py`
- Size (approx): `372 MB`
- Download link: Google Drive folder above
- SHA256: `a27c05b41e51fa5ef319ae959a661d94a3380439cf76c3d1c0b1302b4367922c`

## Suggested Workflow

1. Download files from links above.
2. Place them under `data/` with exactly the target filenames.
3. Verify checksums.
4. Run training/plot scripts from repo root.

If files are still not detected, run:
```bash
python scripts/check_data.py
```
This reports missing filenames and hash mismatches.

Checksum command:
```bash
sha256sum data/PP2D_bcN_l10p1_l2100_l310_om-2_2_nx101.npy data/PS3D_a1_b1_c0p5_al-2_be2_sig0p1I_nx201.npy
```
