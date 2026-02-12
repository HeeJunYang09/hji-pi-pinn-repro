from __future__ import annotations

import hashlib
from pathlib import Path


REQUIRED = [
    {
        "label": "Path planning reference",
        "candidates": ["PP2D_bcN_l10p1_l2100_l310_om-2_2_nx101.npy"],
        "sha256": "b311e88a4491d3243f73a62dcfc5729df621a569de03c47004eb8fa4168e4f3a",
    },
    {
        "label": "PS 3D FDM reference",
        "candidates": ["PS3D_a1_b1_c0p5_al-2_be2_sig0p1I_nx201.npy"],
        "sha256": "a27c05b41e51fa5ef319ae959a661d94a3380439cf76c3d1c0b1302b4367922c",
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    print(f"[info] repo root: {root}")
    print(f"[info] data dir : {data_dir}")

    missing = []
    bad_hash = []

    for item in REQUIRED:
        label = item["label"]
        candidates = item["candidates"]
        expected_hash = item["sha256"]

        existing = None
        for name in candidates:
            p = data_dir / name
            if p.exists():
                existing = p
                break

        if existing is None:
            missing.append((label, candidates))
            continue

        got = sha256_of(existing)
        if got != expected_hash:
            bad_hash.append((label, existing.name, expected_hash, got))
        else:
            print(f"[ok] {label}: {existing.name}")

    if missing:
        print("\n[missing] files not found under data/:")
        for label, candidates in missing:
            print(f"  - {label}")
            print(f"    candidates: {', '.join(candidates)}")

        print("\n[hint] similar files found:")
        for pat in ("*PP2D*nx101*.npy", "*PS3D*a1_b1*c0p5*.npy"):
            found = list(data_dir.glob(pat))
            if found:
                for p in found:
                    print(f"  - {p.name}")
            else:
                print(f"  - (none for pattern {pat})")

    if bad_hash:
        print("\n[hash mismatch] verify download integrity:")
        for label, name, exp, got in bad_hash:
            print(f"  - {label}: {name}")
            print(f"    expected: {exp}")
            print(f"    got     : {got}")

    if not missing and not bad_hash:
        print("\n[data check] passed.")
        return 0

    print("\n[data check] failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
