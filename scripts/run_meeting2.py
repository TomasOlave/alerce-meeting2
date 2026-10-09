"""Corre autoencoder, GMM y la primera versión de MCDSVDD."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ic_project.experimento import correr_meeting2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--epocas-ae", type=int, default=40)
    parser.add_argument("--epocas-mcd", type=int, default=25)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--sin-importancia", action="store_true")
    args = parser.parse_args()
    correr_meeting2(
        z_dim=args.z_dim,
        epocas_ae=args.epocas_ae,
        epocas_mcd=args.epocas_mcd,
        batch=args.batch,
        con_importancia=not args.sin_importancia,
    )


if __name__ == "__main__":
    main()
