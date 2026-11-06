"""Corre el pipeline de notebooks/meeting2.ipynb: autoencoder, GMM y MCDSVDD con 158 features."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ic_project.config import EPOCAS_AE, Z_DIM  # noqa: E402
from ic_project.experimento import correr_meeting2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument("--epocas-ae", type=int, default=EPOCAS_AE)
    parser.add_argument("--epocas-mcd", type=int, default=2000, help="tope; para el early stopping")
    parser.add_argument("--batch", type=int, default=128, help="batch del autoencoder")
    parser.add_argument("--batch-mcd", type=int, default=1024, help="batch de MCDSVDD (paper)")
    parser.add_argument(
        "--sin-balancear",
        action="store_true",
        help="MCDSVDD con promedio simple en vez de la ecuación 5 del paper",
    )
    parser.add_argument("--sin-importancia", action="store_true")
    args = parser.parse_args()
    correr_meeting2(
        z_dim=args.z_dim,
        epocas_ae=args.epocas_ae,
        epocas_mcd=args.epocas_mcd,
        batch=args.batch,
        batch_mcd=args.batch_mcd,
        balancear_clases=not args.sin_balancear,
        con_importancia=not args.sin_importancia,
    )


if __name__ == "__main__":
    main()
