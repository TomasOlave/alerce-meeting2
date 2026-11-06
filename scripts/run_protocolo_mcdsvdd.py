"""Evalúa MCDSVDD con el protocolo de Pérez-Carrasco et al. (2023): una clase como anomalía.

Para cada clase y cada uno de los 5 folds: se entrena el autoencoder y MCDSVDD sin esa clase, y se
mide el AUROC para encontrarla en un test con 10% de anomalías. Ver src/ic_project/protocolo.py.

Con --filtro se usan solo los objetos que pasan los criterios de calidad del código del paper:
- detecciones: al menos 6 detecciones en g y en r;
- completo: además, magnitudes corregidas en las dos bandas y ninguna detección dudosa.

Con --con-gmm y --con-dagmm se evalúan también el GMM (-log-verosimilitud sobre el latente del
autoencoder) y DAGMM (energía) con el mismo protocolo. Con --features pipeline entran las 140
features del pipeline en vez de las 158 (sin FEATURES_EXCLUIDAS).

Uso:
    uv run python scripts/run_protocolo_mcdsvdd.py
    uv run python scripts/run_protocolo_mcdsvdd.py --filtro detecciones
    uv run python scripts/run_protocolo_mcdsvdd.py --features pipeline --con-gmm --con-dagmm
    uv run python scripts/run_protocolo_mcdsvdd.py --folds 2 --clases DSCT LPV

Guarda en results/protocolo_mcdsvdd[...]/ el AUROC de cada fold y el resumen por clase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ic_project.config import FEATURES_EXCLUIDAS, RESULTADOS, Z_DIM  # noqa: E402
from ic_project.datos import (  # noqa: E402
    cargar_features_y_oids,
    rutas_tablas,
    unir_features_etiquetas,
)
from ic_project.protocolo import (  # noqa: E402
    correr_protocolo,
    filtro_detecciones,
    filtro_flags,
    resumir_protocolo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--clases", nargs="*", default=None, help="por defecto, las cinco")
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument(
        "--filtro", choices=("ninguno", "detecciones", "completo"), default="ninguno"
    )
    parser.add_argument("--features", choices=("todas", "pipeline"), default="todas")
    parser.add_argument("--con-gmm", action="store_true")
    parser.add_argument("--con-dagmm", action="store_true")
    args = parser.parse_args()
    sufijo = "" if args.filtro == "ninguno" else f"_filtro_{args.filtro}"
    if args.features == "pipeline":
        sufijo += "_140features"
    if args.con_gmm or args.con_dagmm:
        sufijo += "_con" + "".join(["_gmm"] * args.con_gmm + ["_dagmm"] * args.con_dagmm)
    salida = RESULTADOS / f"protocolo_mcdsvdd{sufijo}"
    salida.mkdir(parents=True, exist_ok=True)

    df_features, df_oids = cargar_features_y_oids()
    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    if args.filtro != "ninguno":
        pasa = filtro_detecciones(x)
        if args.filtro == "completo":
            columnas = ["oid", "fid", "corrected", "dubious"]
            lcs = pd.read_parquet(rutas_tablas()["lcs"], columns=columnas)
            pasa = pasa & filtro_flags(oids, lcs[lcs["oid"].isin(set(oids))])
        conteo = pd.DataFrame({"total": y.value_counts(), "pasan": y[pasa].value_counts()})
        conteo["%"] = (100 * conteo["pasan"] / conteo["total"]).round(1)
        print(f"filtro {args.filtro}: pasan {pasa.sum()} de {len(y)} objetos")
        print(conteo.to_string(), "\n")
        x, y = x[pasa].reset_index(drop=True), y[pasa].reset_index(drop=True)

    resultados = correr_protocolo(
        x, y.to_numpy(), clases=args.clases, n_folds=args.folds, z_dim=args.z_dim,
        excluir=FEATURES_EXCLUIDAS if args.features == "pipeline" else (),
        con_gmm=args.con_gmm, con_dagmm=args.con_dagmm,
    )
    resultados.to_csv(salida / "auroc_por_fold.csv", index=False)
    resumen = resumir_protocolo(resultados)
    resumen.to_csv(salida / "resumen.csv")
    print("\nAUROC por clase anómala (media y desviación sobre folds) y Tabla 1 del paper:")
    print(resumen.round(3).to_string())
    print(f"\nResultados en {salida}")


if __name__ == "__main__":
    main()
