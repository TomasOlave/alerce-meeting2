"""Evalúa MCDSVDD con el protocolo de Pérez-Carrasco et al. (2023): una clase como anomalía.

Para cada clase y cada uno de los 5 folds: se entrena el autoencoder y MCDSVDD sin esa clase, y se
mide el AUROC para encontrarla en un test con 10% de anomalías. Ver src/ic_project/protocolo.py.

Con --filtro se usan solo los objetos que pasan los criterios de calidad del código del paper:
- detecciones: al menos 6 detecciones en g y en r;
- completo: además, magnitudes corregidas en las dos bandas y ninguna detección dudosa.

Con --con-gmm y --con-dagmm se evalúan también el GMM (-log-verosimilitud sobre el latente del
autoencoder) y DAGMM (energía) con el mismo protocolo. Con --con-gmm-clase, el GMM por clase sobre
el latente del autoencoder y sobre el de MCDSVDD; con --con-vade, VaDE sin clases y con clases
(-ELBO). Con --features pipeline entran las 140
features del pipeline en vez de las 158 (sin FEATURES_EXCLUIDAS).

Uso:
    uv run python scripts/run_protocolo_mcdsvdd.py
    uv run python scripts/run_protocolo_mcdsvdd.py --filtro detecciones
    uv run python scripts/run_protocolo_mcdsvdd.py --features pipeline --con-gmm --con-dagmm
    uv run python scripts/run_protocolo_mcdsvdd.py --features pipeline --con-gmm --con-gmm-clase \
        --con-vade
    uv run python scripts/run_protocolo_mcdsvdd.py --folds 2 --clases DSCT LPV

Guarda en results/protocolo_mcdsvdd[...]/ el AUROC y el AUPRC de cada fold (auroc_por_fold.csv)
y el resumen por clase (resumen.csv con AUROC, resumen_auprc.csv con AUPRC).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ic_project.config import EPOCAS_VADE, FEATURES_EXCLUIDAS, RESULTADOS, Z_DIM  # noqa: E402
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
    parser.add_argument("--con-gmm-clase", action="store_true")
    parser.add_argument("--con-vade", action="store_true")
    parser.add_argument("--epocas-vade", type=int, default=EPOCAS_VADE)
    args = parser.parse_args()
    sufijo = "" if args.filtro == "ninguno" else f"_filtro_{args.filtro}"
    if args.features == "pipeline":
        sufijo += "_140features"
    extras = {
        "gmm": args.con_gmm,
        "dagmm": args.con_dagmm,
        "gmm_clase": args.con_gmm_clase,
        "vade": args.con_vade,
    }
    if any(extras.values()):
        sufijo += "_con" + "".join(f"_{nombre}" for nombre, activo in extras.items() if activo)
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
        con_gmm=args.con_gmm, con_dagmm=args.con_dagmm, con_gmm_clase=args.con_gmm_clase,
        con_vade=args.con_vade, epocas_vade=args.epocas_vade,
    )
    resultados.to_csv(salida / "auroc_por_fold.csv", index=False)
    resumen = resumir_protocolo(resultados)
    resumen.to_csv(salida / "resumen.csv")
    resumen_auprc = resumir_protocolo(resultados, metrica="auprc")
    resumen_auprc.to_csv(salida / "resumen_auprc.csv")
    print("\nAUROC por clase anómala (media y desviación sobre folds) y Tabla 1 del paper:")
    print(resumen.round(3).to_string())
    print("\nAUPRC por clase anómala (al azar da 0.1, la fracción de anomalías en TS2):")
    print(resumen_auprc.round(3).to_string())
    print(f"\nResultados en {salida}")


if __name__ == "__main__":
    main()
