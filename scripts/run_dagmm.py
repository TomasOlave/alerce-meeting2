"""Entrena DAGMM (autoencoder y GMM juntos) y lo compara con el pipeline actual (AE y después GMM).

Uso:
    uv run python scripts/run_dagmm.py
    uv run python scripts/run_dagmm.py --lambda-energia 1 --desde-ae

Todo se ajusta con train y las métricas son de test. Guarda en results/dagmm/ la tabla de
métricas, la curva de pérdida, el latente de test (UMAP) por clase y por componente, la tabla
clase contra componente y el modelo entrenado.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from ic_project.clustering import ajustar_gmm, metricas_espacio, tabla_cluster_clase  # noqa: E402
from ic_project.config import EPOCAS_AE, RESULTADOS, SEMILLA, Z_DIM  # noqa: E402
from ic_project.dagmm import entrenar_dagmm, salidas_dagmm  # noqa: E402
from ic_project.datos import cargar_features_y_oids, unir_features_etiquetas  # noqa: E402
from ic_project.modelos import entrenar_autoencoder, latente, mse_reconstruccion  # noqa: E402
from ic_project.preprocesamiento import preparar_conjuntos  # noqa: E402
from ic_project.visualizacion import (  # noqa: E402
    curva_perdida,
    heatmap_tabla,
    proyectar_2d,
    scatter_clases,
    scatter_clusters,
)


def f1_componente_a_clase(
    comp_train: np.ndarray, y_train: np.ndarray, comp_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Etiqueta cada componente con la clase mayoritaria de sus objetos de train y mide F1 macro
    en test. Las etiquetas solo se usan para evaluar."""
    mapa = pd.Series(y_train).groupby(comp_train).agg(lambda s: s.mode()[0])
    pred = pd.Series(comp_test).map(mapa).fillna("sin componente").to_numpy()
    return float(f1_score(y_test, pred, average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5, help="componentes de la mezcla")
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument("--lambda-energia", type=float, default=0.1)
    parser.add_argument("--lambda-cov", type=float, default=0.005)
    parser.add_argument("--epocas", type=int, default=200)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--desde-ae", action="store_true", help="partir de los pesos del autoencoder ya entrenado"
    )
    args = parser.parse_args()
    salida = RESULTADOS / "dagmm"
    salida.mkdir(parents=True, exist_ok=True)

    df_features, df_oids = cargar_features_y_oids()
    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    c, _, _ = preparar_conjuntos(x, y, oids, semilla=SEMILLA, solo_periodo_color=False)
    x_tr, x_va, x_te = c["x_train"], c["x_val"], c["x_test"]
    y_tr, y_te = c["y_train"].to_numpy(), c["y_test"].to_numpy()

    print("1) Pipeline actual: autoencoder solo para reconstruir y después GMM (EM)")
    ae, _ = entrenar_autoencoder(
        x_tr, x_va, z_dim=args.z_dim, n_epocas=EPOCAS_AE, semilla=SEMILLA
    )
    z_tr, z_te = latente(ae, x_tr), latente(ae, x_te)
    gmm = ajustar_gmm(z_tr, args.k, semilla=SEMILLA)
    pred_base = gmm.predict(z_te)
    filas = {
        "AE y después GMM": {
            **metricas_espacio(z_te, pred_base, y_te),
            "f1_componente_clase": f1_componente_a_clase(gmm.predict(z_tr), y_tr, pred_base, y_te),
            "mse_reconstruccion": mse_reconstruccion(ae, x_te),
        }
    }

    origen = "desde el autoencoder" if args.desde_ae else "desde cero"
    print(f"2) DAGMM {origen}: K={args.k}, lambda_energia={args.lambda_energia}")
    modelo, hist = entrenar_dagmm(
        x_tr,
        x_va,
        n_componentes=args.k,
        z_dim=args.z_dim,
        autoencoder=ae if args.desde_ae else None,
        n_epocas=args.epocas,
        batch=args.batch,
        lambda_energia=args.lambda_energia,
        lambda_cov=args.lambda_cov,
        semilla=SEMILLA,
    )
    print(f"   {len(hist.val)} épocas")
    s_tr, s_te = salidas_dagmm(modelo, x_tr), salidas_dagmm(modelo, x_te)
    pred_dagmm = s_te["gamma"].argmax(axis=1)
    filas["DAGMM"] = {
        **metricas_espacio(s_te["z"], pred_dagmm, y_te),
        "f1_componente_clase": f1_componente_a_clase(
            s_tr["gamma"].argmax(axis=1), y_tr, pred_dagmm, y_te
        ),
        "mse_reconstruccion": mse_reconstruccion(modelo.compresion, x_te),
    }

    tabla = pd.DataFrame(filas).T
    tabla.index.name = "modelo"
    tabla.to_csv(salida / "metricas_test.csv")
    print("\nMétricas en test:")
    print(tabla.round(3).to_string())
    tabla_dagmm = tabla_cluster_clase(y_te, pred_dagmm)
    print("\nDAGMM, clase (filas) contra componente (columnas):")
    print(tabla_dagmm.to_string())

    curva_perdida(hist.train, hist.val, "DAGMM: pérdida total", salida / "perdida.png")
    xy = proyectar_2d(s_te["z"], semilla=SEMILLA)
    titulo = "Latente DAGMM (test), color = "
    scatter_clases(xy, y_te, titulo + "clase", salida / "latente_clases.png")
    scatter_clusters(xy, pred_dagmm, titulo + "componente", salida / "latente_componentes.png")
    heatmap_tabla(
        tabla_dagmm, "Test: clase real vs componente DAGMM", salida / "clase_componente.png"
    )
    torch.save(modelo.state_dict(), salida / "dagmm.pt")
    print(f"\nResultados en {salida}")


if __name__ == "__main__":
    main()
