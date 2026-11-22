"""Compara las variantes del GMM en el split del meeting (70/15/15, 140 features), en test.

Variantes:
- GMM: sin etiquetas, sobre el latente del autoencoder (el del pipeline, gmm.py).
- GMM por clase (AE): un GMM por clase sobre el latente del autoencoder (gmm_clase.py).
- MCDSVDD: centro más cercano (referencia, también usa las clases).
- GMM por clase (MCDSVDD): un GMM por clase sobre el latente de MCDSVDD.
- VaDE: autoencoder variacional con prior GMM, sin clases (vade.py, el modelo del repo del paper).
- VaDE con clases: un componente del prior por clase.

Para cada una: ARI, NMI, silhouette y Davies-Bouldin (clustering.metricas_espacio) y F1 macro.
En las variantes sin etiquetas, el F1 etiqueta cada componente con la clase mayoritaria de sus
estrellas de train (las etiquetas solo se usan para evaluar).

Uso:
    uv run python scripts/run_gmm_mejorado.py
    uv run python scripts/run_gmm_mejorado.py --sin-vade

Guarda en results/gmm_mejorado/ la tabla de métricas, las tablas clase contra grupo, el K elegido
por clase y las curvas de pérdida de VaDE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from ic_project.clustering import ajustar_gmm, metricas_espacio, tabla_cluster_clase  # noqa: E402
from ic_project.config import (  # noqa: E402
    EPOCAS_AE,
    EPOCAS_VADE,
    K_MAX_GMM_CLASE,
    N_COMPONENTES_GMM,
    RESULTADOS,
    SEMILLA,
    Z_DIM,
)
from ic_project.datos import cargar_features_y_oids, unir_features_etiquetas  # noqa: E402
from ic_project.gmm_clase import GMMPorClase  # noqa: E402
from ic_project.modelos import (  # noqa: E402
    asignar_mcdsvdd,
    entrenar_autoencoder,
    entrenar_mcdsvdd,
    latente,
)
from ic_project.preprocesamiento import preparar_conjuntos  # noqa: E402
from ic_project.vade import entrenar_vade, salidas_vade  # noqa: E402
from ic_project.visualizacion import curva_perdida, heatmap_tabla  # noqa: E402


def f1_por_mayoria(
    grupo_train: np.ndarray, y_train: np.ndarray, grupo_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Cada grupo toma la clase mayoritaria de sus estrellas de train; F1 macro en test."""
    mapa = pd.Series(y_train).groupby(grupo_train).agg(lambda s: s.mode()[0])
    pred = pd.Series(grupo_test).map(mapa).fillna("sin grupo").to_numpy()
    return float(f1_score(y_test, pred, average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument("--k-max", type=int, default=K_MAX_GMM_CLASE)
    parser.add_argument("--epocas-vade", type=int, default=EPOCAS_VADE)
    parser.add_argument("--sin-vade", action="store_true", help="solo las variantes del GMM")
    args = parser.parse_args()
    salida = RESULTADOS / "gmm_mejorado"
    salida.mkdir(parents=True, exist_ok=True)

    df_features, df_oids = cargar_features_y_oids()
    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    c, _, _ = preparar_conjuntos(x, y, oids, semilla=SEMILLA, solo_periodo_color=False)
    x_tr, x_va, x_te = c["x_train"], c["x_val"], c["x_test"]
    y_tr, y_va, y_te = (c[f"y_{s}"].to_numpy() for s in ("train", "val", "test"))
    clases = sorted(set(y_tr))
    mapa = {clase: i for i, clase in enumerate(clases)}
    y_tr_i = np.array([mapa[v] for v in y_tr])
    y_va_i = np.array([mapa[v] for v in y_va])
    nombres = np.asarray(clases)

    print(f"Split {len(y_tr)}/{len(y_va)}/{len(y_te)}, {x_tr.shape[1]} features")
    ae, _ = entrenar_autoencoder(x_tr, x_va, z_dim=args.z_dim, n_epocas=EPOCAS_AE, semilla=SEMILLA)
    z_tr, z_te = latente(ae, x_tr), latente(ae, x_te)
    mcd, _ = entrenar_mcdsvdd(
        x_tr, y_tr_i, x_va, y_va_i, autoencoder=ae, z_dim=args.z_dim, semilla=SEMILLA
    )
    zm_tr, zm_te = latente(mcd, x_tr), latente(mcd, x_te)

    filas: dict[str, dict] = {}
    tablas: dict[str, pd.DataFrame] = {}
    k_elegidos: dict[str, dict] = {}

    def registrar(nombre: str, z: np.ndarray, pred: np.ndarray, f1: float) -> None:
        filas[nombre] = {**metricas_espacio(z, pred, y_te), "f1_macro": f1}
        tablas[nombre] = tabla_cluster_clase(y_te, pred)
        print(f"  {nombre}: ARI {filas[nombre]['ari']:.3f} | F1 {f1:.3f}")

    print("GMM sin etiquetas (latente del autoencoder)")
    gmm = ajustar_gmm(z_tr, N_COMPONENTES_GMM, semilla=SEMILLA)
    pred = gmm.predict(z_te)
    registrar("GMM", z_te, pred, f1_por_mayoria(gmm.predict(z_tr), y_tr, pred, y_te))

    print("GMM por clase")
    for nombre, (a, b) in {"AE": (z_tr, z_te), "MCDSVDD": (zm_tr, zm_te)}.items():
        modelo = GMMPorClase(args.k_max, random_state=SEMILLA).fit(a, y_tr)
        pred = modelo.predict(b)
        f1 = float(f1_score(y_te, pred, average="macro"))
        registrar(f"GMM por clase ({nombre})", b, pred, f1)
        k_elegidos[nombre] = {str(k): v for k, v in modelo.k_.items()}
        print(f"    componentes por clase: {modelo.k_}")

    print("MCDSVDD (centro más cercano)")
    pred = nombres[asignar_mcdsvdd(mcd, x_te)]
    registrar("MCDSVDD", zm_te, pred, float(f1_score(y_te, pred, average="macro")))

    if not args.sin_vade:
        print("VaDE sin clases")
        vade, hist = entrenar_vade(
            x_tr,
            x_va,
            ae,
            n_componentes=N_COMPONENTES_GMM,
            z_dim=args.z_dim,
            n_epocas=args.epocas_vade,
            semilla=SEMILLA,
        )
        s_tr, s_te = salidas_vade(vade, x_tr), salidas_vade(vade, x_te)
        pred = s_te["componente"]
        registrar("VaDE", s_te["z"], pred, f1_por_mayoria(s_tr["componente"], y_tr, pred, y_te))
        curva_perdida(hist.train, hist.val, "VaDE: -ELBO", salida / "vade_perdida.png")

        print("VaDE con clases")
        vade_c, hist_c = entrenar_vade(
            x_tr,
            x_va,
            ae,
            y_train=y_tr_i,
            y_val=y_va_i,
            z_dim=args.z_dim,
            n_epocas=args.epocas_vade,
            semilla=SEMILLA,
        )
        s_te = salidas_vade(vade_c, x_te)
        pred = nombres[s_te["componente"]]
        registrar("VaDE con clases", s_te["z"], pred, float(f1_score(y_te, pred, average="macro")))
        curva_perdida(
            hist_c.train, hist_c.val, "VaDE con clases: -ELBO", salida / "vade_clases_perdida.png"
        )

    tabla = pd.DataFrame(filas).T
    tabla.index.name = "modelo"
    tabla.to_csv(salida / "metricas_test.csv")
    (salida / "k_por_clase.json").write_text(json.dumps(k_elegidos, indent=2), encoding="utf-8")
    for nombre, t in tablas.items():
        archivo = nombre.lower().replace(" ", "_").replace("(", "").replace(")", "")
        t.to_csv(salida / f"tabla_{archivo}.csv")
        heatmap_tabla(t, f"Test: clase real vs {nombre}", salida / f"tabla_{archivo}.png")
    print("\nMétricas en test:")
    print(tabla.round(3).to_string())
    print(f"\nResultados en {salida}")


if __name__ == "__main__":
    main()
