"""Puntos 1 y 2 del meeting, y primera versión del Multi-Class Deep SVDD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ic_project.clustering import ajustar_gmm, metricas_espacio, tabla_cluster_clase
from ic_project.config import RESULTADOS, SEMILLA
from ic_project.datos import cargar_features_y_oids, unir_features_etiquetas
from ic_project.importancia import importancia_random_forest
from ic_project.modelos import (
    asignar_mcdsvdd,
    entrenar_autoencoder,
    entrenar_mcdsvdd,
    latente,
    mse_reconstruccion,
)
from ic_project.preprocesamiento import preparar_conjuntos
from ic_project.visualizacion import (
    barra_importancia,
    curva_perdida,
    heatmap_tabla,
    proyectar_2d,
    scatter_clases,
    scatter_clusters,
)


def _codificar(
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    clases = sorted(pd.unique(y_train))
    mapa = {clase: i for i, clase in enumerate(clases)}

    def enc(serie: pd.Series) -> np.ndarray:
        cod = serie.map(mapa)
        if cod.isna().any():
            raise ValueError("Hay clases en val o test que no están en entrenamiento")
        return cod.to_numpy(dtype=np.int64)

    return enc(y_train), enc(y_val), enc(y_test), [str(c) for c in clases]


def _log(verbose: bool, mensaje: str) -> None:
    if verbose:
        print(mensaje)


def correr_meeting2(
    *,
    raiz: Path | None = None,
    tablas: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    z_dim: int = 8,
    epocas_ae: int = 40,
    epocas_mcd: int = 25,
    batch: int = 128,
    semilla: int = SEMILLA,
    n_componentes: int | None = None,
    n_init_gmm: int = 5,
    con_importancia: bool = True,
    directorio: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Entrena autoencoder, GMM y MCDSVDD. Las métricas salen del conjunto de test.

    `tablas` es `(features, oids)`. Si no se pasa, se leen los parquet del repo.
    El GMM se ajusta solo con el latente de entrenamiento. UMAP/t-SNE se usa
    únicamente para dibujar el test: no entra al modelo.
    """
    if tablas is None:
        df_features, df_oids = cargar_features_y_oids(raiz)
    else:
        df_features, df_oids = tablas

    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    conjuntos, prep, descartadas = preparar_conjuntos(x, y, oids, semilla=semilla)
    y_train_i, y_val_i, y_test_i, clases = _codificar(
        conjuntos["y_train"], conjuntos["y_val"], conjuntos["y_test"]
    )
    n_comp = len(clases) if n_componentes is None else n_componentes
    salida = Path(directorio) if directorio is not None else RESULTADOS
    salida.mkdir(parents=True, exist_ok=True)

    tamanos = {
        "train": int(len(conjuntos["y_train"])),
        "val": int(len(conjuntos["y_val"])),
        "test": int(len(conjuntos["y_test"])),
        "features": int(conjuntos["x_train"].shape[1]),
        "descartadas": len(descartadas),
    }
    _log(
        verbose,
        f"Split train/val/test = {tamanos['train']}/{tamanos['val']}/{tamanos['test']} | "
        f"features = {tamanos['features']} (descartadas {tamanos['descartadas']})",
    )

    _log(verbose, "Entrenando autoencoder...")
    ae, hist_ae = entrenar_autoencoder(
        conjuntos["x_train"],
        conjuntos["x_val"],
        z_dim=z_dim,
        n_epocas=epocas_ae,
        batch=batch,
        semilla=semilla,
    )
    z_train = latente(ae, conjuntos["x_train"])
    z_test = latente(ae, conjuntos["x_test"])
    mse_test = mse_reconstruccion(ae, conjuntos["x_test"])
    _log(
        verbose,
        f"Autoencoder: {len(hist_ae.val)} épocas, "
        f"MSE val = {hist_ae.val[int(np.argmin(hist_ae.val))]:.4f}, MSE test = {mse_test:.4f}",
    )

    xy_ae = proyectar_2d(z_test, semilla=semilla)
    figuras = {
        "ae_perdida": curva_perdida(
            hist_ae.train,
            hist_ae.val,
            "Autoencoder: error de reconstrucción",
            salida / "ae_perdida.png",
        ),
        "ae_clases": scatter_clases(
            xy_ae,
            conjuntos["y_test"],
            "Espacio latente del autoencoder (test)",
            salida / "ae_latente_clases.png",
        ),
    }

    _log(verbose, f"Ajustando GMM con {n_comp} componentes sobre el latente de train...")
    gmm = ajustar_gmm(z_train, n_comp, semilla=semilla, n_init=n_init_gmm)
    pred_gmm = gmm.predict(z_test)
    metricas_gmm = metricas_espacio(z_test, pred_gmm, conjuntos["y_test"].to_numpy())
    tabla_gmm = tabla_cluster_clase(conjuntos["y_test"].to_numpy(), pred_gmm)
    figuras["gmm_clusters"] = scatter_clusters(
        xy_ae,
        pred_gmm,
        "Clusters GMM en el latente (test)",
        salida / "gmm_clusters.png",
    )
    figuras["gmm_tabla"] = heatmap_tabla(
        tabla_gmm,
        "Test: clase real vs componente GMM",
        salida / "gmm_clase_cluster.png",
    )

    _log(verbose, "Entrenando MCDSVDD a partir del encoder...")
    mcd, hist_mcd = entrenar_mcdsvdd(
        conjuntos["x_train"],
        y_train_i,
        conjuntos["x_val"],
        y_val_i,
        autoencoder=ae,
        z_dim=z_dim,
        n_epocas=epocas_mcd,
        batch=batch,
        semilla=semilla,
    )
    z_mcd = latente(mcd, conjuntos["x_test"])
    pred_mcd_i = asignar_mcdsvdd(mcd, conjuntos["x_test"])
    pred_mcd = np.asarray(clases)[pred_mcd_i]
    metricas_mcd = metricas_espacio(z_mcd, pred_mcd, conjuntos["y_test"].to_numpy())
    tabla_mcd = tabla_cluster_clase(conjuntos["y_test"].to_numpy(), pred_mcd)
    tabla_mcd.columns.name = "centro"
    xy_mcd = proyectar_2d(z_mcd, semilla=semilla)
    _log(
        verbose,
        f"MCDSVDD: {len(hist_mcd.val)} épocas, "
        f"distancia val = {hist_mcd.val[int(np.argmin(hist_mcd.val))]:.4f}",
    )
    figuras["mcd_perdida"] = curva_perdida(
        hist_mcd.train,
        hist_mcd.val,
        "MCDSVDD: distancia al centro de la clase",
        salida / "mcdsvdd_perdida.png",
    )
    figuras["mcd_clases"] = scatter_clases(
        xy_mcd,
        conjuntos["y_test"],
        "Espacio latente MCDSVDD (test, clase real)",
        salida / "mcdsvdd_latente_clases.png",
    )
    figuras["mcd_pred"] = scatter_clases(
        xy_mcd,
        pred_mcd,
        "Espacio latente MCDSVDD (test, centro más cercano)",
        salida / "mcdsvdd_latente_pred.png",
    )
    figuras["mcd_tabla"] = heatmap_tabla(
        tabla_mcd,
        "Test: clase real vs centro MCDSVDD",
        salida / "mcdsvdd_clase_cluster.png",
    )

    comparacion = pd.DataFrame({"GMM": metricas_gmm, "MCDSVDD": metricas_mcd}).T
    comparacion.index.name = "modelo"
    comparacion.to_csv(salida / "metricas_test.csv")
    tabla_gmm.to_csv(salida / "tabla_gmm_test.csv")
    tabla_mcd.to_csv(salida / "tabla_mcdsvdd_test.csv")

    importancia = None
    if con_importancia:
        _log(verbose, "Random Forest de importancia, solo con train...")
        importancia = importancia_random_forest(
            conjuntos["x_train"],
            conjuntos["y_train"].to_numpy(),
            list(prep.columnas_),
            semilla=semilla,
        )
        importancia.to_csv(salida / "importancia_rf.csv", index=False)
        figuras["importancia"] = barra_importancia(
            importancia,
            "Features más importantes (Random Forest, train)",
            salida / "importancia_rf.png",
        )

    torch.save(
        {"state": ae.state_dict(), "in_dim": int(conjuntos["x_train"].shape[1]), "z_dim": z_dim},
        salida / "autoencoder.pt",
    )
    torch.save(
        {
            "state": mcd.state_dict(),
            "clases": clases,
            "in_dim": int(conjuntos["x_train"].shape[1]),
            "z_dim": z_dim,
        },
        salida / "mcdsvdd.pt",
    )
    resumen = {
        "tamanos": tamanos,
        "clases": clases,
        "z_dim": z_dim,
        "mse_test_ae": mse_test,
        "epocas_ae": len(hist_ae.val),
        "epocas_mcdsvdd": len(hist_mcd.val),
        "metricas_gmm": metricas_gmm,
        "metricas_mcdsvdd": metricas_mcd,
    }
    (salida / "resumen_meeting2.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    _log(verbose, "\nMétricas en test:")
    _log(verbose, comparacion.round(3).to_string())
    _log(verbose, f"\nFiguras y tablas en {salida}")

    return {
        "comparacion": comparacion,
        "metricas_gmm": metricas_gmm,
        "metricas_mcdsvdd": metricas_mcd,
        "tabla_gmm": tabla_gmm,
        "tabla_mcdsvdd": tabla_mcd,
        "importancia": importancia,
        "figuras": figuras,
        "tamanos": tamanos,
        "clases": clases,
        "mse_test_ae": mse_test,
        "descartadas": descartadas,
        "directorio": salida,
    }
