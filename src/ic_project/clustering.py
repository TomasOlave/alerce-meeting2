"""GMM sobre el cuello de botella y métricas pedidas para el meeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def ajustar_gmm(
    z_train: np.ndarray,
    n_componentes: int,
    semilla: int = 42,
    covarianza: str = "full",
    n_init: int = 5,
) -> GaussianMixture:
    """Ajusta el GMM solo con el latente de entrenamiento."""
    if n_componentes < 1:
        raise ValueError("n_componentes debe ser >= 1")
    modelo = GaussianMixture(
        n_components=n_componentes,
        covariance_type=covarianza,
        random_state=semilla,
        n_init=n_init,
        max_iter=300,
    )
    modelo.fit(z_train)
    return modelo


def metricas_espacio(
    z: np.ndarray,
    etiquetas_pred: np.ndarray,
    etiquetas_real: np.ndarray,
) -> dict[str, float]:
    """Silhouette / DB de clusters y de clases; ARI y NMI contra la etiqueta real."""
    out: dict[str, float] = {}
    if len(np.unique(etiquetas_pred)) > 1 and len(z) > 2:
        out["silhouette_clusters"] = float(silhouette_score(z, etiquetas_pred))
        out["davies_bouldin_clusters"] = float(davies_bouldin_score(z, etiquetas_pred))
    else:
        out["silhouette_clusters"] = float("nan")
        out["davies_bouldin_clusters"] = float("nan")
    if len(np.unique(etiquetas_real)) > 1 and len(z) > 2:
        out["silhouette_clases"] = float(silhouette_score(z, etiquetas_real))
        out["davies_bouldin_clases"] = float(davies_bouldin_score(z, etiquetas_real))
    else:
        out["silhouette_clases"] = float("nan")
        out["davies_bouldin_clases"] = float("nan")
    out["ari"] = float(adjusted_rand_score(etiquetas_real, etiquetas_pred))
    out["nmi"] = float(normalized_mutual_info_score(etiquetas_real, etiquetas_pred))
    return out


def tabla_cluster_clase(etiquetas_real: np.ndarray, etiquetas_pred: np.ndarray) -> pd.DataFrame:
    """Conteo clase real (filas) contra cluster o clase predicha (columnas)."""
    real = pd.Series(np.asarray(etiquetas_real), name="clase")
    pred = pd.Series(np.asarray(etiquetas_pred), name="cluster")
    return pd.crosstab(real, pred)
