"""Prototipos y outliers a partir de GMM y MCDSVDD."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ic_project.modelos import CodificadorMCDSVDD, asignar_mcdsvdd, latente, puntaje_mcdsvdd


def prototipos_gmm(
    gmm: GaussianMixture, z: np.ndarray, oids: pd.Series, y: pd.Series, k: int = 1
) -> pd.DataFrame:
    """Objetos más cercanos a cada media de componente."""
    filas = []
    dist = ((z[:, None, :] - gmm.means_[None, :, :]) ** 2).sum(axis=2)
    for c in range(gmm.n_components):
        orden = np.argsort(dist[:, c])[:k]
        for rango, i in enumerate(orden):
            filas.append(
                {
                    "modelo": "gmm",
                    "tipo": "prototipo",
                    "componente": c,
                    "rango": rango,
                    "oid": oids.iloc[i],
                    "clase": y.iloc[i],
                    "distancia": float(dist[i, c]),
                }
            )
    return pd.DataFrame(filas)


def outliers_gmm(
    gmm: GaussianMixture, z: np.ndarray, oids: pd.Series, y: pd.Series, k: int = 8
) -> pd.DataFrame:
    """Menor verosimilitud = más periféricos."""
    ll = gmm.score_samples(z)
    orden = np.argsort(ll)[:k]
    return pd.DataFrame(
        {
            "modelo": "gmm",
            "tipo": "outlier",
            "oid": oids.iloc[orden].to_numpy(),
            "clase": y.iloc[orden].to_numpy(),
            "loglik": ll[orden],
            "score_anomalia": -ll[orden],
        }
    )


def prototipos_mcdsvdd(
    modelo: CodificadorMCDSVDD,
    x: np.ndarray,
    oids: pd.Series,
    y: pd.Series,
    k: int = 1,
) -> pd.DataFrame:
    z = latente(modelo, x)
    c = modelo.centros.detach().cpu().numpy()
    filas = []
    dist = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    for j in range(c.shape[0]):
        orden = np.argsort(dist[:, j])[:k]
        for rango, i in enumerate(orden):
            filas.append(
                {
                    "modelo": "mcdsvdd",
                    "tipo": "prototipo",
                    "componente": j,
                    "rango": rango,
                    "oid": oids.iloc[i],
                    "clase": y.iloc[i],
                    "distancia": float(dist[i, j]),
                }
            )
    return pd.DataFrame(filas)


def outliers_mcdsvdd(
    modelo: CodificadorMCDSVDD,
    x: np.ndarray,
    oids: pd.Series,
    y: pd.Series,
    k: int = 8,
) -> pd.DataFrame:
    score = puntaje_mcdsvdd(modelo, x)
    pred = asignar_mcdsvdd(modelo, x)
    orden = np.argsort(-score)[:k]
    return pd.DataFrame(
        {
            "modelo": "mcdsvdd",
            "tipo": "outlier",
            "oid": oids.iloc[orden].to_numpy(),
            "clase": y.iloc[orden].to_numpy(),
            "cluster": pred[orden],
            "score_anomalia": score[orden],
        }
    )
