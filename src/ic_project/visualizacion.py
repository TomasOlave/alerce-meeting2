"""UMAP/t-SNE del latente y curvas de luz plegadas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from ic_project.config import COLORES_CLASE, RESULTADOS
from ic_project.datos import columna_magnitud


def proyectar_2d(z: np.ndarray, semilla: int = 42) -> np.ndarray:
    """UMAP si está instalado; si no, t-SNE."""
    if z.shape[1] == 2:
        return z
    try:
        import umap

        return umap.UMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=semilla
        ).fit_transform(z)
    except Exception:
        n = min(30, max(5, len(z) // 4))
        perplejidad = min(n, len(z) - 1)
        return TSNE(n_components=2, perplexity=perplejidad, random_state=semilla).fit_transform(z)


def scatter_clases(
    xy: np.ndarray,
    clases: np.ndarray | pd.Series,
    titulo: str,
    ruta: Path | None = None,
) -> Path:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    clases = np.asarray(clases)
    for clase, color in COLORES_CLASE.items():
        m = clases == clase
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=8,
            alpha=0.55,
            c=color,
            label=f"{clase} (n={int(m.sum())})",
            linewidths=0,
        )
    otras = ~np.isin(clases, list(COLORES_CLASE))
    if otras.any():
        ax.scatter(
            xy[otras, 0],
            xy[otras, 1],
            s=8,
            alpha=0.55,
            c="gray",
            label=f"otra (n={int(otras.sum())})",
            linewidths=0,
        )
    ax.legend(markerscale=2, frameon=False)
    ax.set_title(titulo)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    destino = ruta or (RESULTADOS / f"{_slug(titulo)}.png")
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    return destino


def scatter_clusters(
    xy: np.ndarray,
    clusters: np.ndarray,
    titulo: str,
    ruta: Path | None = None,
) -> Path:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    for k in np.unique(clusters):
        m = clusters == k
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.55, label=f"cluster {k}", linewidths=0)
    ax.legend(markerscale=2, frameon=False)
    ax.set_title(titulo)
    fig.tight_layout()
    destino = ruta or (RESULTADOS / f"{_slug(titulo)}.png")
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    return destino


def _slug(texto: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in texto).strip("_")[:80]


def curva_perdida(train: list[float], val: list[float], titulo: str, ruta: Path) -> Path:
    """Curva de entrenamiento y validación."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train, label="entrenamiento")
    ax.plot(val, label="validación")
    ax.set_xlabel("época")
    ax.set_ylabel("pérdida")
    ax.set_title(titulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def heatmap_tabla(tabla: pd.DataFrame, titulo: str, ruta: Path) -> Path:
    """Matriz de conteos clase contra cluster."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    valores = tabla.to_numpy()
    ax.imshow(valores, aspect="auto", cmap="Blues")
    ax.set_xticks(range(tabla.shape[1]))
    ax.set_xticklabels([str(c) for c in tabla.columns], rotation=45, ha="right")
    ax.set_yticks(range(tabla.shape[0]))
    ax.set_yticklabels([str(i) for i in tabla.index])
    ax.set_xlabel(tabla.columns.name or "cluster")
    ax.set_ylabel(tabla.index.name or "clase")
    ax.set_title(titulo)
    umbral = float(valores.max()) / 2 if valores.size else 0.0
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            color = "white" if valores[i, j] > umbral else "black"
            ax.text(
                j,
                i,
                f"{int(valores[i, j])}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def barra_importancia(tabla: pd.DataFrame, titulo: str, ruta: Path, k: int = 15) -> Path:
    """Top-k de importancia de un Random Forest ajustado solo en train."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    top = tabla.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importancia"], color="steelblue")
    ax.set_xlabel("importancia")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def fase_plegada(mjd: np.ndarray, periodo: float) -> np.ndarray:
    if not np.isfinite(periodo) or periodo <= 0:
        raise ValueError("periodo inválido para plegar la curva")
    return np.mod(mjd, periodo) / periodo


def curva_luz_plegada(
    df_lcs: pd.DataFrame,
    oid: str,
    periodo: float,
    titulo: str,
    ruta: Path | None = None,
) -> Path | None:
    sub = df_lcs.loc[df_lcs["oid"] == oid]
    if sub.empty or not np.isfinite(periodo) or periodo <= 0:
        return None
    mag_col = columna_magnitud(df_lcs)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fid, color, nombre in ((1, "C0", "g"), (2, "C3", "r")):
        b = sub[sub["fid"] == fid]
        if b.empty:
            continue
        fase = fase_plegada(b["mjd"].to_numpy(), float(periodo))
        ax.scatter(fase, b[mag_col], s=10, alpha=0.7, c=color, label=nombre, linewidths=0)
        ax.scatter(fase + 1.0, b[mag_col], s=10, alpha=0.35, c=color, linewidths=0)
    ax.invert_yaxis()
    ax.set_xlabel("fase")
    ax.set_ylabel(mag_col)
    ax.set_title(titulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    destino = ruta or (RESULTADOS / f"{_slug(titulo)}_{oid}.png")
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino
