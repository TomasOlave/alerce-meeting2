"""Protocolo de evaluación de MCDSVDD de Pérez-Carrasco et al. (2023), sección 4.

Una clase hace de anomalía: se saca del entrenamiento y el modelo tiene que reconocerla como rara
usando solo lo que aprendió de las otras. Pasos:

1. 80% de los datos para entrenar y 20% de test (TS1), estratificado por clase.
2. El 80% se divide en 5 folds estratificados. En cada fold, 4/5 entrenan y 1/5 valida.
3. Para cada clase anómala y cada fold, los objetos de esa clase se sacan de train y de
   validación (la validación tiene solo inliers, como en el paper). El preprocesamiento y el
   autoencoder se ajustan con esos inliers, y MCDSVDD parte del encoder del autoencoder.
4. TS2 = los inliers de TS1 más objetos de la clase anómala sacados de TS1 y de train (el modelo
   nunca los vio), en la cantidad justa para que sean el 10% de TS2.
5. Se mide el AUROC del puntaje de anomalía (distancia al centro más cercano) para separar la
   clase anómala del resto, y se reporta media y desviación sobre los folds. Como en la Tabla 1
   del paper, también se mide el autoencoder usando su error de reconstrucción como puntaje.
   Opcionalmente se miden con el mismo protocolo el GMM (puntaje = -log-verosimilitud sobre el
   latente del autoencoder), DAGMM (puntaje = energía), el GMM por clase (gmm_clase.py, sobre el
   latente del autoencoder y sobre el de MCDSVDD) y VaDE (vade.py, sin clases y con clases;
   puntaje = -ELBO). Además del AUROC se guarda el AUPRC (área bajo la curva precisión-recall,
   que el código del paper también calcula) de cada detector.

Diferencias con el paper: aquí solo hay estrellas periódicas (5 clases, sin Periodic-Other), las
features se escalan con z-score ajustado en train (no a [0, 1]) y el latente tiene Z_DIM
dimensiones, elegido en el meeting 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ic_project.config import (
    EPOCAS_AE,
    EPOCAS_VADE,
    K_MAX_GMM_CLASE,
    N_COMPONENTES_GMM,
    SEMILLA,
    Z_DIM,
)
from ic_project.dagmm import entrenar_dagmm, salidas_dagmm
from ic_project.gmm import MezclaGaussianas
from ic_project.gmm_clase import GMMPorClase
from ic_project.modelos import (
    Autoencoder,
    _tensor_x,
    dispositivo,
    entrenar_autoencoder,
    entrenar_mcdsvdd,
    latente,
    puntaje_mcdsvdd,
)
from ic_project.preprocesamiento import (
    Preprocesador,
    aplicar_transformaciones_elemento,
    columnas_a_descartar,
    especificar_transformaciones,
)
from ic_project.vade import entrenar_vade, salidas_vade

# Tabla 1 del paper, clases periódicas (su "E" es nuestra EB): AUROC (media, desviación).
AUROC_PAPER_MCDSVDD = {
    "CEP": (0.858, 0.025),
    "DSCT": (0.819, 0.015),
    "EB": (0.945, 0.006),
    "LPV": (0.953, 0.008),
    "RRL": (0.953, 0.003),
}
AUROC_PAPER_AE = {
    "CEP": (0.564, 0.024),
    "DSCT": (0.367, 0.015),
    "EB": (0.864, 0.009),
    "LPV": (0.996, 0.000),
    "RRL": (0.907, 0.015),
}


def filtro_detecciones(x_crudo: pd.DataFrame) -> np.ndarray:
    """Criterio del código del paper (preprocessing/main.py): al menos 6 detecciones en g y en r
    (n_det_1 >= 6 y n_det_2 >= 6; aquí n_det_g y n_det_r)."""
    return ((x_crudo["n_det_g"].fillna(0) >= 6) & (x_crudo["n_det_r"].fillna(0) >= 6)).to_numpy()


def filtro_flags(oids: pd.Series, lcs: pd.DataFrame) -> np.ndarray:
    """Criterio del código del paper para fuentes periódicas (flag_data): magnitudes corregidas en
    g y en r y ninguna detección dudosa (flag_ndubious < 1) en cada banda.

    Aquí se reconstruye desde las detecciones: por objeto y banda, todas con `corrected` y ninguna
    con `dubious`. `lcs` necesita las columnas oid, fid (1 = g, 2 = r), corrected y dubious.
    """
    por_banda = lcs.groupby(["oid", "fid"]).agg(
        todas_corregidas=("corrected", "all"), dudosas=("dubious", "sum")
    )
    ok = (por_banda["todas_corregidas"] & (por_banda["dudosas"] == 0)).unstack("fid")
    ok = ok.reindex(index=pd.Index(oids), columns=[1, 2]).fillna(False).astype(bool)
    return (ok[1] & ok[2]).to_numpy()


@dataclass
class Particion:
    """Índices (posiciones) del 80% de entrenamiento, del 20% de test y de cada fold."""

    train: np.ndarray
    ts1: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]


def particionar_protocolo(
    y: np.ndarray, n_folds: int = 5, test: float = 0.2, semilla: int = SEMILLA
) -> Particion:
    """80/20 estratificado y n_folds estratificados dentro del 80% (train, validación)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    idx_train, idx_ts1 = train_test_split(idx, test_size=test, random_state=semilla, stratify=y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    folds = [(idx_train[a], idx_train[b]) for a, b in skf.split(idx_train, y[idx_train])]
    return Particion(train=idx_train, ts1=idx_ts1, folds=folds)


def construir_ts2(
    y: np.ndarray,
    particion: Particion,
    clase_anomala: str,
    proporcion: float = 0.1,
    semilla: int = SEMILLA,
) -> tuple[np.ndarray, np.ndarray]:
    """Índices de TS2 y su etiqueta binaria (1 = clase anómala). Las anomalías son el 10%."""
    y = np.asarray(y)
    rng = np.random.default_rng(semilla)
    inliers = particion.ts1[y[particion.ts1] != clase_anomala]
    candidatos = np.concatenate([particion.ts1, particion.train])
    candidatos = candidatos[y[candidatos] == clase_anomala]
    n_anomalias = int(round(proporcion / (1 - proporcion) * len(inliers)))
    if n_anomalias > len(candidatos):
        # hay pocas anomalías: se usan todas y se recortan inliers para mantener la proporción
        n_anomalias = len(candidatos)
        n_inliers = int(round(n_anomalias * (1 - proporcion) / proporcion))
        inliers = rng.choice(inliers, size=n_inliers, replace=False)
    anomalias = rng.choice(candidatos, size=n_anomalias, replace=False)
    idx = np.concatenate([inliers, anomalias])
    etiqueta = np.concatenate([np.zeros(len(inliers), dtype=int), np.ones(n_anomalias, dtype=int)])
    return idx, etiqueta


def preparar_fold(
    x_crudo: pd.DataFrame,
    idx_train: np.ndarray,
    otros: dict[str, np.ndarray],
    excluir: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Descarta columnas (las 20 de siempre y las de `excluir`), transforma y escala. Todo se
    ajusta solo con `idx_train`. Por defecto entran las 158 features, como en el paper."""
    quitar = columnas_a_descartar(x_crudo.columns) + list(excluir)
    x = x_crudo.drop(columns=quitar, errors="ignore")
    train = x.iloc[idx_train]
    spec = especificar_transformaciones(train)
    train = aplicar_transformaciones_elemento(train, spec)
    prep = Preprocesador().ajustar(train)
    salida = {"train": prep.transformar(train).to_numpy(dtype=np.float32)}
    for nombre, idx in otros.items():
        frame = aplicar_transformaciones_elemento(x.iloc[idx], spec)
        salida[nombre] = prep.transformar(frame).to_numpy(dtype=np.float32)
    return salida


@torch.no_grad()
def error_reconstruccion(modelo: Autoencoder, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    """MSE de reconstrucción de cada objeto: el puntaje de anomalía del autoencoder."""
    device = dispositivo()
    modelo.eval()
    modelo.to(device)
    xt = _tensor_x(x)
    errores = []
    for i in range(0, len(xt), batch):
        xb = xt[i : i + batch].to(device)
        _, rec = modelo(xb)
        errores.append(((rec - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errores)


def evaluar_clase_fold(
    x_crudo: pd.DataFrame,
    y: np.ndarray,
    particion: Particion,
    clase_anomala: str,
    fold: int,
    z_dim: int = Z_DIM,
    epocas_ae: int = EPOCAS_AE,
    semilla: int = SEMILLA,
    excluir: tuple[str, ...] = (),
    con_gmm: bool = False,
    con_dagmm: bool = False,
    epocas_dagmm: int = 200,
    con_gmm_clase: bool = False,
    con_vade: bool = False,
    epocas_vade: int = EPOCAS_VADE,
    **kw_mcdsvdd,
) -> dict:
    """Entrena sin `clase_anomala` en el fold `fold` y mide el AUROC y el AUPRC en TS2.

    Siempre mide MCDSVDD y el autoencoder. Con `con_gmm`, también el GMM sobre el latente del
    mismo autoencoder (N_COMPONENTES_GMM componentes, no usa etiquetas). Con `con_dagmm`, también
    DAGMM entrenado desde cero con los mismos datos (como en el notebook).

    Con `con_gmm_clase`, el GMM por clase (un GMM por clase conocida, K por BIC hasta
    K_MAX_GMM_CLASE) sobre el latente del autoencoder ("gmm_clase_ae") y sobre el de MCDSVDD
    ("gmm_clase_mcd"). Puntaje: -max_c log p(z | c).
    Con `con_vade`, VaDE desde el mismo autoencoder: sin clases, con N_COMPONENTES_GMM
    componentes ("vade"), y con un componente por clase conocida ("vade_clase"). Puntaje: -ELBO.
    """
    y = np.asarray(y)
    idx_tr, idx_va = particion.folds[fold]
    idx_tr = idx_tr[y[idx_tr] != clase_anomala]
    idx_va = idx_va[y[idx_va] != clase_anomala]
    idx_ts2, etiqueta = construir_ts2(y, particion, clase_anomala, semilla=semilla)
    datos = preparar_fold(x_crudo, idx_tr, {"val": idx_va, "ts2": idx_ts2}, excluir)

    inliers = sorted(set(y[idx_tr]))
    mapa = {c: i for i, c in enumerate(inliers)}
    y_tr = np.array([mapa[c] for c in y[idx_tr]])
    y_va = np.array([mapa[c] for c in y[idx_va]])

    t0 = time.perf_counter()
    ae, hist_ae = entrenar_autoencoder(
        datos["train"], datos["val"], z_dim=z_dim, n_epocas=epocas_ae, semilla=semilla
    )
    mcd, hist_mcd = entrenar_mcdsvdd(
        datos["train"], y_tr, datos["val"], y_va, autoencoder=ae, z_dim=z_dim, semilla=semilla,
        **kw_mcdsvdd,
    )
    fila: dict = {"clase_anomala": clase_anomala, "fold": fold}
    # puntaje de anomalía de cada detector sobre TS2 (más alto = más anómalo)
    puntajes = {
        "mcdsvdd": puntaje_mcdsvdd(mcd, datos["ts2"]),
        "ae": error_reconstruccion(ae, datos["ts2"]),
    }
    if con_gmm or con_gmm_clase:
        z_ae_tr, z_ae_ts2 = latente(ae, datos["train"]), latente(ae, datos["ts2"])
    if con_gmm:
        gmm = MezclaGaussianas(N_COMPONENTES_GMM, n_init=5, random_state=semilla)
        gmm.fit(z_ae_tr)
        puntajes["gmm"] = -gmm.score_samples(z_ae_ts2)
    if con_dagmm:
        dagmm, hist_dagmm = entrenar_dagmm(
            datos["train"], datos["val"], n_componentes=N_COMPONENTES_GMM, z_dim=z_dim,
            n_epocas=epocas_dagmm, semilla=semilla,
        )
        puntajes["dagmm"] = salidas_dagmm(dagmm, datos["ts2"])["energia"]
        fila["epocas_dagmm"] = len(hist_dagmm.val)
    if con_gmm_clase:
        por_clase_ae = GMMPorClase(K_MAX_GMM_CLASE, random_state=semilla).fit(z_ae_tr, y_tr)
        puntajes["gmm_clase_ae"] = por_clase_ae.puntaje_anomalia(z_ae_ts2)
        por_clase_mcd = GMMPorClase(K_MAX_GMM_CLASE, random_state=semilla)
        por_clase_mcd.fit(latente(mcd, datos["train"]), y_tr)
        puntajes["gmm_clase_mcd"] = por_clase_mcd.puntaje_anomalia(latente(mcd, datos["ts2"]))
        # componentes elegidos por clase, como "CEP:2 DSCT:1 ..." (índices -> nombres)
        for nombre, modelo in (("ae", por_clase_ae), ("mcd", por_clase_mcd)):
            fila[f"k_gmm_clase_{nombre}"] = " ".join(
                f"{inliers[c]}:{k}" for c, k in sorted(modelo.k_.items())
            )
    if con_vade:
        vade, hist_vade = entrenar_vade(
            datos["train"], datos["val"], ae, n_componentes=N_COMPONENTES_GMM, z_dim=z_dim,
            n_epocas=epocas_vade, semilla=semilla,
        )
        puntajes["vade"] = salidas_vade(vade, datos["ts2"])["menos_elbo"]
        vade_clase, hist_vade_clase = entrenar_vade(
            datos["train"], datos["val"], ae, y_train=y_tr, y_val=y_va, z_dim=z_dim,
            n_epocas=epocas_vade, semilla=semilla,
        )
        puntajes["vade_clase"] = salidas_vade(vade_clase, datos["ts2"])["menos_elbo"]
        fila["epocas_vade"] = len(hist_vade.val)
        fila["epocas_vade_clase"] = len(hist_vade_clase.val)
    for detector, puntaje in puntajes.items():
        fila[f"auroc_{detector}"] = float(roc_auc_score(etiqueta, puntaje))
    for detector, puntaje in puntajes.items():
        fila[f"auprc_{detector}"] = float(average_precision_score(etiqueta, puntaje))
    fila.update({
        "epocas_ae": len(hist_ae.val),
        "epocas_mcdsvdd": len(hist_mcd.val),
        "n_ts2": len(idx_ts2),
        "n_anomalias": int(etiqueta.sum()),
        "segundos": time.perf_counter() - t0,
    })
    return fila


def correr_protocolo(
    x_crudo: pd.DataFrame,
    y: np.ndarray,
    clases: list[str] | None = None,
    n_folds: int = 5,
    z_dim: int = Z_DIM,
    epocas_ae: int = EPOCAS_AE,
    semilla: int = SEMILLA,
    verbose: bool = True,
    excluir: tuple[str, ...] = (),
    con_gmm: bool = False,
    con_dagmm: bool = False,
    epocas_dagmm: int = 200,
    con_gmm_clase: bool = False,
    con_vade: bool = False,
    epocas_vade: int = EPOCAS_VADE,
    **kw_mcdsvdd,
) -> pd.DataFrame:
    """Una fila por (clase anómala, fold). `x_crudo` son las features sin preprocesar."""
    y = np.asarray(y)
    x_crudo = x_crudo.reset_index(drop=True)
    particion = particionar_protocolo(y, n_folds=n_folds, semilla=semilla)
    clases = sorted(set(y)) if clases is None else clases
    filas = []
    for clase in clases:
        for fold in range(n_folds):
            fila = evaluar_clase_fold(
                x_crudo, y, particion, clase, fold, z_dim, epocas_ae, semilla, excluir,
                con_gmm, con_dagmm, epocas_dagmm, con_gmm_clase, con_vade, epocas_vade,
                **kw_mcdsvdd,
            )
            filas.append(fila)
            if verbose:
                aurocs = " | ".join(
                    f"{k.removeprefix('auroc_').upper()} {v:.3f}"
                    for k, v in fila.items() if k.startswith("auroc_")
                )
                segundos = fila["segundos"]
                print(f"{clase:5s} fold {fold}: AUROC {aurocs} | {segundos:.0f}s", flush=True)
    return pd.DataFrame(filas)


def resumir_protocolo(resultados: pd.DataFrame, metrica: str = "auroc") -> pd.DataFrame:
    """Media y desviación por clase anómala de `metrica` ("auroc" o "auprc") para cada detector
    medido. Al lado va la Tabla 1 del paper (AUROC, solo MCDSVDD y el autoencoder).

    El AUPRC de un detector al azar es la fracción de anomalías en TS2 (0.1), no 0.5."""
    if metrica not in ("auroc", "auprc"):
        raise ValueError("metrica debe ser 'auroc' o 'auprc'")
    g = resultados.groupby("clase_anomala")
    tabla = pd.DataFrame(index=sorted(resultados["clase_anomala"].unique()))
    for col in [c for c in resultados.columns if c.startswith(f"{metrica}_")]:
        detector = col.removeprefix(f"{metrica}_")
        tabla[detector] = g[col].mean()
        tabla[f"{detector}_desv"] = g[col].std()
    if metrica == "auroc":
        paper_mcd = [AUROC_PAPER_MCDSVDD.get(c, (np.nan, np.nan))[0] for c in tabla.index]
        tabla["paper_mcdsvdd"] = paper_mcd
        tabla["paper_ae"] = [AUROC_PAPER_AE.get(c, (np.nan, np.nan))[0] for c in tabla.index]
    tabla.index.name = "clase anómala"
    return tabla
