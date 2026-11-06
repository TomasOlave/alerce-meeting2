"""Compara variantes de preprocesamiento en VALIDACIÓN (el test no se usa).

Paso 1 (B1): quitar bloques de features que describen cómo o dónde se observó la estrella.
Paso 2 (C): cambiar cómo se rellenan los datos faltantes (mediana, KNN, iterativa, indicadores).
Paso 3 (B2/B3): quitar redundancia (features casi idénticas), unificar las bandas g y r, o
decorrelacionar todo con PCA y whitening.
Paso 4 (C3): filtrar objetos como Pérez-Carrasco et al. (2023): al menos 6 detecciones en g y en r,
y features en las dos bandas. Se compara entrenar y evaluar con o sin el filtro.
Paso 5: quitar features con mucho NaN y poca importancia, y después las de menor importancia,
hasta quedar con 140, 130 o 120 (NaN e importancia calculados solo con train).
Paso 6: recorte de valores extremos (sin recorte, percentiles 1/2.5/5, Tukey 1.5 y 3 IQR), con
z-score y las 140 features adoptadas.
Paso 7: escala (z-score, robusta, min-max, cuantiles a normal, Yeo-Johnson) con el recorte que se
pase en --recorte.
En todos los pasos la primera variante es la línea base (preprocesamiento actual).

Para cada variante se entrena el autoencoder con 3 semillas y sobre cada latente un GMM con 3
semillas. Se mide:
- ARI y NMI del GMM contra la clase, y silhouette de las clases en el latente;
- NMI del GMM con "le falta una banda" y con los terciles de número de detecciones (diagnóstico:
  cuánto agrupa el GMM por cómo se observó el objeto);
- NMI de los sub-clusters por clase (GMM con K elegido por BIC dentro de cada clase) con "le falta
  una banda", promediado sobre las clases;
- F1 macro de un random forest (con los hiperparámetros elegidos en el notebook);
- ARI de MCDSVDD (solo semilla 42, porque tarda más).

Uso: uv run python scripts/comparar_preprocesamiento.py --paso 2
Guarda las tablas en results/preprocesamiento/.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402, F401
from sklearn.impute import IterativeImputer, KNNImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import PowerTransformer, QuantileTransformer  # noqa: E402

from ic_project.config import (  # noqa: E402
    EPOCAS_AE,
    FEATURES_EXCLUIDAS,
    N_COMPONENTES_GMM,
    RESULTADOS,
    Z_DIM,
)
from ic_project.datos import cargar_features_y_oids, unir_features_etiquetas  # noqa: E402
from ic_project.gmm import MezclaGaussianas  # noqa: E402
from ic_project.modelos import (  # noqa: E402
    asignar_mcdsvdd,
    entrenar_autoencoder,
    entrenar_mcdsvdd,
    latente,
)
from ic_project.preprocesamiento import (  # noqa: E402
    Preprocesador,
    aplicar_transformaciones_elemento,
    columnas_a_descartar,
    especificar_transformaciones,
    features_a_quitar,
    limites_recorte,
    particionar,
    preparar_conjuntos,
)

SEMILLA = 42
SEMILLAS_AE = (42, 1, 2)
SEMILLAS_GMM = (42, 0, 1)
SPLITS = ("train", "val", "test")

# paso 1: familias de features (nombre sin el sufijo de banda _g, _r o _g_r)
BLOQUES = {
    "muestreo": [
        "n_det", "n_pos", "n_neg", "n_non_det_after_fid", "delta_mjd_fid",
        "max_diffmaglim_after_fid", "median_diffmaglim_after_fid", "positive_fraction", "rb",
    ],
    "posicion": ["gal_b", "gal_l", "sgscore1"],
    "magnitud_aparente": ["Mean", "mean_mag", "min_mag", "first_mag"],
    "spm_y_flags": [
        "SPM_A", "SPM_beta", "SPM_chi", "SPM_gamma", "SPM_t0", "SPM_tau_fall", "SPM_tau_rise",
        "MHPS_PN_flag",
    ],
}
# hiperparámetros del random forest elegidos por la búsqueda del notebook
RF = {
    "n_estimators": 200, "max_depth": 16, "min_samples_split": 2, "min_samples_leaf": 2,
    "max_features": "sqrt", "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
}


def familia(columna: str) -> str:
    return re.sub(r"_(g|r|g_r)$", "", columna)


def columnas_de(bloques: list[str], columnas: pd.Index) -> list[str]:
    familias = {f for b in bloques for f in BLOQUES[b]}
    return [c for c in columnas if familia(c) in familias]


def columnas_banda(columnas: pd.Index) -> tuple[list[str], list[str]]:
    cols_g = [c for c in columnas if c.endswith("_g")]
    cols_r = [c for c in columnas if c.endswith("_r") and not c.endswith("_g_r")]
    return cols_g, cols_r


# ------------------------------------------------------------------------------ paso 2: imputación


def preparar_imputacion(
    x: pd.DataFrame, y: pd.Series, oids: pd.Series, imputacion: str, indicadores: bool = False
) -> dict:
    """Mismos pasos que preparar_conjuntos (descarte, partición, log/arcsinh, recorte 1-99%,
    z-score), cambiando solo cómo se rellenan los NaN. Todo se ajusta con train.

    - "mediana": igual que el pipeline actual (Preprocesador).
    - "knn" / "iterativa": se recorta, se estandariza con media y desviación de train (ignorando
      NaN) y se imputa en ese espacio con KNNImputer(10 vecinos) o IterativeImputer.
    - indicadores: agrega falta_g, falta_r (todas las features de esa banda son NaN) y frac_nan.
    """
    x_sel = x.drop(columns=columnas_a_descartar(x.columns), errors="ignore")
    partes = particionar(x_sel, y, oids, semilla=SEMILLA)
    spec = especificar_transformaciones(partes["x_train"])
    frames = {s: aplicar_transformaciones_elemento(partes[f"x_{s}"], spec) for s in SPLITS}
    if indicadores:
        cols_g, cols_r = columnas_banda(x_sel.columns)
        for s, f in frames.items():
            frames[s] = f.assign(
                falta_g=f[cols_g].isna().all(axis=1).astype(float),
                falta_r=f[cols_r].isna().all(axis=1).astype(float),
                frac_nan=f.isna().mean(axis=1),
            )

    if imputacion == "mediana":
        prep = Preprocesador().ajustar(frames["train"])
        x_out = {s: prep.transformar(f).to_numpy(dtype=np.float32) for s, f in frames.items()}
    else:
        inf, sup = frames["train"].quantile(0.01), frames["train"].quantile(0.99)
        recortado = {s: f.clip(lower=inf, upper=sup, axis=1) for s, f in frames.items()}
        media = recortado["train"].mean()
        escala = recortado["train"].std(ddof=0).replace(0, 1.0).fillna(1.0)
        z = {s: (f - media) / escala for s, f in recortado.items()}
        if imputacion == "knn":
            imputador = KNNImputer(n_neighbors=10)
        elif imputacion == "iterativa":
            imputador = IterativeImputer(max_iter=10, random_state=SEMILLA)
        else:
            raise ValueError(imputacion)
        imputador.fit(z["train"])
        x_out = {s: imputador.transform(f).astype(np.float32) for s, f in z.items()}

    salida = {f"x_{s}": x_out[s] for s in SPLITS}
    for s in SPLITS:
        salida[f"y_{s}"] = partes[f"y_{s}"]
        salida[f"oid_{s}"] = partes[f"oid_{s}"]
    return salida


# ------------------------------------------------------------------ paso 3: redundancia y bandas


def redundantes(x_train: pd.DataFrame, umbral: float) -> list[str]:
    """Features a quitar para que ningún par que quede tenga |Spearman| > umbral (solo train).
    Se recorren de menos a más NaN, así que de cada grupo queda la más completa."""
    corr = x_train.corr(method="spearman").abs()
    orden = x_train.isna().mean().sort_values(kind="stable").index
    quedan: list[str] = []
    quitar: list[str] = []
    for c in orden:
        if any(corr.loc[c, k] > umbral for k in quedan):
            quitar.append(c)
        else:
            quedan.append(c)
    return quitar


def unificar_bandas(frame: pd.DataFrame) -> pd.DataFrame:
    """Cada familia con versión g y r pasa a ser su promedio (si falta una banda, queda la otra)."""
    familias: dict[str, list[str]] = {}
    for c in frame.columns:
        familias.setdefault(familia(c), []).append(c)
    nuevas = {}
    for fam, cols in familias.items():
        if len(cols) == 1:
            nuevas[cols[0]] = frame[cols[0]]
        else:
            nuevas[f"{fam}_gr"] = frame[cols].mean(axis=1)
    return pd.DataFrame(nuevas, index=frame.index)


def preparar_paso3(
    x: pd.DataFrame,
    y: pd.Series,
    oids: pd.Series,
    umbral: float | None = None,
    unificar: bool = False,
    pca_varianza: float | None = None,
) -> dict:
    """Mismos pasos que preparar_conjuntos con una de estas variantes, ajustadas con train:
    quitar features redundantes, unificar bandas o PCA con whitening al final."""
    x_sel = x.drop(columns=columnas_a_descartar(x.columns), errors="ignore")
    partes = particionar(x_sel, y, oids, semilla=SEMILLA)
    quitar = redundantes(partes["x_train"], umbral) if umbral is not None else []
    spec = especificar_transformaciones(partes["x_train"])
    frames = {
        s: aplicar_transformaciones_elemento(partes[f"x_{s}"].drop(columns=quitar), spec)
        for s in SPLITS
    }
    if unificar:
        frames = {s: unificar_bandas(f) for s, f in frames.items()}
    prep = Preprocesador().ajustar(frames["train"])
    x_out = {s: prep.transformar(f).to_numpy(dtype=np.float64) for s, f in frames.items()}
    if pca_varianza is not None:
        media = x_out["train"].mean(axis=0)
        _, sv, vt = np.linalg.svd(x_out["train"] - media, full_matrices=False)
        varianza = sv**2 / (len(x_out["train"]) - 1)
        k = int(np.searchsorted(np.cumsum(varianza) / varianza.sum(), pca_varianza) + 1)
        proyeccion = vt[:k].T / np.sqrt(varianza[:k])  # whitening: cada componente con varianza 1
        x_out = {s: (f - media) @ proyeccion for s, f in x_out.items()}

    salida = {f"x_{s}": x_out[s].astype(np.float32) for s in SPLITS}
    for s in SPLITS:
        salida[f"y_{s}"] = partes[f"y_{s}"]
        salida[f"oid_{s}"] = partes[f"oid_{s}"]
    return salida


# ------------------------------------------------------------------------ paso 4: filtrar objetos


def filtro_paper(crudo: pd.DataFrame) -> np.ndarray:
    """Objetos que cumplen los criterios de Pérez-Carrasco et al. (2023, sección 2.2): al menos 6
    detecciones en g y en r, y features calculadas en las dos bandas."""
    cols_g, cols_r = columnas_banda(crudo.columns.drop(columnas_a_descartar(crudo.columns)))
    ambas_bandas = ~(crudo[cols_g].isna().all(axis=1) | crudo[cols_r].isna().all(axis=1))
    detecciones = (crudo["n_det_g"].fillna(0) >= 6) & (crudo["n_det_r"].fillna(0) >= 6)
    return (ambas_bandas & detecciones).to_numpy()


def preparar_filtrado(x: pd.DataFrame, y: pd.Series, oids: pd.Series) -> dict:
    """Misma partición que la base; se sacan de cada conjunto los objetos que no pasan el filtro y
    el preprocesamiento se ajusta solo con el train filtrado."""
    x_sel = x.drop(columns=columnas_a_descartar(x.columns), errors="ignore")
    partes = particionar(x_sel, y, oids, semilla=SEMILLA)
    pasan = {s: filtro_paper(partes[f"x_{s}"]) for s in SPLITS}
    spec = especificar_transformaciones(partes["x_train"][pasan["train"]])
    frames = {
        s: aplicar_transformaciones_elemento(partes[f"x_{s}"][pasan[s]], spec) for s in SPLITS
    }
    prep = Preprocesador().ajustar(frames["train"])
    salida = {f"x_{s}": prep.transformar(f).to_numpy(dtype=np.float32) for s, f in frames.items()}
    for s in SPLITS:
        salida[f"y_{s}"] = partes[f"y_{s}"][pasan[s]].reset_index(drop=True)
        salida[f"oid_{s}"] = partes[f"oid_{s}"][pasan[s]].reset_index(drop=True)
    return salida


# --------------------------------------------------------- pasos 6 y 7: recorte y escala


def preparar_escalado(
    x: pd.DataFrame, y: pd.Series, oids: pd.Series, recorte: tuple, escala: str
) -> dict:
    """Mismos pasos que preparar_conjuntos (140 features, log/arcsinh, imputación con la mediana)
    cambiando el recorte y la escala. Todo se ajusta con train."""
    quitar = columnas_a_descartar(x.columns) + list(FEATURES_EXCLUIDAS)
    partes = particionar(x.drop(columns=quitar, errors="ignore"), y, oids, semilla=SEMILLA)
    spec = especificar_transformaciones(partes["x_train"])
    frames = {s: aplicar_transformaciones_elemento(partes[f"x_{s}"], spec) for s in SPLITS}
    inf, sup = limites_recorte(frames["train"], recorte)
    recortado = {s: f.clip(lower=inf, upper=sup, axis=1) for s, f in frames.items()}
    mediana = recortado["train"].median()
    imputado = {s: f.fillna(mediana) for s, f in recortado.items()}
    train = imputado["train"]

    if escala == "zscore":
        centro, ancho = train.mean(), train.std(ddof=0)
    elif escala == "robusta":
        centro = train.median()
        iqr = train.quantile(0.75) - train.quantile(0.25)
        ancho = iqr.where(iqr > 0, train.std(ddof=0))
    elif escala == "robusta_5_95":
        # rango entre percentiles 5 y 95: no se anula en features casi constantes en el centro
        centro = train.median()
        rango = train.quantile(0.95) - train.quantile(0.05)
        ancho = rango.where(rango > 0, train.std(ddof=0))
    elif escala == "minmax":
        centro, ancho = train.min(), train.max() - train.min()
    if escala in ("zscore", "robusta", "robusta_5_95", "minmax"):
        ancho = ancho.replace(0, 1.0)
        x_out = {s: ((f - centro) / ancho).to_numpy() for s, f in imputado.items()}
    else:
        if escala == "cuantil_normal":
            modelo = QuantileTransformer(
                n_quantiles=1000, output_distribution="normal", random_state=SEMILLA
            )
        elif escala == "yeo_johnson":
            modelo = PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            raise ValueError(escala)
        modelo.fit(train.to_numpy())
        x_out = {s: modelo.transform(f.to_numpy()) for s, f in imputado.items()}

    salida = {f"x_{s}": x_out[s].astype(np.float32) for s in SPLITS}
    for s in SPLITS:
        salida[f"y_{s}"] = partes[f"y_{s}"]
        salida[f"oid_{s}"] = partes[f"oid_{s}"]
    return salida


def fraccion_recortada(x: pd.DataFrame, y: pd.Series, oids: pd.Series, recorte: tuple) -> float:
    """% de los valores no nulos de train que caen fuera de los límites."""
    quitar = columnas_a_descartar(x.columns) + list(FEATURES_EXCLUIDAS)
    partes = particionar(x.drop(columns=quitar, errors="ignore"), y, oids, semilla=SEMILLA)
    train = partes["x_train"]
    train = aplicar_transformaciones_elemento(train, especificar_transformaciones(train))
    inf, sup = limites_recorte(train, recorte)
    fuera = (train.lt(inf, axis=1) | train.gt(sup, axis=1)).to_numpy().sum()
    return 100 * fuera / train.notna().to_numpy().sum()


# ------------------------------------------------------------------------------------ evaluación


def nmi_subclusters(z_tr, y_tr, z_va, y_va, sin_banda_va) -> float:
    """Dentro de cada clase: GMM con K (1 a 6) elegido por BIC en train, y NMI de sus sub-clusters
    de validación con "le falta una banda". Promedio sobre las clases."""
    valores = []
    for clase in sorted(set(y_tr)):
        zc = z_tr[y_tr == clase]
        modelos = [MezclaGaussianas(k, n_init=3, random_state=SEMILLA).fit(zc) for k in range(1, 7)]
        mejor = min(modelos, key=lambda m: m.bic(zc))
        en_val = y_va == clase
        sub = mejor.predict(z_va[en_val])
        valores.append(normalized_mutual_info_score(sin_banda_va[en_val], sub))
    return float(np.mean(valores))


def evaluar(
    nombre: str,
    c: dict,
    sin_banda_val: np.ndarray,
    terciles_det_val: np.ndarray,
    mascara: np.ndarray | None = None,
    semillas_ae: tuple[int, ...] = SEMILLAS_AE,
) -> list:
    """Entrena con c["x_train"] (y early stopping con todo c["x_val"]) y mide en las filas de
    validación indicadas por `mascara` (todas si es None). Los diagnósticos van alineados con
    c["x_val"]."""
    m = np.ones(len(c["y_val"]), dtype=bool) if mascara is None else mascara
    y_tr, y_val_todo = c["y_train"].to_numpy(), c["y_val"].to_numpy()
    y_val = y_val_todo[m]
    sin_banda_val, terciles_det_val = sin_banda_val[m], terciles_det_val[m]
    clases = sorted(set(y_tr))
    mapa = {k: i for i, k in enumerate(clases)}
    y_tr_i = np.array([mapa[k] for k in y_tr])
    y_va_i = np.array([mapa[k] for k in y_val_todo])

    rf = RandomForestClassifier(**RF).fit(c["x_train"], y_tr)
    f1_rf = f1_score(y_val, rf.predict(c["x_val"][m]), average="macro")

    filas = []
    for s_ae in semillas_ae:
        ae, hist = entrenar_autoencoder(
            c["x_train"], c["x_val"], z_dim=Z_DIM, n_epocas=EPOCAS_AE, semilla=s_ae
        )
        z_tr, z_va = latente(ae, c["x_train"]), latente(ae, c["x_val"])[m]
        nmi_sub = nmi_subclusters(z_tr, y_tr, z_va, y_val, sin_banda_val)
        ari_mcd = np.nan
        if s_ae == SEMILLA:
            mcd, _ = entrenar_mcdsvdd(
                c["x_train"], y_tr_i, c["x_val"], y_va_i, autoencoder=ae, z_dim=Z_DIM, semilla=s_ae
            )
            ari_mcd = adjusted_rand_score(y_va_i[m], asignar_mcdsvdd(mcd, c["x_val"])[m])
        for s_gmm in SEMILLAS_GMM:
            pred = (
                MezclaGaussianas(N_COMPONENTES_GMM, n_init=5, random_state=s_gmm)
                .fit(z_tr)
                .predict(z_va)
            )
            filas.append({
                "variante": nombre,
                "n_features": c["x_train"].shape[1],
                "n_train": len(y_tr),
                "n_val_medido": len(y_val),
                "semilla_ae": s_ae,
                "semilla_gmm": s_gmm,
                "ari_gmm": adjusted_rand_score(y_val, pred),
                "nmi_gmm": normalized_mutual_info_score(y_val, pred),
                "silhouette_clases": silhouette_score(z_va, y_val),
                "nmi_con_sin_banda": normalized_mutual_info_score(sin_banda_val, pred),
                "nmi_con_detecciones": normalized_mutual_info_score(terciles_det_val, pred),
                "nmi_subclusters_sin_banda": nmi_sub if s_gmm == SEMILLAS_GMM[0] else np.nan,
                "mse_val": min(hist.val),
                "f1_rf": f1_rf,
                "ari_mcdsvdd": ari_mcd if s_gmm == SEMILLAS_GMM[0] else np.nan,
            })
    return filas


def resumir(filas: list) -> pd.DataFrame:
    t = pd.DataFrame(filas)
    metricas = [
        "ari_gmm", "nmi_gmm", "silhouette_clases", "nmi_con_sin_banda", "nmi_con_detecciones",
        "nmi_subclusters_sin_banda", "mse_val", "f1_rf", "ari_mcdsvdd",
    ]
    g = t.groupby("variante", sort=False)
    resumen = g[metricas].mean()
    resumen.insert(0, "n_features", g["n_features"].first())
    resumen.insert(1, "n_train", g["n_train"].first())
    resumen.insert(2, "n_val_medido", g["n_val_medido"].first())
    resumen.insert(4, "ari_gmm_desv", g["ari_gmm"].std())
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paso", type=int, choices=(1, 2, 3, 4, 5, 6, 7), required=True)
    parser.add_argument(
        "--recorte", nargs="+", default=["percentil", "1"],
        help="paso 7: ninguno | percentil P | tukey K",
    )
    parser.add_argument("--escalas", nargs="*", default=None, help="paso 7: subconjunto de escalas")
    parser.add_argument(
        "--objetivos", type=int, nargs="*", default=[140, 130, 120], help="paso 5: features a dejar"
    )
    parser.add_argument("--semillas-ae", type=int, nargs="*", default=list(SEMILLAS_AE))
    args = parser.parse_args()
    semillas_ae = tuple(args.semillas_ae)

    df_features, df_oids = cargar_features_y_oids()
    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    usadas = x.columns.drop(columnas_a_descartar(x.columns))

    # diagnósticos de validación, calculados sobre las features crudas
    base, _, _ = preparar_conjuntos(x, y, oids, semilla=SEMILLA, solo_periodo_color=False)
    crudo_val = x.set_index(oids).loc[base["oid_val"]]
    cols_g, cols_r = columnas_banda(usadas)
    sin_banda_val = (
        crudo_val[cols_g].isna().all(axis=1) | crudo_val[cols_r].isna().all(axis=1)
    ).to_numpy()
    n_det_val = crudo_val[[c for c in usadas if familia(c) == "n_det"]].fillna(0).sum(axis=1)
    terciles_det_val = pd.qcut(n_det_val, 3, labels=False).to_numpy()
    print(f"validación: {len(sin_banda_val)} objetos, {sin_banda_val.mean():.1%} sin una banda\n")

    columnas_extra: dict[str, dict] = {}  # columnas que se agregan a la tabla resumen
    if args.paso == 1:
        variantes = {
            "base (158 features)": lambda: base,
            **{
                f"sin {b}": (lambda b=b: preparar_conjuntos(
                    x.drop(columns=columnas_de([b], usadas)), y, oids, semilla=SEMILLA,
                    solo_periodo_color=False)[0])
                for b in BLOQUES
            },
            "sin todos los bloques": lambda: preparar_conjuntos(
                x.drop(columns=columnas_de(list(BLOQUES), usadas)), y, oids, semilla=SEMILLA,
                solo_periodo_color=False)[0],
        }
        archivo = "paso1_features_observacion"
    elif args.paso == 2:
        variantes = {
            "base (mediana)": lambda: base,
            "KNN (10 vecinos)": lambda: preparar_imputacion(x, y, oids, "knn"),
            "iterativa": lambda: preparar_imputacion(x, y, oids, "iterativa"),
            "mediana + indicadores": lambda: preparar_imputacion(x, y, oids, "mediana", True),
        }
        archivo = "paso2_imputacion"
    elif args.paso == 3:
        variantes = {
            "base (158 features)": lambda: base,
            "sin redundancia |rho|>0.9": lambda: preparar_paso3(x, y, oids, umbral=0.9),
            "sin redundancia |rho|>0.8": lambda: preparar_paso3(x, y, oids, umbral=0.8),
            "bandas unificadas": lambda: preparar_paso3(x, y, oids, unificar=True),
            "PCA 95% + whitening": lambda: preparar_paso3(x, y, oids, pca_varianza=0.95),
        }
        archivo = "paso3_redundancia"
    elif args.paso == 5:
        # % NaN en train (features crudas) e importancia de un random forest ajustado en train
        x_usadas = x.drop(columns=columnas_a_descartar(x.columns))
        pct_nan = particionar(x_usadas, y, oids, semilla=SEMILLA)["x_train"].isna().mean() * 100
        base_158, _, _ = preparar_conjuntos(
            x, y, oids, semilla=SEMILLA, solo_periodo_color=False, excluir=()
        )
        rf = RandomForestClassifier(**RF).fit(base_158["x_train"], base_158["y_train"])
        importancia = pd.Series(rf.feature_importances_, index=pct_nan.index)
        quitar = {n: features_a_quitar(pct_nan, importancia, n) for n in args.objetivos}
        for n, cols in quitar.items():
            print(f"para quedar con {n}: se quitan {len(cols)} -> {cols}\n")
        variantes = {"base (158 features)": lambda: base_158}
        for n, cols in quitar.items():
            variantes[f"{n} features"] = (
                lambda cols=cols: preparar_conjuntos(
                    x.drop(columns=cols), y, oids, semilla=SEMILLA, solo_periodo_color=False,
                    excluir=(),
                )[0]
            )
        archivo = (
            f"paso5_nan_importancia_{'-'.join(map(str, args.objetivos))}"
            f"_{len(semillas_ae)}semillas"
        )
    elif args.paso == 6:
        recortes = {
            "sin recorte": ("ninguno",),
            "percentil 1-99 (actual)": ("percentil", 1),
            "percentil 2.5-97.5": ("percentil", 2.5),
            "percentil 5-95": ("percentil", 5),
            "Tukey 3 IQR": ("tukey", 3),
            "Tukey 1.5 IQR": ("tukey", 1.5),
        }
        pct = {nombre: fraccion_recortada(x, y, oids, r) for nombre, r in recortes.items()}
        columnas_extra["pct_recortado_train"] = pct
        for nombre, valor in pct.items():
            print(f"{nombre:24s}: {valor:5.2f}% de los valores de train fuera de los limites")
        print()
        variantes = {
            nombre: (lambda r=r: preparar_escalado(x, y, oids, r, "zscore"))
            for nombre, r in recortes.items()
        }
        archivo = "paso6_recorte"
    elif args.paso == 7:
        recorte = (args.recorte[0], *map(float, args.recorte[1:]))
        print(f"recorte usado: {recorte}\n")
        escalas = {
            "z-score (actual)": "zscore",
            "robusta (mediana / IQR)": "robusta",
            "robusta (mediana / P5-P95)": "robusta_5_95",
            "min-max [0, 1]": "minmax",
            "cuantiles -> normal": "cuantil_normal",
            "Yeo-Johnson": "yeo_johnson",
        }
        if args.escalas:
            escalas = {n: e for n, e in escalas.items() if e in args.escalas}
        variantes = {
            nombre: (lambda e=e: preparar_escalado(x, y, oids, recorte, e))
            for nombre, e in escalas.items()
        }
        archivo = f"paso7_escala_{'_'.join(args.recorte)}_{'-'.join(escalas.values())}"
    else:
        # cada variante devuelve (conjuntos, filas de la validación base que contiene, máscara
        # de las filas donde se mide)
        pasa_val = filtro_paper(crudo_val)
        crudo_train = x.set_index(oids).loc[base["oid_train"]]
        pasa_train = filtro_paper(crudo_train)
        resumen_filtro = pd.DataFrame({
            "train": base["y_train"].value_counts(),
            "train filtrado": base["y_train"][pasa_train].value_counts(),
            "val": base["y_val"].value_counts(),
            "val filtrado": base["y_val"][pasa_val].value_counts(),
        })
        resumen_filtro["% de train que queda"] = (
            100 * resumen_filtro["train filtrado"] / resumen_filtro["train"]
        ).round(1)
        print("objetos por clase antes y después del filtro")
        print(resumen_filtro.to_string(), "\n")
        variantes = {
            "base, val completa": lambda: base,
            "base, val filtrada": lambda: (base, None, pasa_val),
            "filtrado (train y val)": lambda: (preparar_filtrado(x, y, oids), pasa_val, None),
        }
        archivo = "paso4_filtrado"

    filas = []
    for nombre, preparar in variantes.items():
        t0 = time.perf_counter()
        resultado = preparar()
        c, filas_val, mascara = (
            resultado if isinstance(resultado, tuple) else (resultado, None, None)
        )
        sel = np.ones(len(base["oid_val"]), dtype=bool) if filas_val is None else filas_val
        assert (c["oid_val"].to_numpy() == base["oid_val"].to_numpy()[sel]).all()
        nuevas = evaluar(
            nombre, c, sin_banda_val[sel], terciles_det_val[sel], mascara, semillas_ae
        )
        filas.extend(nuevas)
        sub = pd.DataFrame(nuevas)
        print(
            f"{nombre:24s} | {c['x_train'].shape[1]} features | ARI GMM {sub.ari_gmm.mean():.3f} ± "
            f"{sub.ari_gmm.std():.3f} | F1 RF {sub.f1_rf.iloc[0]:.3f} | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    salida = RESULTADOS / "preprocesamiento"
    salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(salida / f"{archivo}_detalle.csv", index=False)
    resumen = resumir(filas)
    for columna, valores in columnas_extra.items():
        resumen.insert(0, columna, pd.Series(valores))
    resumen.to_csv(salida / f"{archivo}.csv")
    print(
        f"\nVALIDACIÓN: media sobre {len(semillas_ae)} semillas de autoencoder x "
        f"{len(SEMILLAS_GMM)} de GMM (MCDSVDD: semilla 42)"
    )
    print(resumen.round(3).to_string())
    print(f"\nTablas en {salida}")


if __name__ == "__main__":
    main()
