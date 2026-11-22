"""Rutas, colores y semillas del proyecto."""

from __future__ import annotations

from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"

ARCHIVO_FEATURES = "features.parquet"
ARCHIVO_LCS = "lcs.parquet"
ARCHIVO_OIDS = "oids.parquet"

SEMILLA = 42
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15

# Autoencoder. Z_DIM es la dimensión del latente; la comparten el autoencoder, MCDSVDD (que parte
# de su encoder) y el GMM. Se eligió con VALIDACIÓN entre 4, 6, 8, 10, 12 y 16, con 3 semillas cada
# uno: 10 dio el mejor ARI del GMM (0.27 contra 0.22 con 8) y también el mejor de MCDSVDD.
# EPOCAS_AE es solo un tope: el early stopping con validación decide cuándo parar (paró entre las
# épocas 28 y 64 en la búsqueda).
Z_DIM = 10
EPOCAS_AE = 100

# Componentes del GMM sobre el latente. Se eligió con VALIDACIÓN entre 5, 8, 10, 12 y 15 (con
# Z_DIM = 10 y 3 semillas): el ARI baja al agregar componentes (0.27 con 5, 0.24 con 10).
N_COMPONENTES_GMM = 5

# GMM por clase (gmm_clase.py): cada clase conocida tiene su propio GMM y su número de componentes
# se elige con BIC entre 1 y K_MAX_GMM_CLASE.
K_MAX_GMM_CLASE = 4

# VaDE (vade.py): tope de épocas; el early stopping con validación decide cuándo parar.
EPOCAS_VADE = 300

# Detección de anomalías con MCDSVDD (anomalias.py): un objeto es anómalo si su distancia al
# centro más cercano supera este percentil de las distancias de VALIDACIÓN de ese mismo centro.
PERCENTIL_ANOMALIA = 99

COLORES_CLASE = {
    "CEP": "steelblue",
    "DSCT": "chocolate",
    "EB": "mediumseagreen",
    "LPV": "indianred",
    "RRL": "magenta",
}

# Features que se excluyen además de las 20 descartadas (no-detecciones y WISE): quedan 140.
# Se eligieron solo con train (scripts/comparar_preprocesamiento.py --paso 5): primero las de
# NaN >= 8.5% que están en la mitad menos importante de un random forest, después las de menor
# importancia. En validación, con 5 semillas del autoencoder x 3 del GMM, subieron el ARI del GMM
# de 0.262 a 0.284 y el NMI de 0.307 a 0.339 (el NMI mejoró en las 5 semillas).
FEATURES_EXCLUIDAS = (
    "MHPS_PN_flag_g", "MHPS_PN_flag_r", "Pvar_r", "PairSlopeTrend_r", "AndersonDarling_r",
    "LinearTrend_r", "delta_period_r", "SmallKurtosis_r", "Eta_e_r", "StetsonK_r", "Q31_r",
    "MedianBRP_r", "MedianAbsDev_r", "SPM_chi_r", "Beyond1Std_r", "Std_r", "Con_r", "Mean_r",
)

COLUMNAS_PERIODO = ("Period_band_g", "Period_band_r", "Multiband_period_g_r")
COLUMNAS_COLOR = (
    "g-r_mean_g_r",
    "g-r_mean_corr_g_r",
    "g-r_max_g_r",
    "g-r_max_corr_g_r",
)
# Variante con solo periodo y color g-r (preparar_conjuntos con solo_periodo_color=True).
# El notebook y el script usan las 158 features que no se descartan.
COLUMNAS_MODELO = COLUMNAS_PERIODO + COLUMNAS_COLOR
COLUMNAS_ARCSINH = ("MHPS_ratio_g", "MHPS_ratio_r")
COLUMNAS_MAGNITUD_LC = (
    "magpsf_corr",
    "mag_corr",
    "magpsf",
    "magap",
    "mag",
)


def generador(semilla: int | None = None) -> np.random.Generator:
    """Devuelve un generador NumPy determinista."""
    return np.random.default_rng(SEMILLA if semilla is None else semilla)
