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

COLORES_CLASE = {
    "CEP": "steelblue",
    "DSCT": "chocolate",
    "EB": "mediumseagreen",
    "LPV": "indianred",
    "RRL": "magenta",
}

COLUMNAS_PERIODO = ("Period_band_g", "Period_band_r", "Multiband_period_g_r")
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
