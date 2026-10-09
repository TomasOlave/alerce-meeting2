"""Carga de tablas ALeRCE y datos sintéticos para pruebas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ic_project.config import (
    ARCHIVO_FEATURES,
    ARCHIVO_LCS,
    ARCHIVO_OIDS,
    DATOS_CRUDOS,
    generador,
)


def _primer_existente(nombres: list[str], raiz: Path) -> Path | None:
    for nombre in nombres:
        candidato = raiz / nombre
        if candidato.exists():
            return candidato
    return None


def rutas_tablas(raiz: Path | None = None) -> dict[str, Path]:
    """Resuelve parquet en `data/raw` o en la raíz del repo (como el notebook inicial)."""
    base = DATOS_CRUDOS if raiz is None else Path(raiz)
    busqueda = [base, base.parent.parent if base.name == "raw" else base]
    encontrados: dict[str, Path] = {}
    for clave, nombres in {
        "features": [ARCHIVO_FEATURES],
        "lcs": [ARCHIVO_LCS],
        "oids": [ARCHIVO_OIDS],
    }.items():
        for carpeta in busqueda:
            ruta = _primer_existente(nombres, carpeta)
            if ruta is not None:
                encontrados[clave] = ruta
                break
    return encontrados


def cargar_features_y_oids(raiz: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga features y etiquetas. Las curvas de luz no hacen falta para los puntos 1 a 3."""
    rutas = rutas_tablas(raiz)
    faltan = {"features", "oids"} - set(rutas)
    if faltan:
        raise FileNotFoundError(
            "No se encontraron "
            + ", ".join(sorted(faltan))
            + ". Copia los parquet a data/raw/ o a la raíz del repositorio."
        )
    return pd.read_parquet(rutas["features"]), pd.read_parquet(rutas["oids"])


def cargar_tablas(raiz: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga features, curvas de luz y etiquetas."""
    rutas = rutas_tablas(raiz)
    faltan = {"features", "lcs", "oids"} - set(rutas)
    if faltan:
        raise FileNotFoundError(
            "No se encontraron "
            + ", ".join(sorted(faltan))
            + ". Copia los parquet a data/raw/ o a la raíz del repositorio."
        )
    return (
        pd.read_parquet(rutas["features"]),
        pd.read_parquet(rutas["lcs"]),
        pd.read_parquet(rutas["oids"]),
    )


def unir_features_etiquetas(
    df_features: pd.DataFrame, df_oids: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Alinea objetos con clase ALeRCE. Descarta filas sin etiqueta."""
    if "oid" not in df_features.columns:
        raise ValueError("df_features debe incluir la columna oid")
    etiquetas = df_features[["oid"]].merge(df_oids[["oid", "alerceclass"]], on="oid", how="left")[
        "alerceclass"
    ]
    mask = etiquetas.notna()
    x = df_features.loc[mask].drop(columns=["oid"]).reset_index(drop=True)
    oids = df_features.loc[mask, "oid"].reset_index(drop=True)
    y = etiquetas.loc[mask].reset_index(drop=True)
    return x, oids, y


def tablas_sinteticas(
    n_por_clase: int = 40, n_obs: int = 30, semilla: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera un mini-dataset con el esquema del notebook inicial."""
    rng = generador(semilla)
    clases = ["CEP", "DSCT", "EB", "LPV", "RRL"]
    periodos_clase = {"CEP": 10.0, "DSCT": 0.08, "EB": 1.2, "LPV": 200.0, "RRL": 0.5}
    filas = []
    lcs = []
    oids_rows = []
    feat_extra = [
        "Q31_r",
        "Rcs_r",
        "Skew_r",
        "Std_r",
        "Harmonics_mag_1_g",
        "Harmonics_mag_1_r",
        "GP_DRW_sigma_g",
        "MHPS_low_g",
        "MHPS_high_g",
        "MHPS_ratio_g",
        "MHPS_ratio_r",
        "r-W2",
        "g-W3",
        "dmag_non_det_fid_g",
        "last_mjd_before_fid_r",
        "gal_b",
        "sgscore1",
    ]
    idx = 0
    for clase in clases:
        for _ in range(n_por_clase):
            oid = f"ZTF{idx:08d}"
            idx += 1
            periodo = float(periodos_clase[clase] * rng.uniform(0.8, 1.2))
            fila = {
                "oid": oid,
                "Period_band_g": periodo,
                "Period_band_r": periodo * rng.uniform(0.95, 1.05),
                "Multiband_period_g_r": periodo,
            }
            for c in feat_extra:
                fila[c] = float(rng.normal(0, 1))
            if rng.random() < 0.1:
                fila["Q31_r"] = np.nan
            filas.append(fila)
            oids_rows.append({"oid": oid, "alerceclass": clase})
            mjd0 = 58000.0
            for k in range(n_obs):
                mjd = mjd0 + k * 0.2
                fase = (mjd % periodo) / periodo
                mag = 15.0 + 0.3 * np.sin(2 * np.pi * fase) + float(rng.normal(0, 0.02))
                lcs.append(
                    {
                        "oid": oid,
                        "mjd": mjd,
                        "fid": 1 if k % 2 == 0 else 2,
                        "magpsf_corr": mag,
                    }
                )
    return pd.DataFrame(filas), pd.DataFrame(lcs), pd.DataFrame(oids_rows)


def columna_magnitud(df_lcs: pd.DataFrame) -> str:
    """Elige la columna de magnitud disponible en las curvas de luz."""
    from ic_project.config import COLUMNAS_MAGNITUD_LC

    for col in COLUMNAS_MAGNITUD_LC:
        if col in df_lcs.columns:
            return col
    raise ValueError(
        f"No hay columna de magnitud conocida en lcs. Columnas: {list(df_lcs.columns)}"
    )
