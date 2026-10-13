"""Partición y transformaciones ajustadas solo con el conjunto de entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ic_project.config import (
    COLUMNAS_ARCSINH,
    COLUMNAS_MODELO,
    COLUMNAS_PERIODO,
    PROPORCION_TEST,
    PROPORCION_VAL,
    SEMILLA,
)


def columnas_a_descartar(columnas: pd.Index) -> list[str]:
    """No-detecciones previas a la primera alerta y colores WISE (NaN no físicos)."""
    bloque_before = [
        c
        for c in columnas
        if ("_before_fid_" in c) or ("dmag_non_det_" in c) or ("dmag_first_det_" in c)
    ]
    bloque_wise = [c for c in columnas if ("W1" in c or "W2" in c or "W3" in c)]
    return sorted(set(bloque_before) | set(bloque_wise))


def columnas_del_modelo(columnas: pd.Index) -> list[str]:
    """Periodo y color g-r, en un orden fijo. Si falta alguna, se omite."""
    return [c for c in COLUMNAS_MODELO if c in columnas]


def _columnas_log(frame: pd.DataFrame) -> list[str]:
    candidatas = [
        c
        for c in frame.columns
        if c.startswith("Harmonics_mag_")
        or c.startswith("GP_DRW_")
        or c.startswith("MHPS_low")
        or c.startswith("MHPS_high")
    ]
    validas = []
    for c in candidatas:
        s = frame[c].dropna()
        if len(s) == 0 or (s >= 0).all():
            validas.append(c)
    return validas


def especificar_transformaciones(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Elige columnas a transformar usando solo el frame entregado (el de train)."""
    return {
        "log": _columnas_log(frame),
        "periodo": [c for c in COLUMNAS_PERIODO if c in frame.columns],
        "arcsinh": [c for c in COLUMNAS_ARCSINH if c in frame.columns],
    }


def aplicar_transformaciones_elemento(
    frame: pd.DataFrame, spec: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """log1p / log10 / arcsinh. Si `spec` viene de train, val y test usan las mismas columnas."""
    out = frame.copy()
    if spec is None:
        spec = especificar_transformaciones(out)
    cols_log = [c for c in spec["log"] if c in out.columns]
    if cols_log:
        out[cols_log] = np.log1p(out[cols_log].clip(lower=0))
    for c in spec["periodo"]:
        if c in out.columns:
            out[c] = np.log10(out[c].clip(lower=1e-8))
    cols_a = [c for c in spec["arcsinh"] if c in out.columns]
    if cols_a:
        out[cols_a] = np.arcsinh(out[cols_a])
    return out


@dataclass
class Preprocesador:
    """Winsorización, imputación mediana y z-score, todos aprendidos en train.

    La escala es la desviación estándar y no el IQR: con IQR las features de cola
    pesada (Harmonics_mag_*, delta_period_*) quedaban con valores de cientos y se
    llevaban casi todo el MSE del autoencoder.
    """

    cuantiles_inf_: pd.Series = field(default=None, repr=False)
    cuantiles_sup_: pd.Series = field(default=None, repr=False)
    mediana_: pd.Series = field(default=None, repr=False)
    media_: pd.Series = field(default=None, repr=False)
    escala_: pd.Series = field(default=None, repr=False)
    columnas_: list[str] = field(default_factory=list)

    def ajustar(self, x_train: pd.DataFrame) -> Preprocesador:
        self.columnas_ = list(x_train.columns)
        self.cuantiles_inf_ = x_train.quantile(0.01)
        self.cuantiles_sup_ = x_train.quantile(0.99)
        recortado = self._recortar(x_train)
        self.mediana_ = recortado.median()
        imputado = recortado.fillna(self.mediana_)
        self.media_ = imputado.mean()
        # una columna constante queda en 0 en vez de dividir por 0
        self.escala_ = imputado.std(ddof=0).replace(0, 1.0)
        return self

    def _recortar(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.clip(lower=self.cuantiles_inf_, upper=self.cuantiles_sup_, axis=1)

    def transformar(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.mediana_ is None:
            raise RuntimeError("El preprocesador no está ajustado")
        x = x[self.columnas_]
        recortado = self._recortar(x)
        imputado = recortado.fillna(self.mediana_)
        return (imputado - self.media_) / self.escala_


def particionar(
    x: pd.DataFrame,
    y: pd.Series,
    oids: pd.Series,
    semilla: int = SEMILLA,
    test: float = PROPORCION_TEST,
    val: float = PROPORCION_VAL,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Train / val / test estratificado. Ningún objeto se comparte entre conjuntos."""
    x_tv, x_test, y_tv, y_test, oid_tv, oid_test = train_test_split(
        x, y, oids, test_size=test, random_state=semilla, stratify=y
    )
    val_relativo = val / (1.0 - test)
    x_train, x_val, y_train, y_val, oid_train, oid_val = train_test_split(
        x_tv, y_tv, oid_tv, test_size=val_relativo, random_state=semilla, stratify=y_tv
    )
    return {
        "x_train": x_train.reset_index(drop=True),
        "x_val": x_val.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "oid_train": oid_train.reset_index(drop=True),
        "oid_val": oid_val.reset_index(drop=True),
        "oid_test": oid_test.reset_index(drop=True),
    }


def preparar_conjuntos(
    x: pd.DataFrame,
    y: pd.Series,
    oids: pd.Series,
    semilla: int = SEMILLA,
) -> tuple[dict[str, np.ndarray | pd.Series], Preprocesador, list[str]]:
    """Descarta columnas problemáticas, se queda con periodo y color, y ajusta solo en train."""
    malas = columnas_a_descartar(x.columns)
    x_sel = x.drop(columns=malas, errors="ignore")
    quedan = columnas_del_modelo(x_sel.columns)
    if not quedan:
        raise ValueError("No quedaron columnas de periodo o color g-r para el modelo")
    sobran = [c for c in x_sel.columns if c not in quedan]
    descartadas = sorted(set(malas) | set(sobran))
    x_sel = x_sel.loc[:, quedan]
    partes = particionar(x_sel, y, oids, semilla=semilla)
    spec = especificar_transformaciones(partes["x_train"])
    periodos: dict[str, pd.Series] = {}
    for split in ("train", "val", "test"):
        frame = partes[f"x_{split}"]
        if "Multiband_period_g_r" in frame.columns:
            periodos[split] = frame["Multiband_period_g_r"].copy()
        elif "Period_band_r" in frame.columns:
            periodos[split] = frame["Period_band_r"].copy()
        else:
            periodos[split] = pd.Series(np.nan, index=frame.index)
        partes[f"x_{split}"] = aplicar_transformaciones_elemento(frame, spec)

    prep = Preprocesador().ajustar(partes["x_train"])
    salida: dict[str, np.ndarray | pd.Series] = {}
    for split in ("train", "val", "test"):
        salida[f"x_{split}"] = prep.transformar(partes[f"x_{split}"]).to_numpy(dtype=np.float32)
        salida[f"y_{split}"] = partes[f"y_{split}"]
        salida[f"oid_{split}"] = partes[f"oid_{split}"]
        salida[f"periodo_{split}"] = periodos[split].reset_index(drop=True)
    return salida, prep, descartadas
