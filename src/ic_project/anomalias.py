"""Detección y caracterización de anomalías con MCDSVDD.

1. Puntaje: distancia al centro más cercano (Pérez-Carrasco et al. 2023, ecuación 6).
2. Umbral por centro, fijado con un conjunto de calibración: percentil P de los puntajes de los
   objetos de ese conjunto que quedan en ese centro. Un objeto es anómalo si su puntaje supera el
   umbral de su centro. Un umbral por centro porque cada clase tiene su propia dispersión
   (Pérez-Carrasco et al. 2023, figura 2); fijado de antemano para poder aplicarlo a objetos
   nuevos (Tsang & Schultz 2019 usan un percentil de train). Conviene calibrar con validación y no
   con train: MCDSVDD acerca los objetos de train a sus centros, así que en train las distancias
   quedan más chicas que en datos nuevos y el umbral marcaría más de lo esperado.
3. Categorías, inspiradas en la inspección visual del paper:
   - "posible error de datos": no cumple el filtro de calidad del paper (al menos 6 detecciones en
     g y en r) o su periodo cae en un alias de 1 día;
   - "otra clase": su centro más cercano no es el de su clase;
   - "candidato": el resto.
4. Caracterización: features en que cada anomalía más se aleja de lo típico de su clase en train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ic_project.modelos import CodificadorMCDSVDD, latente

NORMAL = "normal"
ERROR_DATOS = "posible error de datos"
OTRA_CLASE = "otra clase"
CANDIDATO = "candidato"


def distancias(modelo: CodificadorMCDSVDD, x: np.ndarray) -> np.ndarray:
    """Distancia euclidiana de cada objeto a cada centro, en el latente de MCDSVDD. (N, K)."""
    z = latente(modelo, x)
    centros = modelo.centros.detach().cpu().numpy()
    return np.sqrt(((z[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2))


def umbrales_por_centro(dist_calibracion: np.ndarray, percentil: float = 99) -> np.ndarray:
    """Para cada centro, el percentil de la distancia de los objetos de calibración (validación)
    más cercanos a él."""
    centro = dist_calibracion.argmin(axis=1)
    puntaje = dist_calibracion.min(axis=1)
    return np.array(
        [
            np.percentile(puntaje[centro == k], percentil) if (centro == k).any() else np.inf
            for k in range(dist_calibracion.shape[1])
        ]
    )


def detectar(dist: np.ndarray, umbrales: np.ndarray) -> pd.DataFrame:
    """Centro más cercano, puntaje, puntaje / umbral de ese centro y si es anomalía (> 1)."""
    centro = dist.argmin(axis=1)
    puntaje = dist.min(axis=1)
    razon = puntaje / umbrales[centro]
    return pd.DataFrame(
        {"centro": centro, "puntaje": puntaje, "razon_umbral": razon, "anomalia": razon > 1}
    )


def categorizar(
    anomalia: np.ndarray,
    clase: np.ndarray,
    clase_del_centro: np.ndarray,
    calidad_ok: np.ndarray,
    periodo_alias: np.ndarray,
) -> np.ndarray:
    """Categoría de cada objeto. Los no anómalos quedan como "normal". El error de datos tiene
    prioridad: si los datos son dudosos, no se puede afirmar que la clase sea otra."""
    error = ~np.asarray(calidad_ok, dtype=bool) | np.asarray(periodo_alias, dtype=bool)
    otra = np.asarray(clase) != np.asarray(clase_del_centro)
    categoria = np.where(error, ERROR_DATOS, np.where(otra, OTRA_CLASE, CANDIDATO))
    return np.where(np.asarray(anomalia, dtype=bool), categoria, NORMAL)


def features_desviadas(
    x: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    clase: np.ndarray,
    columnas: list[str],
    k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Para cada objeto, las k features con mayor |z| respecto de su clase en train, donde
    z = (valor - media de la clase) / desviación de la clase."""
    y_train = np.asarray(y_train)
    estadisticos = {}
    for c in np.unique(y_train):
        de_la_clase = x_train[y_train == c]
        desviacion = de_la_clase.std(axis=0)
        estadisticos[c] = (de_la_clase.mean(axis=0), np.where(desviacion > 0, desviacion, 1.0))
    salida = []
    for fila, c in zip(x, np.asarray(clase), strict=True):
        media, desviacion = estadisticos[c]
        z = (fila - media) / desviacion
        orden = np.argsort(-np.abs(z))[:k]
        salida.append([(columnas[j], float(z[j])) for j in orden])
    return salida
