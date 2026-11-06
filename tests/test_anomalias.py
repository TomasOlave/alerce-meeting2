"""Detección y caracterización de anomalías con MCDSVDD."""

from __future__ import annotations

import numpy as np

from ic_project.anomalias import (
    CANDIDATO,
    ERROR_DATOS,
    NORMAL,
    OTRA_CLASE,
    categorizar,
    detectar,
    features_desviadas,
    umbrales_por_centro,
)


def test_umbral_por_centro_es_el_percentil_de_su_centro() -> None:
    rng = np.random.default_rng(0)
    # centro 0: distancias chicas; centro 1: distancias grandes
    d0 = np.c_[rng.uniform(0, 1, 500), np.full(500, 10.0)]
    d1 = np.c_[np.full(500, 50.0), rng.uniform(0, 5, 500)]
    umbrales = umbrales_por_centro(np.vstack([d0, d1]), percentil=99)
    assert np.isclose(umbrales[0], np.percentile(d0[:, 0], 99))
    assert np.isclose(umbrales[1], np.percentile(d1[:, 1], 99))


def test_detectar_compara_con_el_umbral_de_su_centro() -> None:
    dist = np.array([[0.5, 9.0], [2.0, 9.0], [9.0, 4.0], [9.0, 6.0]])
    det = detectar(dist, umbrales=np.array([1.0, 5.0]))
    assert det["centro"].tolist() == [0, 0, 1, 1]
    assert det["anomalia"].tolist() == [False, True, False, True]
    assert np.allclose(det["razon_umbral"], [0.5, 2.0, 0.8, 1.2])


def test_categorias_con_prioridad_al_error_de_datos() -> None:
    categoria = categorizar(
        anomalia=np.array([False, True, True, True, True]),
        clase=np.array(["A", "A", "A", "A", "A"]),
        clase_del_centro=np.array(["A", "A", "B", "B", "A"]),
        calidad_ok=np.array([True, True, True, False, True]),
        periodo_alias=np.array([False, False, False, False, True]),
    )
    assert categoria.tolist() == [NORMAL, CANDIDATO, OTRA_CLASE, ERROR_DATOS, ERROR_DATOS]


def test_features_desviadas_respecto_de_su_clase() -> None:
    rng = np.random.default_rng(1)
    x_train = rng.normal(0, 1, size=(400, 3))
    y_train = np.array(["A"] * 200 + ["B"] * 200)
    x_train[y_train == "B", 1] += 10  # en la clase B la feature f1 vale ~10
    objeto = np.array([[0.0, 0.0, 6.0]])  # para A: f2 muy alta; para B: f1 muy baja
    para_a = features_desviadas(objeto, x_train, y_train, np.array(["A"]), ["f0", "f1", "f2"], k=1)
    para_b = features_desviadas(objeto, x_train, y_train, np.array(["B"]), ["f0", "f1", "f2"], k=1)
    assert para_a[0][0][0] == "f2" and para_a[0][0][1] > 4
    assert para_b[0][0][0] == "f1" and para_b[0][0][1] < -4
