"""MCDSVDD con el protocolo de Pérez-Carrasco et al. (2023): pérdida (ecuación 5) y evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ic_project.datos import tablas_sinteticas, unir_features_etiquetas
from ic_project.modelos import entrenar_mcdsvdd, perdida_mcdsvdd
from ic_project.protocolo import (
    construir_ts2,
    correr_protocolo,
    filtro_detecciones,
    filtro_flags,
    particionar_protocolo,
    resumir_protocolo,
)


def test_filtro_detecciones_exige_6_en_cada_banda() -> None:
    x = pd.DataFrame({"n_det_g": [6, 10, 5, np.nan], "n_det_r": [6, 3, 20, 8]})
    assert filtro_detecciones(x).tolist() == [True, False, False, False]


def test_filtro_flags_exige_corregidas_y_sin_dudosas_en_ambas_bandas() -> None:
    lcs = pd.DataFrame(
        {
            "oid": ["a", "a", "b", "b", "c", "c", "d"],
            "fid": [1, 2, 1, 2, 1, 2, 1],
            "corrected": [True, True, True, False, True, True, True],
            "dubious": [False, False, False, False, False, True, False],
        }
    )
    # a: todo bien | b: r sin corregir | c: una dudosa en r | d: no tiene banda r
    assert filtro_flags(pd.Series(["a", "b", "c", "d", "e"]), lcs).tolist() == [
        True, False, False, False, False,
    ]


def test_perdida_es_la_ecuacion_5_del_paper() -> None:
    torch.manual_seed(0)
    centros = torch.randn(3, 4)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 2])
    z = torch.randn(len(y), 4)
    d2 = ((z - centros[y]) ** 2).sum(dim=1)
    # sum_j (1/M_j) sum_{i: y_i=j} ||z_i - c_j||^2
    esperado = sum(d2[y == j].mean() for j in range(3))
    assert torch.isclose(perdida_mcdsvdd(z, y, centros), esperado)
    assert torch.isclose(perdida_mcdsvdd(z, y, centros, balancear_clases=False), d2.mean())


def test_con_la_ecuacion_5_cada_clase_pesa_lo_mismo() -> None:
    centros = torch.zeros(2, 1)
    # la clase 0 tiene 9 objetos a distancia 1 y la clase 1 uno solo a distancia 1
    z = torch.ones(10, 1)
    y = torch.tensor([0] * 9 + [1])
    # duplicar los objetos de la clase grande no cambia la pérdida
    z2 = torch.ones(19, 1)
    y2 = torch.tensor([0] * 18 + [1])
    assert torch.isclose(perdida_mcdsvdd(z, y, centros), perdida_mcdsvdd(z2, y2, centros))


def test_mcdsvdd_exige_autoencoder_preentrenado() -> None:
    x = np.zeros((10, 3), dtype=np.float32)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="preentrenado"):
        entrenar_mcdsvdd(x, y, x, y)


def test_particion_y_ts2_no_filtran_la_clase_anomala() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice(["A", "B", "C", "D"], size=2000, p=[0.4, 0.3, 0.2, 0.1])
    part = particionar_protocolo(y, n_folds=5, semilla=0)
    assert set(part.train).isdisjoint(part.ts1)
    assert len(part.train) + len(part.ts1) == len(y)
    validaciones = np.concatenate([va for _, va in part.folds])
    assert sorted(validaciones) == sorted(part.train)  # cada objeto valida en un solo fold
    for tr, va in part.folds:
        assert set(tr).isdisjoint(va)

    idx, etiqueta = construir_ts2(y, part, "D", semilla=0)
    assert np.isclose(etiqueta.mean(), 0.1, atol=0.002)
    assert (y[idx[etiqueta == 1]] == "D").all()
    assert (y[idx[etiqueta == 0]] != "D").all()
    assert set(idx[etiqueta == 0]) <= set(part.ts1)  # los inliers salen solo de TS1
    assert len(set(idx)) == len(idx)
    idx2, _ = construir_ts2(y, part, "D", semilla=0)
    assert np.array_equal(idx, idx2)


def test_protocolo_corre_de_punta_a_punta() -> None:
    features, _, oids = tablas_sinteticas(n_por_clase=40, n_obs=4, semilla=1)
    x, _, y = unir_features_etiquetas(features, oids)
    resultados = correr_protocolo(
        x, y.to_numpy(), clases=["DSCT", "LPV"], n_folds=2, z_dim=3, epocas_ae=2,
        con_gmm=True, con_dagmm=True, epocas_dagmm=2, n_epocas=3, batch=64, verbose=False,
    )
    assert len(resultados) == 4
    for detector in ("mcdsvdd", "ae", "gmm", "dagmm"):
        assert resultados[f"auroc_{detector}"].between(0, 1).all()
    resumen = resumir_protocolo(resultados)
    assert list(resumen.index) == ["DSCT", "LPV"]
    assert {"gmm", "gmm_desv", "dagmm", "dagmm_desv"} <= set(resumen.columns)
    assert resumen.loc["LPV", "paper_mcdsvdd"] == 0.953
