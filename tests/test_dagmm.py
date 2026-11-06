"""DAGMM: la mezcla en torch coincide con gmm.py y el entrenamiento corre de punta a punta."""

from __future__ import annotations

import numpy as np
import torch

from ic_project.dagmm import (
    DAGMM,
    energia,
    entrenar_dagmm,
    parametros_mezcla,
    salidas_dagmm,
)
from ic_project.gmm import MezclaGaussianas, _paso_m


def _blobs(n: int = 300, d: int = 4, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    centros = rng.normal(0, 4, size=(3, d))
    return np.vstack([rng.normal(c, 1.0, size=(n // 3, d)) for c in centros])


def test_parametros_mezcla_son_el_paso_m_de_gmm() -> None:
    x = _blobs()
    resp = np.random.default_rng(1).dirichlet(np.ones(3), size=len(x))
    pesos, medias, cov = _paso_m(x, resp, 1e-6)
    phi, mu, sigma = parametros_mezcla(torch.tensor(x), torch.tensor(resp), 1e-6)
    assert np.allclose(phi.numpy(), pesos)
    assert np.allclose(mu.numpy(), medias)
    assert np.allclose(sigma.numpy(), cov)


def test_energia_es_menos_la_log_verosimilitud_de_gmm() -> None:
    x = _blobs()
    gmm = MezclaGaussianas(3, random_state=0).fit(x)
    e = energia(
        torch.tensor(x),
        torch.tensor(gmm.weights_),
        torch.tensor(gmm.means_),
        torch.tensor(gmm.covariances_),
    )
    assert np.allclose(e.numpy(), -gmm.score_samples(x))


def test_la_energia_deja_pasar_el_gradiente_al_latente() -> None:
    z = torch.tensor(_blobs(), requires_grad=True)
    gamma = torch.softmax(torch.randn(len(z), 3, dtype=torch.float64), dim=1)
    energia(z, *parametros_mezcla(z, gamma)).mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_entrenamiento_corre_y_entrega_salidas_validas() -> None:
    x = _blobs(n=240, d=12).astype(np.float32)
    modelo, hist = entrenar_dagmm(
        x[:180], x[180:], n_componentes=3, z_dim=2, n_epocas=3, batch=64, paciencia=5
    )
    assert isinstance(modelo, DAGMM)
    assert len(hist.val) == 3 and np.isfinite(hist.val).all()
    salida = salidas_dagmm(modelo, x[180:])
    assert salida["z"].shape == (60, 2 + 2)
    assert np.allclose(salida["gamma"].sum(axis=1), 1, atol=1e-5)
    assert np.allclose(salida["probabilidad"].sum(axis=1), 1)
    assert np.isfinite(salida["energia"]).all()
    assert np.allclose(modelo.phi.sum().item(), 1)
