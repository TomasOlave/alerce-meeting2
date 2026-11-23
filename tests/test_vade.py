"""VaDE: fórmulas de la ELBO, inicialización desde el autoencoder y entrenamiento."""

from __future__ import annotations

import math

import numpy as np
import torch

from ic_project.modelos import entrenar_autoencoder, latente
from ic_project.vade import VaDE, entrenar_vade, log_gaussianas_diagonales, salidas_vade


def _blobs(n_por_clase: int = 80, d: int = 12, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    centros = rng.normal(0, 4, size=(3, d))
    x = np.vstack([rng.normal(c, 1.0, size=(n_por_clase, d)) for c in centros])
    return x.astype(np.float32), np.repeat(np.arange(3), n_por_clase)


def test_log_gaussianas_diagonales_es_la_formula_directa() -> None:
    rng = np.random.default_rng(0)
    z, medias = rng.normal(size=(7, 3)), rng.normal(size=(2, 3))
    var = rng.uniform(0.5, 2.0, size=(2, 3))
    obtenido = log_gaussianas_diagonales(
        torch.tensor(z), torch.tensor(medias), torch.tensor(np.log(var))
    ).numpy()
    for c in range(2):
        densidad = np.prod(
            np.exp(-((z - medias[c]) ** 2) / (2 * var[c])) / np.sqrt(2 * math.pi * var[c]), axis=1
        )
        assert np.allclose(obtenido[:, c], np.log(densidad))


def test_con_clases_la_kl_de_z_es_la_kl_entre_gaussianas() -> None:
    torch.manual_seed(0)
    modelo = VaDE(in_dim=6, z_dim=3, n_componentes=2).eval()
    with torch.no_grad():
        modelo.medias_prior.normal_()
        modelo.log_var_prior.uniform_(-1, 1)
        modelo.logits_pi.copy_(torch.tensor([0.3, -0.2]))
    x = torch.randn(5, 6)
    y = torch.tensor([0, 1, 1, 0, 1])
    t = modelo.terminos(x, y)
    media, log_var = modelo.encode(x)
    mu_p, log_var_p = modelo.medias_prior[y], modelo.log_var_prior[y]
    # KL(N(m, s2) || N(mu, sigma2)) = 1/2 sum [log sigma2 - log s2 + (s2 + (m - mu)^2) / sigma2 - 1]
    kl = 0.5 * (
        log_var_p - log_var + (log_var.exp() + (media - mu_p) ** 2) / log_var_p.exp() - 1
    ).sum(dim=1)
    assert torch.allclose(t.kl_z, kl, atol=1e-5)
    assert torch.allclose(t.kl_c, -torch.log_softmax(modelo.logits_pi, 0)[y])
    assert torch.allclose(t.gamma.sum(dim=1), torch.ones(5))


def test_sin_clases_gamma_es_la_posterior_y_la_kl_de_c_no_es_negativa() -> None:
    torch.manual_seed(1)
    modelo = VaDE(in_dim=6, z_dim=3, n_componentes=4).eval()
    with torch.no_grad():
        modelo.medias_prior.normal_(0, 2)
    x = torch.randn(9, 6)
    t = modelo.terminos(x)
    media, _ = modelo.encode(x)
    assert torch.allclose(t.gamma, torch.softmax(modelo.log_conjunta(media), dim=1))
    assert (t.kl_c >= -1e-6).all()  # es una KL
    assert torch.isfinite(t.total).all()


def test_inicializar_copia_el_autoencoder_y_pone_el_prior_en_el_latente() -> None:
    x, y = _blobs()
    ae, _ = entrenar_autoencoder(x[:200], x[200:], z_dim=2, n_epocas=2, batch=64)
    z = latente(ae, x[:200])

    modelo = VaDE(x.shape[1], z_dim=2, n_componentes=3)
    modelo.inicializar(ae, x[:200], y[:200])
    s = salidas_vade(modelo, x[:200])
    assert np.allclose(s["z"], z, atol=1e-5)  # el encoder (media) es el del autoencoder
    medias = modelo.medias_prior.detach().cpu().numpy()
    for c in range(3):
        assert np.allclose(medias[c], z[y[:200] == c].mean(axis=0), atol=1e-5)
    pi = torch.softmax(modelo.logits_pi, 0).detach().cpu().numpy()
    assert np.allclose(pi, np.bincount(y[:200]) / 200)

    sin_clases = VaDE(x.shape[1], z_dim=2, n_componentes=3)
    sin_clases.inicializar(ae, x[:200])
    assert np.isclose(torch.softmax(sin_clases.logits_pi, 0).sum().item(), 1)


def test_entrenamiento_corre_en_las_dos_versiones() -> None:
    x, y = _blobs()
    orden = np.random.default_rng(0).permutation(len(x))
    x, y = x[orden], y[orden]
    x_tr, x_va, y_tr, y_va = x[:180], x[180:], y[:180], y[180:]
    ae, _ = entrenar_autoencoder(x_tr, x_va, z_dim=2, n_epocas=3, batch=64)

    modelo, hist = entrenar_vade(x_tr, x_va, ae, n_componentes=3, z_dim=2, n_epocas=3, batch=64)
    assert len(hist.val) == 3 and np.isfinite(hist.val).all()
    s = salidas_vade(modelo, x_va)
    assert s["z"].shape == (60, 2) and s["gamma"].shape == (60, 3)
    assert np.allclose(s["gamma"].sum(axis=1), 1, atol=1e-5)
    assert np.isfinite(s["menos_elbo"]).all() and np.isfinite(s["energia_latente"]).all()

    con_clases, hist = entrenar_vade(
        x_tr, x_va, ae, y_train=y_tr, y_val=y_va, z_dim=2, n_epocas=3, batch=64
    )
    assert con_clases.n_componentes == 3
    # con clases separadas y el prior en la media de cada clase, el componente es la clase
    acierto = (salidas_vade(con_clases, x_va)["componente"] == y_va).mean()
    assert acierto > 0.8
