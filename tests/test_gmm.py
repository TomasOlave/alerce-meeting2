"""La mezcla de gaussianas propia: EM, verosimilitud, BIC y comparación con scikit-learn.

scikit-learn se usa solo aquí, como referencia para validar la implementación.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from ic_project.gmm import MezclaGaussianas, kmeans, logsumexp

MEDIAS = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
COVARIANZAS = np.array(
    [
        [[1.0, 0.5], [0.5, 1.0]],
        [[0.5, 0.0], [0.0, 0.5]],
        [[2.0, -0.3], [-0.3, 0.5]],
    ]
)
PESOS = np.array([0.5, 0.3, 0.2])


def _tres_gaussianas(n: int = 600, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    n_k = (PESOS * n).astype(int)
    x = np.vstack(
        [
            rng.multivariate_normal(m, c, size=k)
            for m, c, k in zip(MEDIAS, COVARIANZAS, n_k, strict=True)
        ]
    )
    return x, np.repeat(np.arange(3), n_k)


def _log_densidad_directa(x: np.ndarray, modelo: MezclaGaussianas) -> np.ndarray:
    """log p(x) con la fórmula de la gaussiana escrita tal cual, sin Cholesky ni logaritmos."""
    d = x.shape[1]
    densidad = np.zeros(len(x))
    for pi, mu, sigma in zip(modelo.weights_, modelo.means_, modelo.covariances_, strict=True):
        dif = x - mu
        maha = np.einsum("ij,jk,ik->i", dif, np.linalg.inv(sigma), dif)
        densidad += pi * np.exp(-0.5 * maha) / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
    return np.log(densidad)


def test_logsumexp_coincide_con_la_formula_y_no_hace_underflow() -> None:
    a = np.array([[0.1, -2.0, 3.0], [1.0, 1.0, 1.0]])
    assert np.allclose(logsumexp(a, axis=1), np.log(np.exp(a).sum(axis=1)))
    muy_chicos = np.array([[-1000.0, -1001.0]])
    assert np.allclose(logsumexp(muy_chicos, axis=1), -1000.0 + np.log1p(np.exp(-1.0)))


def test_kmeans_separa_grupos_lejanos() -> None:
    x, y = _tres_gaussianas()
    etiquetas = kmeans(x, 3, np.random.default_rng(0))
    # cada grupo real cae casi entero en un cluster distinto
    mayoritario = [np.bincount(etiquetas[y == g], minlength=3).argmax() for g in range(3)]
    assert len(set(mayoritario)) == 3
    acierto = np.mean([np.mean(etiquetas[y == g] == mayoritario[g]) for g in range(3)])
    assert acierto > 0.95


@pytest.mark.filterwarnings("ignore:EM no convergió")
def test_em_nunca_baja_la_verosimilitud() -> None:
    # con un componente de más EM avanza lento: da muchas iteraciones para revisar
    x, _ = _tres_gaussianas()
    modelo = MezclaGaussianas(4, tol=1e-10, max_iter=300, random_state=0).fit(x)
    assert len(modelo.historial_) > 50
    assert np.all(np.diff(modelo.historial_) >= -1e-10)


def test_recupera_los_parametros_de_una_mezcla_conocida() -> None:
    x, _ = _tres_gaussianas(n=6000)
    modelo = MezclaGaussianas(3, n_init=3, random_state=0).fit(x)
    # los componentes salen en cualquier orden: se emparejan por la media más cercana
    orden = [int(np.argmin(((modelo.means_ - m) ** 2).sum(axis=1))) for m in MEDIAS]
    assert sorted(orden) == [0, 1, 2]
    assert np.allclose(modelo.means_[orden], MEDIAS, atol=0.1)
    assert np.allclose(modelo.weights_[orden], PESOS, atol=0.02)
    assert np.allclose(modelo.covariances_[orden], COVARIANZAS, atol=0.15)


def test_verosimilitud_y_probabilidades_son_consistentes() -> None:
    x, _ = _tres_gaussianas()
    modelo = MezclaGaussianas(3, random_state=0).fit(x)
    assert np.allclose(modelo.score_samples(x), _log_densidad_directa(x, modelo))
    proba = modelo.predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(modelo.predict(x), proba.argmax(axis=1))
    assert np.allclose(modelo.precisions_, np.linalg.inv(modelo.covariances_))
    assert modelo.score(x) == pytest.approx(modelo.score_samples(x).mean())


def test_igual_que_sklearn_partiendo_de_los_mismos_parametros() -> None:
    x, _ = _tres_gaussianas()
    rng = np.random.default_rng(1)
    inicio = {
        "weights_init": np.full(3, 1 / 3),
        "means_init": x[rng.choice(len(x), size=3, replace=False)],
        "precisions_init": np.stack([np.eye(2)] * 3),
    }
    propio = MezclaGaussianas(3, tol=1e-8, max_iter=1000, **inicio).fit(x)
    referencia = GaussianMixture(3, tol=1e-8, max_iter=1000, **inicio).fit(x)

    assert propio.n_iter_ == referencia.n_iter_
    assert np.allclose(propio.weights_, referencia.weights_, atol=1e-8)
    assert np.allclose(propio.means_, referencia.means_, atol=1e-8)
    assert np.allclose(propio.covariances_, referencia.covariances_, atol=1e-8)
    assert propio.lower_bound_ == pytest.approx(referencia.lower_bound_, abs=1e-10)
    assert np.allclose(propio.score_samples(x), referencia.score_samples(x), atol=1e-8)
    assert np.allclose(propio.predict_proba(x), referencia.predict_proba(x), atol=1e-8)
    assert propio.bic(x) == pytest.approx(referencia.bic(x), rel=1e-10)
    assert propio.aic(x) == pytest.approx(referencia.aic(x), rel=1e-10)


def test_bic_elige_el_numero_correcto_de_componentes() -> None:
    x, _ = _tres_gaussianas()
    bic = {k: MezclaGaussianas(k, n_init=3, random_state=0).fit(x).bic(x) for k in range(1, 6)}
    assert min(bic, key=bic.get) == 3


def test_misma_semilla_mismo_resultado() -> None:
    x, _ = _tres_gaussianas()
    a = MezclaGaussianas(3, n_init=2, random_state=7).fit(x)
    b = MezclaGaussianas(3, n_init=2, random_state=7).fit(x)
    assert np.array_equal(a.means_, b.means_)
    assert np.array_equal(a.historial_, b.historial_)


def test_errores_claros() -> None:
    with pytest.raises(ValueError, match="full"):
        MezclaGaussianas(2, covariance_type="diag")
    with pytest.raises(RuntimeError, match="no está ajustado"):
        MezclaGaussianas(2).predict(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="al menos"):
        MezclaGaussianas(5).fit(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="NaN"):
        MezclaGaussianas(1).fit(np.array([[0.0, np.nan], [1.0, 1.0]]))
