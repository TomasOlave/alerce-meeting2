"""GMM por clase: elección de K, puntaje de anomalía y clasificación. Solo NumPy."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ic_project.gmm import MezclaGaussianas, logsumexp
from ic_project.gmm_clase import GMMPorClase, n_parametros_gmm


def _dos_clases(semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Clase "A": una gaussiana. Clase "B": tres gaussianas separadas."""
    rng = np.random.default_rng(semilla)
    a = rng.normal([0.0, 0.0], 0.5, size=(400, 2))
    b = np.vstack([rng.normal(c, 0.4, size=(200, 2)) for c in ([8, 0], [8, 6], [14, 3])])
    z = np.vstack([a, b])
    y = np.array(["A"] * len(a) + ["B"] * len(b))
    return z, y


def test_n_parametros_coincide_con_mezcla_gaussianas() -> None:
    z, _ = _dos_clases()
    gmm = MezclaGaussianas(3, random_state=0).fit(z)
    assert n_parametros_gmm(3, 2) == gmm._n_parametros()


def test_bic_elige_el_numero_de_componentes_de_cada_clase() -> None:
    z, y = _dos_clases()
    modelo = GMMPorClase(k_max=5, random_state=0).fit(z, y)
    assert modelo.k_ == {"A": 1, "B": 3}
    assert list(modelo.clases_) == ["A", "B"]
    assert np.allclose(np.exp(modelo.log_prior_), [400 / 1000, 600 / 1000])


def test_criterio_validacion_tambien_recupera_los_componentes() -> None:
    z, y = _dos_clases(0)
    z_val, y_val = _dos_clases(1)
    modelo = GMMPorClase(k_max=4, criterio="val", random_state=0).fit(z, y, z_val, y_val)
    assert modelo.k_["B"] == 3
    with pytest.raises(ValueError, match="z_val"):
        GMMPorClase(criterio="val").fit(z, y)


def test_log_verosimilitud_es_la_del_gmm_de_cada_clase() -> None:
    z, y = _dos_clases()
    modelo = GMMPorClase(k_max=4, random_state=0).fit(z, y)
    ll = modelo.log_verosimilitud_por_clase(z)
    for j, gmm in enumerate(modelo.modelos_):
        assert np.allclose(ll[:, j], gmm.score_samples(z))
    # los dos modos del puntaje con su fórmula
    assert np.allclose(modelo.puntaje_anomalia(z, "max"), -ll.max(axis=1))
    mezcla = -logsumexp(ll + modelo.log_prior_, axis=1)
    assert np.allclose(modelo.puntaje_anomalia(z, "mezcla"), mezcla)
    assert np.allclose(modelo.score_samples(z), -mezcla)
    assert np.allclose(modelo.predict_proba(z).sum(axis=1), 1)


def test_clasifica_clases_separadas() -> None:
    z, y = _dos_clases()
    z_test, y_test = _dos_clases(5)
    modelo = GMMPorClase(k_max=4, random_state=0).fit(z, y)
    assert (modelo.predict(z_test) == y_test).mean() > 0.99


def test_detecta_una_clase_escondida_en_un_hueco_entre_las_conocidas() -> None:
    # Las conocidas forman una cruz: A estirada en x y B estirada en y. La clase escondida C cae en
    # una esquina de la cruz: un GMM sin etiquetas de 1 componente la cubre con su elipse, pero
    # ninguna de las dos clases la explica.
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(0, 3, 600), rng.normal(0, 0.3, 600)])
    b = np.column_stack([rng.normal(0, 0.3, 600), rng.normal(0, 3, 600)])
    c = rng.normal([2.5, 2.5], 0.3, size=(60, 2))
    z_train = np.vstack([a[:500], b[:500]])
    y_train = np.array(["A"] * 500 + ["B"] * 500)
    z_test = np.vstack([a[500:], b[500:], c])
    etiqueta = np.r_[np.zeros(200), np.ones(60)]

    por_clase = GMMPorClase(k_max=1, random_state=0).fit(z_train, y_train)
    sin_etiquetas = MezclaGaussianas(1, random_state=0).fit(z_train)
    auroc_clase = roc_auc_score(etiqueta, por_clase.puntaje_anomalia(z_test))
    auroc_global = roc_auc_score(etiqueta, -sin_etiquetas.score_samples(z_test))
    assert auroc_clase > 0.99
    assert auroc_clase > auroc_global + 0.2


def test_salta_los_k_con_mas_parametros_que_estrellas() -> None:
    rng = np.random.default_rng(0)
    # 30 estrellas en 5 dimensiones: K=1 tiene 20 parámetros, K=2 tiene 41
    z = rng.normal(size=(60, 5))
    y = np.array([0] * 30 + [1] * 30)
    modelo = GMMPorClase(k_max=4, random_state=0).fit(z, y)
    assert modelo.k_ == {0: 1, 1: 1}
    assert set(modelo.seleccion_[0]) == {1}


def test_errores() -> None:
    z, y = _dos_clases()
    with pytest.raises(RuntimeError):
        GMMPorClase().puntaje_anomalia(z)
    with pytest.raises(ValueError, match="modo"):
        GMMPorClase(k_max=1).fit(z, y).puntaje_anomalia(z, "otro")
    with pytest.raises(ValueError, match="menos de 2"):
        GMMPorClase().fit(z[:401], y[:401])
    with pytest.raises(ValueError):
        GMMPorClase(k_max=0)
