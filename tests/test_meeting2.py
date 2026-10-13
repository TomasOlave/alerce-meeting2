"""La partición no filtra información y el pipeline del meeting corre en datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ic_project.datos import tablas_sinteticas, unir_features_etiquetas
from ic_project.experimento import correr_meeting2
from ic_project.preprocesamiento import (
    aplicar_transformaciones_elemento,
    especificar_transformaciones,
    preparar_conjuntos,
)


def test_oids_no_se_comparten_y_la_escala_sale_de_train() -> None:
    features, _, oids = tablas_sinteticas(n_por_clase=30, semilla=0)
    x, oid, y = unir_features_etiquetas(features, oids)
    conjuntos, _, _ = preparar_conjuntos(x, y, oid, semilla=0)
    train = set(conjuntos["oid_train"])
    val = set(conjuntos["oid_val"])
    test = set(conjuntos["oid_test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert np.allclose(conjuntos["x_train"].mean(axis=0), 0, atol=1e-5)
    desviacion = conjuntos["x_train"].std(axis=0)
    assert np.allclose(desviacion[desviacion > 0], 1, atol=1e-4)


def test_se_puede_entrenar_con_todas_las_features_no_descartadas() -> None:
    features, _, oids = tablas_sinteticas(n_por_clase=30, semilla=0)
    x, oid, y = unir_features_etiquetas(features, oids)
    solo_modelo, prep_modelo, _ = preparar_conjuntos(x, y, oid, semilla=0)
    todas, prep_todas, descartadas = preparar_conjuntos(
        x, y, oid, semilla=0, solo_periodo_color=False
    )
    assert set(prep_modelo.columnas_) < set(prep_todas.columnas_)
    assert todas["x_train"].shape[1] == x.shape[1] - len(descartadas)
    assert solo_modelo["oid_train"].equals(todas["oid_train"])


def test_las_columnas_log_se_eligen_solo_con_train() -> None:
    train = pd.DataFrame({"Harmonics_mag_1_g": [0.1, 0.2, 0.3], "Period_band_r": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"Harmonics_mag_1_g": [-5.0, 0.2, 0.3], "Period_band_r": [1.0, 2.0, 4.0]})
    spec = especificar_transformaciones(train)
    assert "Harmonics_mag_1_g" in spec["log"]
    out = aplicar_transformaciones_elemento(test, spec)
    assert out["Harmonics_mag_1_g"].iloc[0] == np.log1p(0)


def test_pipeline_meeting2_en_sintetico(tmp_path) -> None:
    features, _, oids = tablas_sinteticas(n_por_clase=24, n_obs=4, semilla=1)
    resultado = correr_meeting2(
        tablas=(features, oids),
        z_dim=4,
        epocas_ae=2,
        epocas_mcd=2,
        batch=32,
        n_init_gmm=1,
        con_importancia=False,
        directorio=tmp_path,
        verbose=False,
    )
    n_test = resultado["tamanos"]["test"]
    assert int(resultado["tabla_gmm"].to_numpy().sum()) == n_test
    assert int(resultado["tabla_mcdsvdd"].to_numpy().sum()) == n_test
    assert set(resultado["comparacion"].columns) >= {"ari", "nmi", "silhouette_clases"}
    assert (tmp_path / "ae_latente_clases.png").exists()
    assert (tmp_path / "gmm_clase_cluster.png").exists()
    assert (tmp_path / "mcdsvdd_latente_clases.png").exists()
    assert (tmp_path / "metricas_test.csv").exists()
