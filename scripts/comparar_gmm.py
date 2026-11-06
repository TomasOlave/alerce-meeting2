"""Compara el GMM propio (ic_project.gmm) con el de scikit-learn sobre el latente del autoencoder.

Entrena el mismo autoencoder que scripts/run_meeting2.py (158 features, z de 8 dimensiones) y
ajusta los dos GMM con el latente de train. Revisa dos cosas:

1. Partiendo de los mismos parámetros iniciales, los dos EM llegan a los mismos parámetros.
2. Cada uno con su propia inicialización (k-means++ propio contra el KMeans de scikit-learn),
   cuánto se parecen las particiones y las métricas en test.

Guarda la tabla en results/comparacion_gmm.csv.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from ic_project.config import EPOCAS_AE, RESULTADOS, SEMILLA, Z_DIM  # noqa: E402
from ic_project.datos import cargar_features_y_oids, unir_features_etiquetas  # noqa: E402
from ic_project.gmm import MezclaGaussianas  # noqa: E402
from ic_project.modelos import entrenar_autoencoder, latente  # noqa: E402
from ic_project.preprocesamiento import preparar_conjuntos  # noqa: E402

K = 5


def main() -> None:
    df_features, df_oids = cargar_features_y_oids()
    x, oids, y = unir_features_etiquetas(df_features, df_oids)
    conjuntos, _, _ = preparar_conjuntos(x, y, oids, semilla=SEMILLA, solo_periodo_color=False)
    print("Entrenando autoencoder...")
    ae, _ = entrenar_autoencoder(
        conjuntos["x_train"],
        conjuntos["x_val"],
        z_dim=Z_DIM,
        n_epocas=EPOCAS_AE,
        batch=128,
        semilla=SEMILLA,
    )
    # en float64 para que scikit-learn no calcule en float32 y la comparación sea justa
    z_train = latente(ae, conjuntos["x_train"]).astype(np.float64)
    z_test = latente(ae, conjuntos["x_test"]).astype(np.float64)
    y_test = conjuntos["y_test"].to_numpy()

    print("\n1) Mismos parámetros iniciales")
    rng = np.random.default_rng(SEMILLA)
    inicio = {
        "weights_init": np.full(K, 1 / K),
        "means_init": z_train[rng.choice(len(z_train), size=K, replace=False)],
        "precisions_init": np.stack([np.linalg.inv(np.cov(z_train.T))] * K),
    }
    propio = MezclaGaussianas(K, tol=1e-6, max_iter=2000, **inicio).fit(z_train)
    sk = GaussianMixture(K, tol=1e-6, max_iter=2000, **inicio).fit(z_train)
    print(f"   iteraciones: propio {propio.n_iter_}, scikit-learn {sk.n_iter_}")
    for nombre, a, b in [
        ("pesos", propio.weights_, sk.weights_),
        ("medias", propio.means_, sk.means_),
        ("covarianzas", propio.covariances_, sk.covariances_),
        ("log-verosimilitud test", propio.score_samples(z_test), sk.score_samples(z_test)),
    ]:
        print(f"   máxima diferencia en {nombre}: {np.abs(a - b).max():.2e}")

    print("\n2) Cada uno con su inicialización (n_init=5, semilla 42), como en el notebook")
    filas = {}
    modelos = {}
    for nombre, modelo in [
        ("propio", MezclaGaussianas(K, n_init=5, random_state=SEMILLA)),
        ("scikit-learn", GaussianMixture(K, n_init=5, random_state=SEMILLA)),
    ]:
        t0 = time.perf_counter()
        modelo.fit(z_train)
        segundos = time.perf_counter() - t0
        pred = modelo.predict(z_test)
        modelos[nombre] = (modelo, pred)
        filas[nombre] = {
            "loglik_media_train": modelo.score(z_train),
            "loglik_media_test": modelo.score(z_test),
            "bic_train": modelo.bic(z_train),
            "iteraciones": modelo.n_iter_,
            "segundos": segundos,
            "ari_test": adjusted_rand_score(y_test, pred),
            "nmi_test": normalized_mutual_info_score(y_test, pred),
        }
    tabla = pd.DataFrame(filas).T
    print(tabla.round(4).to_string())

    (m_propio, pred_propio), (m_sk, pred_sk) = modelos["propio"], modelos["scikit-learn"]
    ll_propio, ll_sk = m_propio.score_samples(z_test), m_sk.score_samples(z_test)
    n_bajo = int(0.05 * len(z_test))
    peores_propio = set(np.argsort(ll_propio)[:n_bajo])
    peores_sk = set(np.argsort(ll_sk)[:n_bajo])
    acuerdo = adjusted_rand_score(pred_sk, pred_propio)
    correlacion = np.corrcoef(ll_propio, ll_sk)[0, 1]
    print(f"\n   ARI entre las dos particiones de test: {acuerdo:.3f}")
    print(f"   correlación de la log-verosimilitud por objeto: {correlacion:.4f}")
    print(f"   5% menos verosímil en común: {len(peores_propio & peores_sk)} de {n_bajo}")
    print("\n   componente scikit-learn (filas) contra componente propio (columnas), test:")
    print(pd.crosstab(pred_sk, pred_propio, rownames=["sklearn"], colnames=["propio"]).to_string())

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTADOS / "comparacion_gmm.csv")
    print(f"\nTabla en {RESULTADOS / 'comparacion_gmm.csv'}")


if __name__ == "__main__":
    main()
