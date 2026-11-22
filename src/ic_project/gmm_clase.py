"""GMM por clase: una mezcla de gaussianas para cada clase conocida. Solo usa NumPy.

El GMM de gmm.py se ajusta sin etiquetas sobre el latente de todas las estrellas. En el latente del
autoencoder las clases se pisan (silhouette de las clases ~0), así que EM arma grupos por período y
calidad de observación, no por clase. Como detector de anomalías eso tiene un problema: si se
esconde RRL, sus estrellas caen en una zona densa que comparte con EB y DSCT, y el GMM las ve
normales (AUROC 0.45 en el protocolo, peor que el azar).

Aquí cada clase conocida c tiene su propio GMM con K_c componentes:

    p(z | c) = sum_k pi_ck N(z | mu_ck, Sigma_ck),

ajustado con EM (MezclaGaussianas) usando solo las estrellas de esa clase. K_c se elige por clase
con BIC (o con la log-verosimilitud de validación), entre 1 y `k_max`. Una clase compacta queda
con una gaussiana y una con subgrupos (por ejemplo LPV con y sin una banda) con varias.

Puntaje de anomalía (más alto = más anómalo):

- modo "max" (por defecto): -max_c log p(z | c). La estrella es normal si alguna clase conocida la
  explica bien. Es el análogo probabilístico de MCDSVDD, que usa la distancia al centro más cercano:
  aquí cada clase tiene su propia forma (covarianza) y puede tener varios centros.
- modo "mezcla": -log sum_c pi_c p(z | c), con pi_c la fracción de la clase en train. Es la
  densidad de un GMM grande cuyos componentes están amarrados a las clases. Las clases chicas
  pesan menos, así que una estrella rara de DSCT puede salir más anómala que con "max".

Con las etiquetas, el modelo también clasifica: predict devuelve la clase de máxima posterior
pi_c p(z | c) / sum_j pi_j p(z | j). Así el ARI es comparable con el de MCDSVDD, que también usa
las clases al entrenar.
"""

from __future__ import annotations

import warnings

import numpy as np

from ic_project.gmm import MezclaGaussianas, logsumexp

MODOS_PUNTAJE = ("max", "mezcla")


def n_parametros_gmm(k: int, d: int) -> int:
    """Parámetros libres de un GMM de covarianza completa: K-1 pesos, K*D medias, K*D(D+1)/2."""
    return int(k * d * (d + 1) // 2 + k * d + k - 1)


class GMMPorClase:
    """Un GMM (MezclaGaussianas) por clase conocida.

    Parámetros
    ----------
    k_max: máximo de componentes por clase. Se prueban K = 1..k_max.
    criterio: "bic" elige K_c con el BIC de train de la clase; "val" con la log-verosimilitud
        media de validación de la clase (hay que pasar z_val e y_val a `fit`).
    reg_covar, n_init, max_iter, tol: se pasan a cada MezclaGaussianas.
    random_state: semilla de k-means++ (la misma para todas las clases).

    Un K solo se prueba si la clase tiene al menos tantas estrellas como parámetros libres tiene el
    GMM (con menos, la covarianza no se puede estimar bien). K = 1 siempre se prueba.

    Atributos después de `fit`
    --------------------------
    clases_: clases en orden. modelos_: un MezclaGaussianas por clase (mismo orden).
    k_: componentes elegidos por clase (dict). seleccion_: el criterio para cada K probado (dict de
    dicts; BIC más bajo es mejor, log-verosimilitud de validación más alta es mejor).
    log_prior_: log de la fracción de cada clase en train.
    """

    def __init__(
        self,
        k_max: int = 4,
        *,
        criterio: str = "bic",
        reg_covar: float = 1e-6,
        n_init: int = 5,
        max_iter: int = 300,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        if k_max < 1:
            raise ValueError("k_max debe ser >= 1")
        if criterio not in ("bic", "val"):
            raise ValueError("criterio debe ser 'bic' o 'val'")
        self.k_max = k_max
        self.criterio = criterio
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def __repr__(self) -> str:
        return f"GMMPorClase(k_max={self.k_max}, criterio={self.criterio!r})"

    # ------------------------------------------------------------------ ajuste

    def _nuevo_gmm(self, k: int) -> MezclaGaussianas:
        return MezclaGaussianas(
            k,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        z_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> GMMPorClase:
        """Ajusta un GMM por clase con las estrellas de train de esa clase y elige su K."""
        z = _validar_z(z)
        y = np.asarray(y)
        if len(y) != len(z):
            raise ValueError("z e y deben tener el mismo número de filas")
        if self.criterio == "val":
            if z_val is None or y_val is None:
                raise ValueError("Con criterio 'val' hay que pasar z_val e y_val")
            z_val, y_val = _validar_z(z_val), np.asarray(y_val)

        self.clases_ = np.array(sorted(set(y.tolist())))
        d = z.shape[1]
        self.modelos_: list[MezclaGaussianas] = []
        self.k_: dict = {}
        self.seleccion_: dict = {}
        conteos = []
        for clase in self.clases_.tolist():
            z_c = z[y == clase]
            if len(z_c) < 2:
                raise ValueError(f"La clase {clase} tiene menos de 2 estrellas en train")
            conteos.append(len(z_c))
            candidatos = [
                k for k in range(1, self.k_max + 1) if k == 1 or len(z_c) >= n_parametros_gmm(k, d)
            ]
            z_c_val = None
            if self.criterio == "val":
                z_c_val = z_val[y_val == clase]
                if len(z_c_val) == 0:
                    raise ValueError(f"La clase {clase} no tiene estrellas en validación")
            mejor_k, mejor_valor, mejor_modelo = None, None, None
            valores = {}
            for k in candidatos:
                with warnings.catch_warnings():
                    # EM sin converger en max_iter no invalida la comparación
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        modelo = self._nuevo_gmm(k).fit(z_c)
                    except ValueError:
                        # covarianza singular: un componente quedó con muy pocas estrellas
                        continue
                if self.criterio == "bic":
                    valor = modelo.bic(z_c)
                    es_mejor = mejor_valor is None or valor < mejor_valor
                else:
                    valor = modelo.score(z_c_val)
                    es_mejor = mejor_valor is None or valor > mejor_valor
                valores[k] = float(valor)
                if es_mejor:
                    mejor_k, mejor_valor, mejor_modelo = k, valor, modelo
            if mejor_modelo is None:
                raise ValueError(f"No se pudo ajustar ningún GMM para la clase {clase}")
            self.modelos_.append(mejor_modelo)
            self.k_[clase] = mejor_k
            self.seleccion_[clase] = valores
        conteos = np.asarray(conteos, dtype=float)
        self.log_prior_ = np.log(conteos / conteos.sum())
        self.n_features_in_ = d
        return self

    # ------------------------------------------------------------- inferencia

    def log_verosimilitud_por_clase(self, z: np.ndarray) -> np.ndarray:
        """log p(z_n | c) de cada estrella con el GMM de cada clase. (N, C), columnas = clases_."""
        self._comprobar_ajustado()
        z = _validar_z(z)
        return np.column_stack([m.score_samples(z) for m in self.modelos_])

    def puntaje_anomalia(self, z: np.ndarray, modo: str = "max") -> np.ndarray:
        """Más alto = más anómalo. Ver el docstring del módulo para los dos modos."""
        if modo not in MODOS_PUNTAJE:
            raise ValueError(f"modo debe ser uno de {MODOS_PUNTAJE}")
        ll = self.log_verosimilitud_por_clase(z)
        if modo == "max":
            return -ll.max(axis=1)
        return -logsumexp(ll + self.log_prior_, axis=1)

    def score_samples(self, z: np.ndarray) -> np.ndarray:
        """log p(z) = log sum_c pi_c p(z | c), como en GaussianMixture. Más bajo = más anómalo."""
        return -self.puntaje_anomalia(z, modo="mezcla")

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Posterior de cada clase, pi_c p(z | c) / sum_j pi_j p(z | j). (N, C)."""
        lc = self.log_verosimilitud_por_clase(z) + self.log_prior_
        return np.exp(lc - logsumexp(lc, axis=1)[:, None])

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Clase de máxima posterior."""
        lc = self.log_verosimilitud_por_clase(z) + self.log_prior_
        return self.clases_[lc.argmax(axis=1)]

    def resumen(self) -> dict:
        """Componentes elegidos por clase y el criterio de cada K probado."""
        self._comprobar_ajustado()
        return {"k": dict(self.k_), "criterio": self.criterio, "seleccion": dict(self.seleccion_)}

    def _comprobar_ajustado(self) -> None:
        if not hasattr(self, "modelos_"):
            raise RuntimeError("El modelo no está ajustado: llamar primero a fit")


def _validar_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z debe ser una matriz de N estrellas por D dimensiones")
    return z
