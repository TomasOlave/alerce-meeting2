"""Mezcla de gaussianas (GMM) con covarianza completa, ajustada con EM. Solo usa NumPy.

Reemplaza a ``sklearn.mixture.GaussianMixture(covariance_type="full")``. Conserva los nombres de
sus parámetros, atributos y métodos (``fit``, ``predict``, ``predict_proba``, ``score_samples``,
``bic``, ``means_``, ``precisions_``...) para que el notebook y ``cualitativo.py`` la usen igual.

Notación: N objetos x_n de dimensión D y K componentes. El componente k tiene peso pi_k, media
mu_k y covarianza Sigma_k. La densidad del modelo es

    p(x) = sum_k pi_k N(x | mu_k, Sigma_k).

El algoritmo EM parte de unos parámetros iniciales (k-means++) y alterna dos pasos hasta que la
log-verosimilitud media deja de subir:

- Paso E: responsabilidad de cada componente sobre cada objeto,
      gamma_nk = pi_k N(x_n | mu_k, Sigma_k) / sum_j pi_j N(x_n | mu_j, Sigma_j).
- Paso M: con N_k = sum_n gamma_nk,
      pi_k = N_k / N,
      mu_k = sum_n gamma_nk x_n / N_k,
      Sigma_k = sum_n gamma_nk (x_n - mu_k)(x_n - mu_k)^T / N_k + reg_covar * I.

Las densidades se calculan en logaritmos (con logsumexp) para que no haya underflow cuando son muy
chicas. Sigma_k no se invierte directamente: con su Cholesky Sigma_k = L L^T se usa P = L^{-T},
que cumple P P^T = Sigma_k^{-1}. Así la distancia de Mahalanobis es ||(x - mu_k)^T P||^2 y
log |Sigma_k|^{-1/2} es la suma del logaritmo de la diagonal de P.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """log(sum(exp(a))) en el eje `axis`. Resta antes el máximo para no tener underflow."""
    maximo = np.max(a, axis=axis, keepdims=True)
    maximo = np.where(np.isfinite(maximo), maximo, 0.0)
    with np.errstate(divide="ignore"):
        suma = np.log(np.sum(np.exp(a - maximo), axis=axis, keepdims=True))
    return np.squeeze(suma + maximo, axis=axis)


def distancias_cuadradas(x: np.ndarray, centros: np.ndarray) -> np.ndarray:
    """Distancia euclidiana al cuadrado de cada fila de `x` (N, D) a cada centro (K, D)."""
    d2 = (x**2).sum(axis=1)[:, None] - 2.0 * x @ centros.T + (centros**2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


def kmeans(
    x: np.ndarray, n_clusters: int, rng: np.random.Generator, max_iter: int = 300
) -> np.ndarray:
    """K-means con semillas k-means++. Devuelve el cluster (0..K-1) de cada fila de `x`.

    k-means++ elige el primer centro al azar entre los objetos, y cada uno de los siguientes con
    probabilidad proporcional a la distancia al cuadrado al centro más cercano ya elegido. Después
    se itera Lloyd (asignar cada objeto a su centro más cercano y recalcular los centros como la
    media de sus objetos) hasta que las asignaciones no cambian.
    """
    n = len(x)
    centros = np.empty((n_clusters, x.shape[1]))
    centros[0] = x[rng.integers(n)]
    d2_min = ((x - centros[0]) ** 2).sum(axis=1)
    for j in range(1, n_clusters):
        total = d2_min.sum()
        i = rng.choice(n, p=d2_min / total) if total > 0 else rng.integers(n)
        centros[j] = x[i]
        d2_min = np.minimum(d2_min, ((x - centros[j]) ** 2).sum(axis=1))

    etiquetas = np.full(n, -1)
    for _ in range(max_iter):
        d2 = distancias_cuadradas(x, centros)
        nuevas = d2.argmin(axis=1)
        if np.array_equal(nuevas, etiquetas):
            break
        etiquetas = nuevas
        for j in range(n_clusters):
            miembros = etiquetas == j
            if miembros.any():
                centros[j] = x[miembros].mean(axis=0)
            else:
                # cluster vacío: se queda con el objeto que está más lejos de su centro
                lejano = int(d2[np.arange(n), etiquetas].argmax())
                etiquetas[lejano] = j
                centros[j] = x[lejano]
    return etiquetas


def _paso_m(
    x: np.ndarray, resp: np.ndarray, reg_covar: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paso M: pesos (K,), medias (K, D) y covarianzas (K, D, D) desde las responsabilidades."""
    # el 10 * eps evita dividir por cero si un componente se queda sin objetos
    n_k = resp.sum(axis=0) + 10 * np.finfo(float).eps
    medias = resp.T @ x / n_k[:, None]
    n_comp, d = medias.shape
    covarianzas = np.empty((n_comp, d, d))
    for k in range(n_comp):
        dif = x - medias[k]
        covarianzas[k] = (resp[:, k, None] * dif).T @ dif / n_k[k]
        covarianzas[k].flat[:: d + 1] += reg_covar  # suma reg_covar a la diagonal
    return n_k / n_k.sum(), medias, covarianzas


def _cholesky_precisiones(covarianzas: np.ndarray) -> np.ndarray:
    """Para cada Sigma_k = L L^T devuelve P_k = L^{-T}, que cumple P_k P_k^T = Sigma_k^{-1}."""
    n_comp, d, _ = covarianzas.shape
    identidad = np.eye(d)
    salida = np.empty_like(covarianzas)
    for k in range(n_comp):
        try:
            chol = np.linalg.cholesky(covarianzas[k])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"La covarianza del componente {k} no es definida positiva. Pasa cuando un "
                "componente se queda con muy pocos objetos: probar con más reg_covar o con "
                "menos componentes."
            ) from error
        salida[k] = np.linalg.solve(chol, identidad).T
    return salida


def _log_gaussianas(x: np.ndarray, medias: np.ndarray, prec_chol: np.ndarray) -> np.ndarray:
    """log N(x_n | mu_k, Sigma_k) de cada objeto en cada componente. Devuelve (N, K)."""
    d = x.shape[1]
    # P_k es triangular: log det P_k = log |Sigma_k|^{-1/2} es la suma del log de su diagonal
    log_det = np.log(np.diagonal(prec_chol, axis1=1, axis2=2)).sum(axis=1)
    # y_nk = (x_n - mu_k)^T P_k, y ||y_nk||^2 es la distancia de Mahalanobis al cuadrado
    y = np.einsum("nd,kde->nke", x, prec_chol) - np.einsum("kd,kde->ke", medias, prec_chol)
    mahalanobis2 = (y**2).sum(axis=2)
    return -0.5 * (d * _LOG_2PI + mahalanobis2) + log_det


@dataclass
class _Corrida:
    """Resultado de una inicialización de EM."""

    parametros: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    cota: float
    historial: list[float]
    convergio: bool


class MezclaGaussianas:
    """GMM de covarianza completa ajustado con EM. Misma interfaz que sklearn GaussianMixture.

    Parámetros
    ----------
    n_components: número de componentes K.
    covariance_type: solo se acepta "full" (covarianza completa). Está para que las llamadas
        escritas para scikit-learn sigan funcionando.
    tol: EM para cuando la log-verosimilitud media cambia menos que esto entre dos iteraciones.
    reg_covar: se suma a la diagonal de cada covarianza para que siga siendo invertible.
    max_iter: máximo de iteraciones de EM en cada inicialización.
    n_init: cuántas inicializaciones de k-means++ probar. Se queda con la de mayor
        log-verosimilitud en train.
    random_state: semilla de k-means++.
    weights_init, means_init, precisions_init: parámetros iniciales opcionales. Los que se den
        reemplazan a los que salen de k-means. Sirven para comparar con otra implementación
        partiendo del mismo punto.

    Atributos después de `fit`
    --------------------------
    weights_ (K,), means_ (K, D), covariances_ (K, D, D), precisions_ (K, D, D) y
    precisions_cholesky_ (K, D, D): parámetros de la mejor inicialización.
    lower_bound_: log-verosimilitud media de train en la última iteración.
    historial_: esa misma cantidad en cada iteración de la mejor inicialización. En EM nunca baja.
    n_iter_: iteraciones de la mejor inicialización.
    converged_: si la mejor inicialización alcanzó `tol` antes de `max_iter`.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 300,
        n_init: int = 1,
        random_state: int | None = None,
        weights_init: np.ndarray | None = None,
        means_init: np.ndarray | None = None,
        precisions_init: np.ndarray | None = None,
    ) -> None:
        if covariance_type != "full":
            raise ValueError("Solo está implementada la covarianza completa ('full')")
        if n_components < 1:
            raise ValueError("n_components debe ser >= 1")
        if n_init < 1 or max_iter < 1:
            raise ValueError("n_init y max_iter deben ser >= 1")
        if tol < 0 or reg_covar < 0:
            raise ValueError("tol y reg_covar no pueden ser negativos")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def __repr__(self) -> str:
        return (
            f"MezclaGaussianas(n_components={self.n_components}, n_init={self.n_init}, "
            f"random_state={self.random_state})"
        )

    # ------------------------------------------------------------------ ajuste

    def fit(self, X: np.ndarray) -> MezclaGaussianas:
        """Ajusta el modelo con EM. Corre `n_init` inicializaciones y se queda con la mejor."""
        x = self._validar(X, ajustando=True)
        rng = np.random.default_rng(self.random_state)
        mejor: _Corrida | None = None
        for _ in range(self.n_init):
            corrida = self._ajustar_una_vez(x, rng)
            if mejor is None or corrida.cota > mejor.cota:
                mejor = corrida
        assert mejor is not None

        self.weights_, self.means_, self.covariances_, self.precisions_cholesky_ = mejor.parametros
        self.precisions_ = self.precisions_cholesky_ @ self.precisions_cholesky_.transpose(0, 2, 1)
        self.lower_bound_ = mejor.cota
        self.historial_ = np.asarray(mejor.historial)
        self.n_iter_ = len(mejor.historial)
        self.converged_ = mejor.convergio
        self.n_features_in_ = x.shape[1]
        if not self.converged_:
            warnings.warn(
                f"EM no convergió en {self.max_iter} iteraciones. Probar con más max_iter o con "
                "una tol más grande.",
                stacklevel=2,
            )
        return self

    def _inicializar(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parámetros de partida: k-means++ y un paso M con las asignaciones duras de k-means."""
        if self.weights_init is None or self.means_init is None or self.precisions_init is None:
            etiquetas = kmeans(x, self.n_components, rng)
            resp = np.zeros((len(x), self.n_components))
            resp[np.arange(len(x)), etiquetas] = 1.0
            pesos, medias, covarianzas = _paso_m(x, resp, self.reg_covar)
            prec_chol = _cholesky_precisiones(covarianzas)
        if self.weights_init is not None:
            pesos = np.asarray(self.weights_init, dtype=float)
        if self.means_init is not None:
            medias = np.asarray(self.means_init, dtype=float)
        if self.precisions_init is not None:
            # cualquier P con P P^T = precisión sirve; la de Cholesky inferior también es triangular
            prec_chol = np.linalg.cholesky(np.asarray(self.precisions_init, dtype=float))
        return pesos, medias, prec_chol

    def _ajustar_una_vez(self, x: np.ndarray, rng: np.random.Generator) -> _Corrida:
        """Una inicialización completa: parte de k-means++ y alterna pasos E y M."""
        pesos, medias, prec_chol = self._inicializar(x, rng)
        cota = -np.inf
        historial: list[float] = []
        convergio = False
        for _ in range(self.max_iter):
            anterior = cota
            # paso E: log(pi_k N(x_n | k)), log p(x_n) y responsabilidades gamma_nk
            log_conjunta = _log_gaussianas(x, medias, prec_chol) + np.log(pesos)
            log_densidad = logsumexp(log_conjunta, axis=1)
            resp = np.exp(log_conjunta - log_densidad[:, None])
            # paso M: nuevos pesos, medias y covarianzas
            pesos, medias, covarianzas = _paso_m(x, resp, self.reg_covar)
            prec_chol = _cholesky_precisiones(covarianzas)
            # criterio de convergencia: cambio de la log-verosimilitud media (la del paso E)
            cota = float(log_densidad.mean())
            historial.append(cota)
            if abs(cota - anterior) < self.tol:
                convergio = True
                break
        return _Corrida((pesos, medias, covarianzas, prec_chol), cota, historial, convergio)

    # ------------------------------------------------------------- inferencia

    def _log_conjunta(self, X: np.ndarray) -> np.ndarray:
        """log(pi_k N(x_n | mu_k, Sigma_k)) para cada objeto y componente. (N, K)."""
        self._comprobar_ajustado()
        x = self._validar(X)
        return _log_gaussianas(x, self.means_, self.precisions_cholesky_) + np.log(self.weights_)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log-verosimilitud log p(x_n) de cada objeto. Más baja = más anómalo."""
        return logsumexp(self._log_conjunta(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Log-verosimilitud media."""
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidad de pertenencia a cada componente (responsabilidades). (N, K)."""
        log_conjunta = self._log_conjunta(X)
        return np.exp(log_conjunta - logsumexp(log_conjunta, axis=1)[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Componente más probable de cada objeto."""
        return self._log_conjunta(X).argmax(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    # -------------------------------------------------------- selección de K

    def _n_parametros(self) -> int:
        """K-1 pesos libres, K*D medias y K*D*(D+1)/2 entradas de covarianza."""
        k, d = self.means_.shape
        return int(k * d * (d + 1) // 2 + k * d + k - 1)

    def bic(self, X: np.ndarray) -> float:
        """Criterio de información bayesiano. Más bajo es mejor."""
        x = self._validar(X)
        return float(-2 * self.score(x) * len(x) + self._n_parametros() * np.log(len(x)))

    def aic(self, X: np.ndarray) -> float:
        """Criterio de información de Akaike. Más bajo es mejor."""
        x = self._validar(X)
        return float(-2 * self.score(x) * len(x) + 2 * self._n_parametros())

    # ------------------------------------------------------------- auxiliares

    def _comprobar_ajustado(self) -> None:
        if not hasattr(self, "means_"):
            raise RuntimeError("El modelo no está ajustado: llamar primero a fit")

    def _validar(self, X: np.ndarray, ajustando: bool = False) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X debe ser una matriz de N objetos por D dimensiones")
        if not np.isfinite(x).all():
            raise ValueError("X tiene NaN o infinitos")
        if ajustando and len(x) < self.n_components:
            raise ValueError(
                f"Hay {len(x)} objetos y {self.n_components} componentes: se necesitan al menos "
                "tantos objetos como componentes"
            )
        if not ajustando and x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tiene {x.shape[1]} dimensiones y el modelo se ajustó con {self.n_features_in_}"
            )
        return x
