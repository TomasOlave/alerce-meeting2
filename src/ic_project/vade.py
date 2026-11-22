"""VaDE: autoencoder variacional con una mezcla de gaussianas como prior (Jiang et al., IJCAI 2017).

Es el modelo con GMM que trae el repositorio del paper (AnomalyALeRCE, src/models/VaDE.py). Ese
archivo quedó a medio hacer: `compute_pcz` usa `self.VaDE`, que no existe; no tiene
`compute_metrics` ni `compute_anomaly_score`; `build_network` no lo acepta (no está en
`implemented_networks`) y no hay resultados de VaDE en el paper. Aquí está completo, con la misma
arquitectura de encoder y decoder que el resto del proyecto.

Modelo generativo, con C componentes:

    c ~ Cat(pi),    z | c ~ N(mu_c, diag(sigma2_c)),    x | z ~ decoder(z).

El encoder entrega q(z | x) = N(mu~(x), diag(sigma2~(x))). La pérdida es menos la ELBO:

    L = ||x - x'||^2                                   (reconstrucción, z muestreado)
      + 1/2 sum_c gamma_c sum_d [log sigma2_cd + (sigma2~_d + (mu~_d - mu_cd)^2) / sigma2_cd]
      - 1/2 sum_d (1 + log sigma2~_d)                  (KL de z contra el prior de su componente)
      + sum_c gamma_c (log gamma_c - log pi_c)         (KL de la asignación contra pi)

con gamma_c = q(c | x) ~ p(c | z) = pi_c N(z | mu_c, sigma2_c) / sum_j pi_j N(z | mu_j, sigma2_j).
Es lo mismo que calcula `compute_loss` del repositorio. Diferencias con ese código:

- gamma se calcula en logaritmos (softmax de log pi + log N). El repositorio hace exp de cada
  término y con 10 dimensiones da underflow (gamma = 0/0).
- pi se parametriza con un softmax, así que siempre suma 1 y es positivo. En el repositorio pi es
  un parámetro libre y el gradiente lo puede sacar del simplex.
- La reconstrucción es el error cuadrático sumado sobre features (verosimilitud gaussiana), igual
  que en DAGMM. El repositorio usa el MSE promediado sobre features, que con 140 features hace
  que la reconstrucción pese 140 veces menos que la KL.
- El decoder termina lineal: las features están en z-score, no en [-1, 1] (el repositorio usa tanh).

Inicialización, como en el paper de VaDE: el encoder y el decoder parten del autoencoder del
proyecto ya entrenado, y el prior parte de un GMM (gmm.py) ajustado sobre el latente de train.

Versión con clases (`y_train`): hay un componente por clase conocida y gamma es la clase verdadera
(one-hot) en vez de p(c | z). El prior de cada clase parte de la media y la varianza del latente
de esa clase. Es el análogo generativo de MCDSVDD: cada clase tiene un centro, pero además una
varianza aprendida por dimensión.

Puntaje de anomalía: -ELBO de cada estrella con z = mu~(x) y gamma = p(c | z), en las dos versiones
(en test no se conoce la clase). Es una cota superior de -log p(x), así que junta qué tan mal se
reconstruye la estrella y qué tan lejos cae de los componentes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ic_project.gmm import MezclaGaussianas
from ic_project.modelos import (
    Autoencoder,
    _loader,
    _tensor_x,
    _tensor_y,
    dispositivo,
    latente,
)

_LOG_2PI = math.log(2.0 * math.pi)
# límites del log de las varianzas, para que exp no se vaya a 0 ni a infinito
_LOG_VAR_MIN, _LOG_VAR_MAX = -10.0, 10.0


def log_gaussianas_diagonales(
    z: torch.Tensor, medias: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """log N(z_n | mu_c, diag(exp(log_var_c))) de cada estrella y componente. (N, C)."""
    dif2 = (z[:, None, :] - medias[None, :, :]) ** 2
    return -0.5 * (_LOG_2PI + log_var[None] + dif2 / log_var.exp()[None]).sum(dim=2)


@dataclass
class TerminosVaDE:
    """Términos de -ELBO de cada estrella (N,) y la pertenencia gamma (N, C)."""

    reconstruccion: torch.Tensor
    kl_z: torch.Tensor
    kl_c: torch.Tensor
    gamma: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.reconstruccion + self.kl_z + self.kl_c


class VaDE(nn.Module):
    """Encoder in → 512 → 256 → 128 → (mu~, log sigma2~), decoder como el del autoencoder, y el
    prior: pesos pi (softmax de `logits_pi`), medias y log-varianzas de C componentes."""

    def __init__(self, in_dim: int, z_dim: int = 10, n_componentes: int = 5) -> None:
        super().__init__()
        if n_componentes < 1:
            raise ValueError("n_componentes debe ser >= 1")
        base = Autoencoder(in_dim, z_dim=z_dim)
        # las capas del encoder del autoencoder menos la última (la que da el latente)
        self.tronco = nn.Sequential(*list(base.encoder.children())[:-1])
        self.cabeza_media = nn.Linear(128, z_dim)
        self.cabeza_log_var = nn.Linear(128, z_dim)
        self.decoder = base.decoder
        self.logits_pi = nn.Parameter(torch.zeros(n_componentes))
        self.medias_prior = nn.Parameter(torch.zeros(n_componentes, z_dim))
        self.log_var_prior = nn.Parameter(torch.zeros(n_componentes, z_dim))

    @property
    def n_componentes(self) -> int:
        return int(self.logits_pi.shape[0])

    def log_pi(self) -> torch.Tensor:
        return torch.log_softmax(self.logits_pi, dim=0)

    def _log_var_prior(self) -> torch.Tensor:
        return self.log_var_prior.clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tronco(x)
        log_var = self.cabeza_log_var(h).clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return self.cabeza_media(h), log_var

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """mu~, log sigma2~, z y la reconstrucción. En entrenamiento z se muestrea
        (reparametrización); en evaluación z = mu~."""
        media, log_var = self.encode(x)
        ruido = torch.randn_like(media) * torch.exp(0.5 * log_var)
        z = media + ruido if self.training else media
        return media, log_var, z, self.decoder(z)

    def log_conjunta(self, z: torch.Tensor) -> torch.Tensor:
        """log(pi_c N(z | mu_c, sigma2_c)) de cada estrella y componente. (N, C)."""
        log_gauss = log_gaussianas_diagonales(z, self.medias_prior, self._log_var_prior())
        return log_gauss + self.log_pi()

    def terminos(self, x: torch.Tensor, y: torch.Tensor | None = None) -> TerminosVaDE:
        """Términos de -ELBO por estrella. Con `y`, gamma es la clase verdadera (versión con
        clases); sin `y`, gamma = p(c | z)."""
        media, log_var, z, x_rec = self(x)
        reconstruccion = ((x - x_rec) ** 2).sum(dim=1)
        log_pi = self.log_pi()
        if y is None:
            gamma = torch.softmax(self.log_conjunta(z), dim=1)
            kl_c = (gamma * (torch.log(gamma + 1e-10) - log_pi[None])).sum(dim=1)
        else:
            gamma = nn.functional.one_hot(y, self.n_componentes).to(media.dtype)
            kl_c = -log_pi[y]
        log_var_p = self._log_var_prior()
        var_p = log_var_p.exp()
        por_componente = (
            log_var_p[None]
            + (log_var.exp()[:, None, :] + (media[:, None, :] - self.medias_prior[None]) ** 2)
            / var_p[None]
        ).sum(dim=2)
        kl_z = 0.5 * (gamma * por_componente).sum(dim=1) - 0.5 * (1 + log_var).sum(dim=1)
        return TerminosVaDE(reconstruccion, kl_z, kl_c, gamma)

    @torch.no_grad()
    def inicializar(
        self,
        autoencoder: Autoencoder,
        x_train: np.ndarray,
        y_train: np.ndarray | None = None,
        semilla: int = 42,
    ) -> None:
        """Copia los pesos del autoencoder y pone el prior en un GMM sobre su latente de train.

        Sin `y_train`, el prior es un GMM de `n_componentes` (gmm.py), del que se toma la diagonal
        de cada covarianza. Con `y_train` (enteros 0..C-1), media y varianza del latente por clase.
        La varianza del encoder parte en el 1% de la varianza del latente en cada dimensión.
        """
        estado = autoencoder.encoder.state_dict()
        ultima = str(len(autoencoder.encoder) - 1)
        self.tronco.load_state_dict(
            {k: v for k, v in estado.items() if not k.startswith(ultima + ".")}
        )
        self.cabeza_media.weight.copy_(autoencoder.encoder[-1].weight)
        self.cabeza_media.bias.zero_()
        self.decoder.load_state_dict(autoencoder.decoder.state_dict())

        z = latente(autoencoder, x_train).astype(np.float64)
        var_latente = z.var(axis=0) + 1e-8
        self.cabeza_log_var.weight.zero_()
        self.cabeza_log_var.bias.copy_(torch.as_tensor(np.log(0.01 * var_latente)))

        if y_train is None:
            gmm = MezclaGaussianas(self.n_componentes, n_init=5, random_state=semilla).fit(z)
            pesos = gmm.weights_
            medias = gmm.means_
            varianzas = np.diagonal(gmm.covariances_, axis1=1, axis2=2)
        else:
            y_train = np.asarray(y_train)
            if set(np.unique(y_train)) != set(range(self.n_componentes)):
                raise ValueError("y_train debe tener las clases 0..n_componentes-1")
            pesos = np.bincount(y_train, minlength=self.n_componentes) / len(y_train)
            medias = np.stack([z[y_train == c].mean(axis=0) for c in range(self.n_componentes)])
            varianzas = np.stack([z[y_train == c].var(axis=0) for c in range(self.n_componentes)])
        dtype = self.logits_pi.dtype
        self.logits_pi.copy_(torch.as_tensor(np.log(pesos), dtype=dtype))
        self.medias_prior.copy_(torch.as_tensor(medias, dtype=dtype))
        self.log_var_prior.copy_(torch.as_tensor(np.log(varianzas + 1e-6), dtype=dtype))


@dataclass
class HistorialVaDE:
    train: list[float]
    val: list[float]
    reconstruccion_val: list[float]
    kl_val: list[float]


def entrenar_vade(
    x_train: np.ndarray,
    x_val: np.ndarray,
    autoencoder: Autoencoder,
    n_componentes: int = 5,
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    z_dim: int = 10,
    n_epocas: int = 300,
    lr: float = 5e-4,
    batch: int = 256,
    paciencia: int = 30,
    semilla: int = 42,
) -> tuple[VaDE, HistorialVaDE]:
    """Entrena VaDE desde el autoencoder ya entrenado. Early stopping con -ELBO de validación.

    Con `y_train` e `y_val` (enteros desde 0) es la versión con clases: un componente por clase
    y `n_componentes` se ignora. `n_epocas` es solo un tope.
    """
    torch.manual_seed(semilla)
    device = dispositivo()
    con_clases = y_train is not None
    if con_clases:
        if y_val is None:
            raise ValueError("La versión con clases necesita también y_val")
        n_componentes = int(np.max(y_train)) + 1
    modelo = VaDE(x_train.shape[1], z_dim=z_dim, n_componentes=n_componentes)
    modelo.inicializar(autoencoder, x_train, y_train, semilla=semilla)
    modelo.to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    loader = _loader(x_train, y_train, batch, True)
    x_val_t = _tensor_x(x_val).to(device)
    y_val_t = _tensor_y(y_val).to(device) if con_clases else None

    hist = HistorialVaDE(train=[], val=[], reconstruccion_val=[], kl_val=[])
    mejor = float("inf")
    espera = 0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    for _ in range(n_epocas):
        modelo.train()
        perdidas = []
        for lote in loader:
            xb = lote[0].to(device)
            yb = lote[1].to(device) if con_clases else None
            if xb.shape[0] < 2:
                continue
            opt.zero_grad()
            perdida = modelo.terminos(xb, yb).total.mean()
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            perdidas.append(perdida.item())
        if not perdidas:
            raise RuntimeError("BatchNorm necesita al menos un batch con 2 muestras")
        modelo.eval()
        with torch.no_grad():
            t = modelo.terminos(x_val_t, y_val_t)
        hist.train.append(float(np.mean(perdidas)))
        hist.val.append(float(t.total.mean()))
        hist.reconstruccion_val.append(float(t.reconstruccion.mean()))
        hist.kl_val.append(float((t.kl_z + t.kl_c).mean()))
        if not np.isfinite(hist.val[-1]):
            break
        if hist.val[-1] < mejor - 1e-4:
            mejor = hist.val[-1]
            espera = 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    modelo.to(device)
    return modelo, hist


@torch.no_grad()
def salidas_vade(modelo: VaDE, x: np.ndarray, batch: int = 1024) -> dict[str, np.ndarray]:
    """Para cada estrella: latente mu~, pertenencia gamma = p(c | mu~), componente más probable,
    -ELBO (puntaje de anomalía) y -log p(mu~) bajo el prior (solo la parte del latente)."""
    device = dispositivo()
    modelo.eval()
    modelo.to(device)
    xt = _tensor_x(x)
    medias, gammas, elbo, energia = [], [], [], []
    for i in range(0, len(xt), batch):
        xb = xt[i : i + batch].to(device)
        t = modelo.terminos(xb)  # en evaluación z = mu~ y gamma = p(c | z)
        media, _ = modelo.encode(xb)
        medias.append(media.cpu())
        gammas.append(t.gamma.cpu())
        elbo.append(t.total.cpu())
        energia.append((-torch.logsumexp(modelo.log_conjunta(media), dim=1)).cpu())
    gamma = torch.cat(gammas).numpy()
    return {
        "z": torch.cat(medias).numpy(),
        "gamma": gamma,
        "componente": gamma.argmax(axis=1),
        "menos_elbo": torch.cat(elbo).numpy(),
        "energia_latente": torch.cat(energia).numpy(),
    }
