"""DAGMM: autoencoder y mezcla de gaussianas entrenados juntos (Zong et al., ICLR 2018).

En el pipeline del meeting 2 el autoencoder se entrena solo para reconstruir y el GMM se ajusta
después sobre su latente. El autoencoder nunca sabe que después se va a agrupar, así que nada obliga
al latente a formar grupos. DAGMM entrena las dos cosas a la vez, con una sola pérdida:

    J = (1/N) sum_n ||x_n - x'_n||^2 + lambda_energia (1/N) sum_n E(z_n) + lambda_cov P(Sigma)

Partes del modelo:

1. Red de compresión: el mismo autoencoder del proyecto. Entrega el latente z_c y la
   reconstrucción x'. A z_c se le agregan dos números que miden qué tan mal se reconstruyó el objeto
   (distancia euclidiana relativa y similitud coseno entre x y x'): z = [z_c, z_r].
2. Red de estimación: un MLP chico que mira z y entrega la pertenencia gamma_nk de cada objeto a
   cada uno de los K componentes (softmax). Hace el papel del paso E de EM.
3. Mezcla de gaussianas: con gamma se calculan pesos, medias y covarianzas con las fórmulas del
   paso M (las mismas de gmm.py). Con ellos, la energía de cada objeto es
       E(z) = -log sum_k phi_k N(z | mu_k, Sigma_k),
   o sea, menos la log-verosimilitud del GMM. Energía baja = el objeto cae en un grupo denso.

Al minimizar J por gradiente, el encoder aprende un latente que se reconstruye bien y que además
se deja describir por K gaussianas compactas. P(Sigma) = sum_k sum_j 1 / Sigma_kjj castiga las
varianzas que tienden a cero, que llevarían la energía a -infinito (covarianza singular).

A diferencia de gmm.py, aquí no hay iteraciones de EM: la red de estimación reemplaza al paso E y
los parámetros salen de fórmulas derivables, para que el gradiente llegue hasta el encoder. Lo de la
mezcla se calcula en float64 para que Cholesky no falle por redondeo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ic_project.modelos import Autoencoder, _loader, _tensor_x, dispositivo

# ------------------------------------------------------------------ mezcla de gaussianas en torch


def caracteristicas_reconstruccion(
    x: torch.Tensor, x_rec: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Distancia euclidiana relativa y similitud coseno entre x y su reconstrucción. (N, 2)."""
    norma = torch.linalg.vector_norm(x, dim=1) + eps
    distancia_relativa = torch.linalg.vector_norm(x - x_rec, dim=1) / norma
    coseno = nn.functional.cosine_similarity(x, x_rec, dim=1, eps=eps)
    return torch.stack([distancia_relativa, coseno], dim=1)


def parametros_mezcla(
    z: torch.Tensor, gamma: torch.Tensor, reg_covar: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pesos (K,), medias (K, D) y covarianzas (K, D, D) a partir de la pertenencia gamma (N, K).

    Son las fórmulas del paso M de gmm.py, escritas en torch para poder derivarlas.
    """
    z, gamma = z.double(), gamma.double()
    n_k = gamma.sum(dim=0) + 10 * torch.finfo(torch.float64).eps
    phi = n_k / n_k.sum()
    mu = gamma.T @ z / n_k[:, None]
    dif = z[:, None, :] - mu[None, :, :]
    sigma = torch.einsum("nk,nkd,nke->kde", gamma, dif, dif) / n_k[:, None, None]
    sigma = sigma + reg_covar * torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    return phi, mu, sigma


def log_conjunta(
    z: torch.Tensor, phi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """log(phi_k N(z_n | mu_k, Sigma_k)) de cada objeto y componente. (N, K)."""
    z = z.double()
    d = z.shape[1]
    chol = torch.linalg.cholesky(sigma)
    dif = (z[:, None, :] - mu[None, :, :]).unsqueeze(-1)
    # con Sigma = L L^T, resolver L y = (z - mu) da ||y||^2 = distancia de Mahalanobis al cuadrado
    y = torch.linalg.solve_triangular(chol.unsqueeze(0), dif, upper=False).squeeze(-1)
    mahalanobis2 = (y**2).sum(dim=-1)
    log_det = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    log_gauss = -0.5 * (d * math.log(2 * math.pi) + log_det + mahalanobis2)
    return log_gauss + torch.log(phi)


def energia(
    z: torch.Tensor, phi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """E(z) = -log sum_k phi_k N(z | mu_k, Sigma_k). Es lo mismo que -score_samples de gmm.py."""
    return -torch.logsumexp(log_conjunta(z, phi, mu, sigma), dim=1)


def penalizacion_covarianza(sigma: torch.Tensor) -> torch.Tensor:
    """P(Sigma) = suma de 1 / varianza sobre todas las diagonales. Evita covarianzas singulares."""
    return (1.0 / torch.diagonal(sigma, dim1=-2, dim2=-1)).sum()


# ------------------------------------------------------------------------------------- modelo


class DAGMM(nn.Module):
    """Red de compresión (autoencoder) + red de estimación + mezcla de K gaussianas."""

    def __init__(
        self,
        in_dim: int,
        z_dim: int = 8,
        n_componentes: int = 5,
        oculta: int = 10,
        dropout: float = 0.5,
        reg_covar: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_componentes < 1:
            raise ValueError("n_componentes debe ser >= 1")
        self.compresion = Autoencoder(in_dim, z_dim=z_dim)
        self.estimacion = nn.Sequential(
            nn.Linear(z_dim + 2, oculta),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(oculta, n_componentes),
        )
        self.reg_covar = reg_covar
        # parámetros de la mezcla calculados con todo train (ajustar_mezcla); se usan en inferencia
        d = z_dim + 2
        self.register_buffer("phi", torch.full((n_componentes,), 1.0 / n_componentes).double())
        self.register_buffer("mu", torch.zeros(n_componentes, d, dtype=torch.float64))
        self.register_buffer("sigma", torch.eye(d, dtype=torch.float64).repeat(n_componentes, 1, 1))

    @property
    def n_componentes(self) -> int:
        return int(self.phi.shape[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve el latente z = [z_c, z_r], la reconstrucción y la pertenencia gamma."""
        z_c, x_rec = self.compresion(x)
        z = torch.cat([z_c, caracteristicas_reconstruccion(x, x_rec)], dim=1)
        gamma = torch.softmax(self.estimacion(z), dim=1)
        return z, x_rec, gamma

    @torch.no_grad()
    def ajustar_mezcla(self, x: torch.Tensor, batch: int = 1024) -> None:
        """Calcula phi, mu y Sigma con todo el conjunto `x` (train) y los guarda en el modelo."""
        self.eval()
        zs, gammas = [], []
        for i in range(0, len(x), batch):
            z, _, gamma = self(x[i : i + batch])
            zs.append(z)
            gammas.append(gamma)
        phi, mu, sigma = parametros_mezcla(torch.cat(zs), torch.cat(gammas), self.reg_covar)
        self.phi.copy_(phi)
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)


def _perdida(
    modelo: DAGMM,
    x: torch.Tensor,
    lambda_energia: float,
    lambda_cov: float,
    mezcla: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pérdida total, término de reconstrucción y energía media.

    Sin `mezcla`, los parámetros del GMM salen del mismo batch (entrenamiento, como en el paper).
    Con `mezcla`, se usan los parámetros guardados de train (validación e inferencia).
    """
    z, x_rec, gamma = modelo(x)
    reconstruccion = ((x - x_rec) ** 2).sum(dim=1).mean()
    if mezcla is None:
        mezcla = parametros_mezcla(z, gamma, modelo.reg_covar)
    e = energia(z, *mezcla).mean()
    total = reconstruccion + lambda_energia * e + lambda_cov * penalizacion_covarianza(mezcla[2])
    return total, reconstruccion, e


@dataclass
class HistorialDAGMM:
    train: list[float]
    val: list[float]
    reconstruccion_val: list[float]
    energia_val: list[float]


def entrenar_dagmm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    n_componentes: int = 5,
    z_dim: int = 8,
    autoencoder: Autoencoder | None = None,
    n_epocas: int = 60,
    lr: float = 1e-4,
    batch: int = 512,
    lambda_energia: float = 0.1,
    lambda_cov: float = 0.005,
    reg_covar: float = 1e-6,
    paciencia: int = 10,
    semilla: int = 42,
) -> tuple[DAGMM, HistorialDAGMM]:
    """Entrena DAGMM de punta a punta. Early stopping con la pérdida de validación.

    `autoencoder`: si se pasa, la red de compresión parte de sus pesos (como MCDSVDD) en vez de
    partir al azar. lambda_energia y lambda_cov son los del paper (0.1 y 0.005).
    En validación la energía se mide con la mezcla ajustada en train, igual que en inferencia.
    """
    torch.manual_seed(semilla)
    device = dispositivo()
    modelo = DAGMM(x_train.shape[1], z_dim, n_componentes, reg_covar=reg_covar).to(device)
    if autoencoder is not None:
        modelo.compresion.load_state_dict(autoencoder.state_dict())
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    loader = _loader(x_train, None, batch, True)
    x_train_t = _tensor_x(x_train).to(device)
    x_val_t = _tensor_x(x_val).to(device)

    hist = HistorialDAGMM(train=[], val=[], reconstruccion_val=[], energia_val=[])
    mejor = float("inf")
    espera = 0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    for _ in range(n_epocas):
        modelo.train()
        perdidas = []
        for (xb,) in loader:
            xb = xb.to(device)
            if xb.shape[0] < 2:
                continue
            opt.zero_grad()
            total, _, _ = _perdida(modelo, xb, lambda_energia, lambda_cov)
            total.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            perdidas.append(total.item())
        if not perdidas:
            raise RuntimeError("BatchNorm necesita al menos un batch con 2 muestras")

        modelo.ajustar_mezcla(x_train_t)
        with torch.no_grad():
            mezcla = (modelo.phi, modelo.mu, modelo.sigma)
            total_val, rec_val, e_val = _perdida(
                modelo, x_val_t, lambda_energia, lambda_cov, mezcla=mezcla
            )
        hist.train.append(float(np.mean(perdidas)))
        hist.val.append(float(total_val))
        hist.reconstruccion_val.append(float(rec_val))
        hist.energia_val.append(float(e_val))
        if hist.val[-1] < mejor - 1e-5:
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


# --------------------------------------------------------------------------------- inferencia


@torch.no_grad()
def salidas_dagmm(modelo: DAGMM, x: np.ndarray, batch: int = 1024) -> dict[str, np.ndarray]:
    """Para cada objeto: latente z, pertenencia gamma (red de estimación), probabilidad de cada
    componente según la mezcla ajustada en train, y energía (puntaje de anomalía)."""
    device = dispositivo()
    modelo.eval()
    modelo.to(device)
    xt = _tensor_x(x)
    zs, gammas = [], []
    for i in range(0, len(xt), batch):
        z, _, gamma = modelo(xt[i : i + batch].to(device))
        zs.append(z)
        gammas.append(gamma)
    z, gamma = torch.cat(zs), torch.cat(gammas)
    lc = log_conjunta(z, modelo.phi, modelo.mu, modelo.sigma)
    return {
        "z": z.cpu().numpy(),
        "gamma": gamma.cpu().numpy(),
        "probabilidad": torch.softmax(lc, dim=1).cpu().numpy(),
        "energia": (-torch.logsumexp(lc, dim=1)).cpu().numpy(),
    }


def asignar_dagmm(modelo: DAGMM, x: np.ndarray) -> np.ndarray:
    """Componente de cada objeto: el de mayor pertenencia según la red de estimación."""
    return salidas_dagmm(modelo, x)["gamma"].argmax(axis=1)
