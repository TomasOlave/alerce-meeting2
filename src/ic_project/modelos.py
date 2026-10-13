"""Autoencoder MLP y Multi-Class Deep SVDD (Pérez-Carrasco et al. 2023)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    """MLP in → 512 → 256 → 128 → z → 128 → 256 → 512 → in, como AnomalyALeRCE.

    La salida es lineal. El autoencoder del paper termina en tanh porque sus
    features quedan en [-1, 1]; aquí el escalado es z-score y no está acotado.
    """

    def __init__(self, in_dim: int, z_dim: int = 8) -> None:
        super().__init__()
        if in_dim < 1 or z_dim < 1:
            raise ValueError("in_dim y z_dim deben ser positivos")
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, z_dim, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z)


class CodificadorMCDSVDD(nn.Module):
    """Misma arquitectura de encoder que el autoencoder; centros por clase."""

    def __init__(self, in_dim: int, z_dim: int = 8, n_clases: int = 5) -> None:
        super().__init__()
        if n_clases < 2:
            raise ValueError("MCDSVDD necesita al menos dos clases inlier")
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, z_dim, bias=False),
        )
        self.register_buffer("centros", torch.zeros(n_clases, z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def cargar_encoder(self, autoencoder: Autoencoder) -> None:
        self.encoder.load_state_dict(autoencoder.encoder.state_dict())

    @torch.no_grad()
    def inicializar_centros(self, x: torch.Tensor, y: torch.Tensor, eps: float = 0.01) -> None:
        """Media del espacio latente por clase; evita ceros exactos (Ruff et al.)."""
        self.eval()
        z = self.forward(x)
        n_clases = self.centros.shape[0]
        centros = []
        for k in range(n_clases):
            mask = y == k
            if int(mask.sum()) == 0:
                raise ValueError(f"La clase {k} no tiene muestras para inicializar el centro")
            c = z[mask].mean(dim=0)
            negativo = c < 0
            pequeno = c.abs() < eps
            c = torch.where(pequeno & negativo, torch.full_like(c, -eps), c)
            c = torch.where(pequeno & ~negativo, torch.full_like(c, eps), c)
            centros.append(c)
        self.centros.copy_(torch.stack(centros))


def _tensor_x(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32))


def _tensor_y(y: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(y, dtype=np.int64))


def _loader(x: np.ndarray, y: np.ndarray | None, batch: int, shuffle: bool) -> DataLoader:
    xt = _tensor_x(x)
    ds = TensorDataset(xt) if y is None else TensorDataset(xt, _tensor_y(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@dataclass
class Historial:
    train: list[float]
    val: list[float]


def entrenar_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    z_dim: int = 8,
    n_epocas: int = 40,
    lr: float = 1e-3,
    batch: int = 128,
    paciencia: int = 8,
    semilla: int = 42,
) -> tuple[Autoencoder, Historial]:
    """Minimiza MSE de reconstrucción. Early stopping con validación."""
    torch.manual_seed(semilla)
    device = dispositivo()
    modelo = Autoencoder(x_train.shape[1], z_dim=z_dim).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-5)
    hist = Historial(train=[], val=[])
    mejor = float("inf")
    espera = 0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

    train_loader = _loader(x_train, None, batch, True)
    x_val_t = _tensor_x(x_val).to(device)

    for _ in range(n_epocas):
        modelo.train()
        losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            if xb.shape[0] < 2:
                continue
            opt.zero_grad()
            _, rec = modelo(xb)
            loss = nn.functional.mse_loss(rec, xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if not losses:
            raise RuntimeError("BatchNorm necesita al menos un batch con 2 muestras")
        modelo.eval()
        with torch.no_grad():
            _, rec_val = modelo(x_val_t)
            val_loss = float(nn.functional.mse_loss(rec_val, x_val_t).item())
        hist.train.append(float(np.mean(losses)))
        hist.val.append(val_loss)
        if val_loss < mejor - 1e-5:
            mejor = val_loss
            espera = 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    modelo.to(device)
    return modelo, hist


def entrenar_mcdsvdd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    autoencoder: Autoencoder | None = None,
    z_dim: int = 8,
    n_epocas: int = 30,
    lr: float = 1e-4,
    batch: int = 128,
    weight_decay: float = 1e-6,
    paciencia: int = 8,
    semilla: int = 42,
) -> tuple[CodificadorMCDSVDD, Historial]:
    """Loss del paper: distancia al centro de la clase etiquetada. Los centros quedan fijos."""
    torch.manual_seed(semilla)
    device = dispositivo()
    if int(np.min(y_train)) < 0:
        raise ValueError("y_train debe estar codificado desde 0")
    n_clases = int(np.max(y_train)) + 1
    modelo = CodificadorMCDSVDD(x_train.shape[1], z_dim=z_dim, n_clases=n_clases).to(device)
    if autoencoder is not None:
        modelo.cargar_encoder(autoencoder)
    x_all = _tensor_x(x_train).to(device)
    y_all = _tensor_y(y_train).to(device)
    modelo.inicializar_centros(x_all, y_all)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    hist = Historial(train=[], val=[])
    mejor = float("inf")
    espera = 0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    train_loader = _loader(x_train, y_train, batch, True)
    x_val_t = _tensor_x(x_val).to(device)
    y_val_t = _tensor_y(y_val).to(device)

    for _ in range(n_epocas):
        modelo.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.shape[0] < 2:
                continue
            opt.zero_grad()
            z = modelo(xb)
            loss = torch.mean(torch.sum((z - modelo.centros[yb]) ** 2, dim=1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if not losses:
            raise RuntimeError("BatchNorm necesita al menos un batch con 2 muestras")
        modelo.eval()
        with torch.no_grad():
            z_val = modelo(x_val_t)
            val_loss = float(
                torch.mean(torch.sum((z_val - modelo.centros[y_val_t]) ** 2, dim=1)).item()
            )
        hist.train.append(float(np.mean(losses)))
        hist.val.append(val_loss)
        if val_loss < mejor - 1e-5:
            mejor = val_loss
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
def latente(modelo: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    device = dispositivo()
    modelo.eval()
    modelo.to(device)
    outs = []
    xt = _tensor_x(x)
    for i in range(0, len(xt), batch):
        xb = xt[i : i + batch].to(device)
        z = modelo.encode(xb) if isinstance(modelo, Autoencoder) else modelo(xb)
        outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def mse_reconstruccion(modelo: Autoencoder, x: np.ndarray, batch: int = 512) -> float:
    """MSE medio de reconstrucción, en batches para no cargar todo el test de una vez."""
    device = dispositivo()
    modelo.eval()
    modelo.to(device)
    total = 0.0
    n = 0
    xt = _tensor_x(x)
    for i in range(0, len(xt), batch):
        xb = xt[i : i + batch].to(device)
        _, rec = modelo(xb)
        total += float(nn.functional.mse_loss(rec, xb, reduction="sum").item())
        n += int(xb.numel())
    if n == 0:
        raise ValueError("x está vacío")
    return total / n


@torch.no_grad()
def puntaje_mcdsvdd(modelo: CodificadorMCDSVDD, x: np.ndarray, batch: int = 512) -> np.ndarray:
    """Distancia al centro más cercano (anomaly score del paper)."""
    z = latente(modelo, x, batch=batch)
    c = modelo.centros.detach().cpu().numpy()
    d2 = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return d2.min(axis=1)


@torch.no_grad()
def asignar_mcdsvdd(modelo: CodificadorMCDSVDD, x: np.ndarray) -> np.ndarray:
    z = latente(modelo, x)
    c = modelo.centros.detach().cpu().numpy()
    d2 = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)
