"""Autoencoder, MCDSVDD y VaDE en NumPy (backprop a mano), con la misma arquitectura e
hiperparámetros que src/ic_project/modelos.py y vade.py. Para correr sin PyTorch."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

DT = np.float32


class Lineal:
    def __init__(self, n_in, n_out, rng, bias=True):
        k = 1.0 / np.sqrt(n_in)
        self.W = rng.uniform(-k, k, size=(n_in, n_out)).astype(DT)
        self.b = rng.uniform(-k, k, size=n_out).astype(DT) if bias else None
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b) if bias else None

    def forward(self, x, train):
        self.x = x
        y = x @ self.W
        return y + self.b if self.b is not None else y

    def backward(self, g):
        self.gW[...] = self.x.T @ g
        if self.b is not None:
            self.gb[...] = g.sum(axis=0)
        return g @ self.W.T

    def params(self):
        out = [(self.W, self.gW)]
        if self.b is not None:
            out.append((self.b, self.gb))
        return out


class BN:
    def __init__(self, n, eps=1e-5, momentum=0.1):
        self.g = np.ones(n, DT); self.b = np.zeros(n, DT)
        self.gg = np.zeros(n, DT); self.gbb = np.zeros(n, DT)
        self.rm = np.zeros(n, DT); self.rv = np.ones(n, DT)
        self.eps = eps; self.mom = momentum

    def forward(self, x, train):
        if train:
            mu = x.mean(axis=0); var = x.var(axis=0)
            n = x.shape[0]
            self.rm = (1 - self.mom) * self.rm + self.mom * mu
            self.rv = (1 - self.mom) * self.rv + self.mom * var * n / max(n - 1, 1)
        else:
            mu, var = self.rm, self.rv
        self.inv = 1.0 / np.sqrt(var + self.eps)
        self.xh = (x - mu) * self.inv
        self.train = train
        return self.xh * self.g + self.b

    def backward(self, gy):
        self.gg[...] = (gy * self.xh).sum(axis=0)
        self.gbb[...] = gy.sum(axis=0)
        gxh = gy * self.g
        if not self.train:
            return gxh * self.inv
        n = gy.shape[0]
        return (self.inv / n) * (n * gxh - gxh.sum(axis=0) - self.xh * (gxh * self.xh).sum(axis=0))

    def params(self):
        return [(self.g, self.gg), (self.b, self.gbb)]


class LReLU:
    def forward(self, x, train):
        self.m = x > 0
        return np.where(self.m, x, 0.01 * x)

    def backward(self, g):
        return np.where(self.m, g, 0.01 * g)

    def params(self):
        return []


class Seq:
    def __init__(self, capas):
        self.capas = capas

    def forward(self, x, train):
        for c in self.capas:
            x = c.forward(x, train)
        return x

    def backward(self, g):
        for c in reversed(self.capas):
            g = c.backward(g)
        return g

    def params(self):
        return [p for c in self.capas for p in c.params()]


def encoder(n_in, z, rng, ultima=True):
    capas = []
    for a, b in ((n_in, 512), (512, 256), (256, 128)):
        capas += [Lineal(a, b, rng), BN(b), LReLU()]
    if ultima:
        capas.append(Lineal(128, z, rng, bias=False))
    return Seq(capas)


def decoder(z, n_out, rng):
    capas = []
    for a, b in ((z, 128), (128, 256), (256, 512)):
        capas += [Lineal(a, b, rng), BN(b), LReLU()]
    capas.append(Lineal(512, n_out, rng))
    return Seq(capas)


class Adam:
    """Como torch.optim.Adam: weight_decay suma wd * p al gradiente."""

    def __init__(self, params, lr, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
        self.ps = params; self.lr = lr; self.wd = wd
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self, clip=None):
        self.t += 1
        gs = [g + self.wd * p if self.wd else g for p, g in self.ps]
        if clip is not None:
            norma = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in gs))
            if norma > clip:
                gs = [g * (clip / (norma + 1e-6)) for g in gs]
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for i, ((p, _), g) in enumerate(zip(self.ps, gs)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= (self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)).astype(p.dtype)


def lotes(n, batch, rng):
    idx = rng.permutation(n)
    for i in range(0, n, batch):
        b = idx[i:i + batch]
        if len(b) >= 2:
            yield b


def _estado(objs):
    return copy.deepcopy(objs)


@dataclass
class Historial:
    train: list
    val: list


# ------------------------------------------------------------------ autoencoder

class AE:
    def __init__(self, n_in, z, rng):
        self.enc = encoder(n_in, z, rng); self.dec = decoder(z, n_in, rng)

    def params(self):
        return self.enc.params() + self.dec.params()

    def latente(self, x):
        return np.concatenate([self.enc.forward(x[i:i + 4096], False) for i in range(0, len(x), 4096)])

    def reconstruir(self, x):
        return self.dec.forward(self.latente(x), False)


def entrenar_ae(x_tr, x_va, z=10, n_epocas=100, lr=1e-3, batch=128, paciencia=8, semilla=42):
    rng = np.random.default_rng(semilla)
    x_tr = x_tr.astype(DT); x_va = x_va.astype(DT)
    m = AE(x_tr.shape[1], z, rng)
    opt = Adam(m.params(), lr, wd=1e-5)
    hist = Historial([], []); mejor = np.inf; espera = 0; mejor_m = _estado(m)
    for _ in range(n_epocas):
        perd = []
        for b in lotes(len(x_tr), batch, rng):
            xb = x_tr[b]
            z_ = m.enc.forward(xb, True); xr = m.dec.forward(z_, True)
            d = xr - xb
            perd.append(float((d ** 2).mean()))
            g = (2.0 / d.size) * d
            m.enc.backward(m.dec.backward(g))
            opt.step()
        val = float(((m.reconstruir(x_va) - x_va) ** 2).mean())
        hist.train.append(float(np.mean(perd))); hist.val.append(val)
        if val < mejor - 1e-5:
            mejor = val; espera = 0; mejor_m = _estado(m)
        else:
            espera += 1
            if espera >= paciencia:
                break
    return mejor_m, hist


def error_reconstruccion(ae, x):
    return ((ae.reconstruir(x.astype(DT)) - x) ** 2).mean(axis=1)


# ------------------------------------------------------------------ MCDSVDD

class MCD:
    def __init__(self, ae: AE, x_tr, y_tr, eps=0.01):
        self.enc = copy.deepcopy(ae.enc)
        z = self.latente(x_tr)
        cs = []
        for k in range(int(y_tr.max()) + 1):
            c = z[y_tr == k].mean(axis=0)
            c = np.where((np.abs(c) < eps) & (c < 0), -eps, c)
            c = np.where((np.abs(c) < eps) & (c >= 0), eps, c)
            cs.append(c)
        self.c = np.stack(cs).astype(DT)

    def latente(self, x):
        return np.concatenate([self.enc.forward(x[i:i + 4096], False) for i in range(0, len(x), 4096)])

    def perdida(self, z, y):
        d2 = ((z - self.c[y]) ** 2).sum(axis=1)
        tot = 0.0
        for k in np.unique(y):
            tot += d2[y == k].mean()
        return float(tot)

    def puntaje(self, x):
        z = self.latente(x.astype(DT))
        return ((z[:, None, :] - self.c[None]) ** 2).sum(axis=2).min(axis=1)

    def asignar(self, x):
        z = self.latente(x.astype(DT))
        return ((z[:, None, :] - self.c[None]) ** 2).sum(axis=2).argmin(axis=1)


def entrenar_mcdsvdd(ae, x_tr, y_tr, x_va, y_va, n_epocas=2000, lr=1e-4, batch=1024,
                     wd=0.5e-6, paciencia=100, semilla=42):
    """Ecuación 5 del paper: sum_j (1/M_j) sum_{i en j} ||z_i - c_j||^2."""
    rng = np.random.default_rng(semilla)
    x_tr = x_tr.astype(DT); x_va = x_va.astype(DT)
    m = MCD(ae, x_tr, y_tr)
    opt = Adam(m.enc.params(), lr, wd=wd)
    hist = Historial([], []); mejor = np.inf; espera = 0; mejor_m = _estado(m)
    for _ in range(n_epocas):
        perd = []
        for b in lotes(len(x_tr), batch, rng):
            xb, yb = x_tr[b], y_tr[b]
            z = m.enc.forward(xb, True)
            dif = z - m.c[yb]
            conteo = np.bincount(yb, minlength=len(m.c)).astype(DT)
            g = 2.0 * dif / conteo[yb][:, None]
            perd.append(m.perdida(z, yb))
            m.enc.backward(g.astype(DT)); opt.step()
        val = m.perdida(m.latente(x_va), y_va)
        hist.train.append(float(np.mean(perd))); hist.val.append(val)
        if val < mejor - 1e-5:
            mejor = val; espera = 0; mejor_m = _estado(m)
        else:
            espera += 1
            if espera >= paciencia:
                break
    return mejor_m, hist


# ------------------------------------------------------------------ VaDE

LV_MIN, LV_MAX = -10.0, 10.0
LOG2PI = float(np.log(2 * np.pi))


def _logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class VaDE:
    """Mismo modelo que src/ic_project/vade.py. En el gradiente, gamma = p(c|z) se trata como
    constante (paso E); el valor de la pérdida es el mismo."""

    def __init__(self, ae: AE, x_tr, n_comp, y_tr=None, semilla=42):
        from ic_project.gmm import MezclaGaussianas
        self.tronco = Seq(copy.deepcopy(ae.enc.capas[:-1]))
        ult = ae.enc.capas[-1]
        rng = np.random.default_rng(semilla)
        self.cm = Lineal(128, ult.W.shape[1], rng); self.cm.W[...] = ult.W; self.cm.b[...] = 0
        self.cl = Lineal(128, ult.W.shape[1], rng)
        self.dec = copy.deepcopy(ae.dec)
        z = ae.latente(x_tr.astype(DT)).astype(np.float64)
        var = z.var(axis=0) + 1e-8
        self.cl.W[...] = 0; self.cl.b[...] = np.log(0.01 * var)
        if y_tr is None:
            g = MezclaGaussianas(n_comp, n_init=5, random_state=semilla).fit(z)
            pi, mu, v = g.weights_, g.means_, np.diagonal(g.covariances_, axis1=1, axis2=2)
        else:
            n_comp = int(y_tr.max()) + 1
            pi = np.bincount(y_tr, minlength=n_comp) / len(y_tr)
            mu = np.stack([z[y_tr == c].mean(0) for c in range(n_comp)])
            v = np.stack([z[y_tr == c].var(0) for c in range(n_comp)])
        self.logits = np.log(pi).astype(DT); self.mp = mu.astype(DT); self.lvp = np.log(v + 1e-6).astype(DT)
        self.g_logits = np.zeros_like(self.logits); self.g_mp = np.zeros_like(self.mp)
        self.g_lvp = np.zeros_like(self.lvp)

    def params(self):
        return (self.tronco.params() + self.cm.params() + self.cl.params() + self.dec.params()
                + [(self.logits, self.g_logits), (self.mp, self.g_mp), (self.lvp, self.g_lvp)])

    def log_pi(self):
        return self.logits - _logsumexp(self.logits[None], axis=1)[0]

    def log_conjunta(self, z):
        lvp = np.clip(self.lvp, LV_MIN, LV_MAX)
        d2 = (z[:, None, :] - self.mp[None]) ** 2
        return -0.5 * (LOG2PI + lvp[None] + d2 / np.exp(lvp)[None]).sum(axis=2) + self.log_pi()

    def paso(self, x, y=None, train=True, rng=None, backward=False):
        """Términos de -ELBO por estrella; con backward=True deja los gradientes de la media."""
        h = self.tronco.forward(x, train)
        mu = self.cm.forward(h, train)
        lv_raw = self.cl.forward(h, train)
        lv = np.clip(lv_raw, LV_MIN, LV_MAX)
        if train:
            eps = rng.standard_normal(mu.shape).astype(DT)
            sd = np.exp(0.5 * lv); z = mu + eps * sd
        else:
            z = mu
        xr = self.dec.forward(z, train)
        rec = ((x - xr) ** 2).sum(axis=1)
        log_pi = self.log_pi()
        C = len(self.logits)
        if y is None:
            lc = self.log_conjunta(z)
            gam = np.exp(lc - _logsumexp(lc)[:, None])
            klc = (gam * (np.log(gam + 1e-10) - log_pi[None])).sum(axis=1)
        else:
            gam = np.eye(C, dtype=DT)[y]
            klc = -log_pi[y]
        lvp = np.clip(self.lvp, LV_MIN, LV_MAX); vp = np.exp(lvp)
        dif = mu[:, None, :] - self.mp[None]
        porc = (lvp[None] + (np.exp(lv)[:, None, :] + dif ** 2) / vp[None]).sum(axis=2)
        klz = 0.5 * (gam * porc).sum(axis=1) - 0.5 * (1 + lv).sum(axis=1)
        total = rec + klz + klc
        if backward:
            n = len(x)
            gxr = (2.0 / n) * (xr - x)
            gz = self.dec.backward(gxr.astype(DT))
            gmu = gz + (gam[:, :, None] * dif / vp[None]).sum(axis=1) / n
            glv = gz * eps * sd * 0.5 + (0.5 * (gam[:, :, None] * np.exp(lv)[:, None, :] / vp[None]).sum(axis=1) - 0.5) / n
            glv = glv * ((lv_raw > LV_MIN) & (lv_raw < LV_MAX))
            self.g_mp[...] = -(gam[:, :, None] * dif / vp[None]).sum(axis=0) / n
            dlvp = 0.5 * gam[:, :, None] * (1 - (np.exp(lv)[:, None, :] + dif ** 2) / vp[None])
            self.g_lvp[...] = dlvp.sum(axis=0) / n * ((self.lvp > LV_MIN) & (self.lvp < LV_MAX))
            pi = np.exp(log_pi)
            self.g_logits[...] = (pi[None] - gam).sum(axis=0) / n
            gh = self.cm.backward(gmu.astype(DT)) + self.cl.backward(glv.astype(DT))
            self.tronco.backward(gh)
        return total, rec, klz + klc, gam

    def salidas(self, x):
        x = x.astype(DT); out = {"menos_elbo": [], "gamma": [], "z": []}
        for i in range(0, len(x), 4096):
            xb = x[i:i + 4096]
            tot, _, _, gam = self.paso(xb, train=False)
            h = self.tronco.forward(xb, False)
            out["menos_elbo"].append(tot); out["gamma"].append(gam); out["z"].append(self.cm.forward(h, False))
        return {k: np.concatenate(v) for k, v in out.items()}


def entrenar_vade(ae, x_tr, x_va, n_comp=5, y_tr=None, y_va=None, n_epocas=300, lr=5e-4,
                  batch=256, paciencia=30, semilla=42):
    rng = np.random.default_rng(semilla)
    x_tr = x_tr.astype(DT); x_va = x_va.astype(DT)
    m = VaDE(ae, x_tr, n_comp, y_tr, semilla)
    opt = Adam(m.params(), lr)
    hist = Historial([], []); mejor = np.inf; espera = 0; mejor_m = _estado(m)
    for _ in range(n_epocas):
        perd = []
        for b in lotes(len(x_tr), batch, rng):
            yb = None if y_tr is None else y_tr[b]
            tot, _, _, _ = m.paso(x_tr[b], yb, True, rng, backward=True)
            opt.step(clip=5.0); perd.append(float(tot.mean()))
        tv, _, _, _ = m.paso(x_va, y_va, train=False)
        val = float(tv.mean())
        hist.train.append(float(np.mean(perd))); hist.val.append(val)
        if not np.isfinite(val):
            break
        if val < mejor - 1e-4:
            mejor = val; espera = 0; mejor_m = _estado(m)
        else:
            espera += 1
            if espera >= paciencia:
                break
    return mejor_m, hist
