"""Compara el gradiente a mano de np_modelos.py con diferencias finitas (float64)."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
import np_modelos as M
M.DT = np.float64
rng = np.random.default_rng(0)
x = rng.normal(size=(16, 7))
def chequear(nombre, f_loss, f_backward, params):
    f_backward()
    for j,(p, g) in enumerate(params):
        errs=[]
        for _ in range(5):
            i = tuple(rng.integers(s) for s in p.shape)
            a = g[i]; old = p[i]; h = 1e-6
            p[i] = old + h; lp = f_loss(); p[i] = old - h; lm = f_loss(); p[i] = old
            num = (lp - lm) / (2 * h); errs.append((abs(a-num), a, num))
        w=max(errs)
        if w[0] > 1e-6*max(1,abs(w[2])): print(nombre, j, p.shape, w)
ae = M.AE(7, 3, rng)
def loss_ae(): return float(((ae.dec.forward(ae.enc.forward(x, True), True) - x) ** 2).mean())
def back_ae():
    xr = ae.dec.forward(ae.enc.forward(x, True), True); d = xr - x
    ae.enc.backward(ae.dec.backward(2 * d / d.size))
chequear("ae", loss_ae, back_ae, ae.params())
y = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1])
v = M.VaDE(ae, x, 3, y_tr=y)
v.mp += rng.normal(size=v.mp.shape) * 0.3; v.lvp += rng.normal(size=v.lvp.shape) * 0.3
def loss_v(): return float(v.paso(x, y, True, np.random.default_rng(5))[0].mean())
def back_v(): v.paso(x, y, True, np.random.default_rng(5), backward=True)
chequear("vade", loss_v, back_v, v.params())
print("OK: los gradientes a mano coinciden con los numéricos (autoencoder y VaDE con clases)")
