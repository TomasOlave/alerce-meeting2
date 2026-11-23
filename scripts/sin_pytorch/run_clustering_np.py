"""Igual que scripts/run_vade.py (más MCDSVDD como referencia), con los modelos en NumPy."""

import json
import time
from pathlib import Path

import np_modelos as M
import numpy as np
import pandas as pd
from comun import cargar
from sklearn.metrics import f1_score

from ic_project.clustering import ajustar_gmm, metricas_espacio, tabla_cluster_clase
from ic_project.preprocesamiento import preparar_conjuntos

t0 = time.time()
x, oids, y = cargar()
c, _, _ = preparar_conjuntos(x, y, oids, semilla=42, solo_periodo_color=False)
xtr, xva, xte = c["x_train"], c["x_val"], c["x_test"]
ytr, yva, yte = (c[f"y_{s}"].to_numpy() for s in ("train", "val", "test"))
clases = sorted(set(ytr))
mapa = {k: i for i, k in enumerate(clases)}
nombres = np.array(clases)
ytr_i = np.array([mapa[v] for v in ytr])
yva_i = np.array([mapa[v] for v in yva])


def f1_mayoria(gtr, gte):
    m = pd.Series(ytr).groupby(gtr).agg(lambda s: s.mode()[0])
    return float(f1_score(yte, pd.Series(gte).map(m).fillna("x").to_numpy(), average="macro"))


ae, hae = M.entrenar_ae(xtr, xva, z=10, n_epocas=100)
print(
    f"AE: {len(hae.val)} épocas, MSE val {min(hae.val):.3f}  ({time.time() - t0:.0f}s)", flush=True
)
ztr, zte = ae.latente(xtr), ae.latente(xte)
mcd, hm = M.entrenar_mcdsvdd(ae, xtr, ytr_i, xva, yva_i)
print(f"MCDSVDD: {len(hm.val)} épocas ({time.time() - t0:.0f}s)", flush=True)
zmtr, zmte = mcd.latente(xtr.astype(np.float32)), mcd.latente(xte.astype(np.float32))

filas, tablas = {}, {}


def reg(nombre, z, pred, f1):
    filas[nombre] = {**metricas_espacio(z, pred, yte), "f1_macro": f1}
    tablas[nombre] = tabla_cluster_clase(yte, pred)
    print(
        f"  {nombre}: ARI {filas[nombre]['ari']:.3f} NMI {filas[nombre]['nmi']:.3f} F1 {f1:.3f}",
        flush=True,
    )


g = ajustar_gmm(ztr, 5, semilla=42)
p = g.predict(zte)
reg("GMM (sin etiquetas)", zte, p, f1_mayoria(g.predict(ztr), p))
p = nombres[mcd.asignar(xte)]
reg("MCDSVDD", zmte, p, float(f1_score(yte, p, average="macro")))

v, hv = M.entrenar_vade(ae, xtr, xva, n_comp=5)
print(f"VaDE: {len(hv.val)} épocas ({time.time() - t0:.0f}s)", flush=True)
str_, ste = v.salidas(xtr), v.salidas(xte)
p = ste["gamma"].argmax(1)
reg("VaDE (sin etiquetas)", ste["z"], p, f1_mayoria(str_["gamma"].argmax(1), p))
print("    tamaños componentes VaDE (test):", np.bincount(p, minlength=5).tolist())
vc, hvc = M.entrenar_vade(ae, xtr, xva, y_tr=ytr_i, y_va=yva_i)
print(f"VaDE con clases: {len(hvc.val)} épocas ({time.time() - t0:.0f}s)", flush=True)
ste = vc.salidas(xte)
p = nombres[ste["gamma"].argmax(1)]
reg("VaDE con clases", ste["z"], p, float(f1_score(yte, p, average="macro")))

out = pd.DataFrame(filas).T
out.index.name = "modelo"
out.to_csv("clustering_numpy.csv")
epocas = {
    "ae": len(hae.val),
    "mcdsvdd": len(hm.val),
    "vade": len(hv.val),
    "vade_clase": len(hvc.val),
}
Path("clustering_numpy.json").write_text(json.dumps({"epocas": epocas}, indent=1))
for k, t in tablas.items():
    t.to_csv(f"tabla_{k.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
print(out.round(3).to_string())
print(f"total {time.time() - t0:.0f}s")
