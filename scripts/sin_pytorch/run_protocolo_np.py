"""Protocolo de Pérez-Carrasco et al. (2023) como src/ic_project/protocolo.py (140 features),
con los modelos en NumPy. Agrega filas a results/protocolo_numpy.csv y salta las (clase, fold) ya hechas."""
from pathlib import Path
import os, time, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from comun import cargar
from ic_project.config import FEATURES_EXCLUIDAS
from ic_project.preprocesamiento import (Preprocesador, aplicar_transformaciones_elemento,
                                         columnas_a_descartar, especificar_transformaciones)
from ic_project.gmm import MezclaGaussianas
from ic_project.gmm_clase import GMMPorClase
import np_modelos as M

SEM = 42
x, _, y = cargar(); x = x.reset_index(drop=True); y = y.to_numpy()

# --- copiado de protocolo.py ---
idx = np.arange(len(y))
idx_train, idx_ts1 = train_test_split(idx, test_size=0.2, random_state=SEM, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEM)
folds = [(idx_train[a], idx_train[b]) for a, b in skf.split(idx_train, y[idx_train])]

def construir_ts2(clase, proporcion=0.1):
    rng = np.random.default_rng(SEM)
    inl = idx_ts1[y[idx_ts1] != clase]
    cand = np.concatenate([idx_ts1, idx_train]); cand = cand[y[cand] == clase]
    n_an = int(round(proporcion / (1 - proporcion) * len(inl)))
    if n_an > len(cand):
        n_an = len(cand); inl = rng.choice(inl, size=int(round(n_an * (1 - proporcion) / proporcion)), replace=False)
    an = rng.choice(cand, size=n_an, replace=False)
    return np.concatenate([inl, an]), np.r_[np.zeros(len(inl), int), np.ones(n_an, int)]

def preparar_fold(i_tr, otros):
    xx = x.drop(columns=columnas_a_descartar(x.columns) + list(FEATURES_EXCLUIDAS), errors="ignore")
    tr = xx.iloc[i_tr]; spec = especificar_transformaciones(tr); tr = aplicar_transformaciones_elemento(tr, spec)
    prep = Preprocesador().ajustar(tr)
    out = {"train": prep.transformar(tr).to_numpy(np.float32)}
    for k, ii in otros.items():
        out[k] = prep.transformar(aplicar_transformaciones_elemento(xx.iloc[ii], spec)).to_numpy(np.float32)
    return out
# --------------------------------

(Path(__file__).resolve().parents[2] / "results").mkdir(exist_ok=True)
ruta = str(Path(__file__).resolve().parents[2] / "results" / "protocolo_numpy.csv")
hechos = set()
if os.path.exists(ruta):
    d = pd.read_csv(ruta); hechos = set(zip(d.clase_anomala, d.fold))

for clase in sorted(set(y)):
    for fold in range(5):
        if (clase, fold) in hechos:
            continue
        t0 = time.time()
        i_tr, i_va = folds[fold]
        i_tr = i_tr[y[i_tr] != clase]; i_va = i_va[y[i_va] != clase]
        i_ts2, et = construir_ts2(clase)
        dat = preparar_fold(i_tr, {"val": i_va, "ts2": i_ts2})
        inl = sorted(set(y[i_tr])); mp = {c: i for i, c in enumerate(inl)}
        ytr = np.array([mp[c] for c in y[i_tr]]); yva = np.array([mp[c] for c in y[i_va]])

        ae, hae = M.entrenar_ae(dat["train"], dat["val"], z=10, n_epocas=100, semilla=SEM)
        mcd, hm = M.entrenar_mcdsvdd(ae, dat["train"], ytr, dat["val"], yva, semilla=SEM)
        P = {"mcdsvdd": mcd.puntaje(dat["ts2"]), "ae": M.error_reconstruccion(ae, dat["ts2"])}
        za_tr, za_ts = ae.latente(dat["train"]), ae.latente(dat["ts2"])
        zm_tr, zm_ts = mcd.latente(dat["train"]), mcd.latente(dat["ts2"])
        P["gmm"] = -MezclaGaussianas(5, n_init=5, random_state=SEM).fit(za_tr).score_samples(za_ts)
        gca = GMMPorClase(4, random_state=SEM).fit(za_tr, ytr)
        gcm = GMMPorClase(4, random_state=SEM).fit(zm_tr, ytr)
        P["gmm_clase_ae"] = gca.puntaje_anomalia(za_ts)
        P["gmm_clase_mcd"] = gcm.puntaje_anomalia(zm_ts)
        P["gmm_clase_mcd_mezcla"] = gcm.puntaje_anomalia(zm_ts, "mezcla")
        v, hv = M.entrenar_vade(ae, dat["train"], dat["val"], n_comp=5, semilla=SEM)
        P["vade"] = v.salidas(dat["ts2"])["menos_elbo"]
        vc, hvc = M.entrenar_vade(ae, dat["train"], dat["val"], y_tr=ytr, y_va=yva, semilla=SEM)
        P["vade_clase"] = vc.salidas(dat["ts2"])["menos_elbo"]

        fila = {"clase_anomala": clase, "fold": fold}
        for k, s in P.items():
            fila[f"auroc_{k}"] = roc_auc_score(et, s)
        for k, s in P.items():
            fila[f"auprc_{k}"] = average_precision_score(et, s)
        fila.update({"k_gmm_clase_ae": " ".join(f"{inl[c]}:{k}" for c, k in sorted(gca.k_.items())),
                     "k_gmm_clase_mcd": " ".join(f"{inl[c]}:{k}" for c, k in sorted(gcm.k_.items())),
                     "epocas_ae": len(hae.val), "epocas_mcdsvdd": len(hm.val), "epocas_vade": len(hv.val),
                     "epocas_vade_clase": len(hvc.val), "n_ts2": len(i_ts2), "segundos": time.time() - t0})
        pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not os.path.exists(ruta), index=False)
        print(clase, fold, " ".join(f"{k}={fila['auroc_'+k]:.3f}" for k in P), f"{fila['segundos']:.0f}s", flush=True)
print("FIN", flush=True)
