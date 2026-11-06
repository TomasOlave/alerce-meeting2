"""Genera las figuras de la presentacion preliminar con el mismo pipeline del notebook meeting2.

Uso: figuras_presentacion.py <raiz del repo> <carpeta de salida> <carpeta con prep_iqr.py>
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
raiz = Path(sys.argv[1])
salida = Path(sys.argv[2])
salida.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(raiz / "src"))
sys.path.insert(0, sys.argv[3])

import umap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

import prep_iqr
from ic_project.modelos import asignar_mcdsvdd, entrenar_autoencoder, entrenar_mcdsvdd, latente
from ic_project.preprocesamiento import (
    Preprocesador,
    aplicar_transformaciones_elemento,
    especificar_transformaciones,
    particionar,
    preparar_conjuntos,
)
from ic_project.visualizacion import fase_plegada

AZUL = "#214796"  # color structure del template
VECINOS = 20  # n_neighbors de todos los UMAP de la presentacion
COLORES = {"CEP": "steelblue", "DSCT": "chocolate", "EB": "mediumseagreen", "LPV": "indianred", "RRL": "magenta"}
COL_CLUSTER = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
BANDAS = {1: "tab:green", 2: "tab:red"}
AZULES = LinearSegmentedColormap.from_list("azules", ["#ffffff", AZUL])

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.dpi": 220,
        "figure.dpi": 100,
    }
)


def guardar(fig, nombre):
    fig.savefig(salida / nombre, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("figura:", nombre, flush=True)


def sin_ejes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#bbbbbb")


def puntos_por_clase(ax, xy, clases, s=5, alpha=0.55):
    for clase, color in COLORES.items():
        m = clases == clase
        ax.scatter(xy[m, 0], xy[m, 1], s=s, alpha=alpha, c=color, linewidths=0, label=clase, rasterized=True)
    sin_ejes(ax)


def leyenda_clases(fig, y=-0.02):
    manijas = [plt.Line2D([], [], marker="o", ls="", ms=4.5, color=c, label=k) for k, c in COLORES.items()]
    fig.legend(handles=manijas, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y), columnspacing=1.4, handletextpad=0.2)


def mapa_calor(ax, tabla, titulo, xlabel):
    frac = tabla.div(tabla.sum(axis=1), axis=0).to_numpy()
    ax.imshow(frac, cmap=AZULES, vmin=0, vmax=1, aspect="auto")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            ax.text(j, i, int(tabla.iat[i, j]), ha="center", va="center", fontsize=7.5, color="white" if frac[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(tabla.shape[1]), [str(c) for c in tabla.columns])
    ax.set_yticks(range(tabla.shape[0]), list(tabla.index))
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- datos
ruta_datos = raiz / "data" / "raw"
df_features = pd.read_parquet(ruta_datos / "features.parquet")
df_oids = pd.read_parquet(ruta_datos / "oids.parquet")
X = df_features.drop(columns="oid")
oids = df_features["oid"].reset_index(drop=True)
labels = df_features[["oid"]].merge(df_oids, on="oid", how="left")["alerceclass"].reset_index(drop=True)
print("objetos con features:", len(X), "| etiquetados en total:", len(df_oids), "| features:", X.shape[1])
print("por clase:", labels.value_counts().to_dict())
print("features con NaN:", int((X.isna().sum() > 0).sum()), "| NaN maximo:", round(float(X.isna().mean().max() * 100), 1))

conjuntos, prep, descartadas = preparar_conjuntos(X, labels, oids, solo_periodo_color=False)
x_tr, x_va, x_te = conjuntos["x_train"], conjuntos["x_val"], conjuntos["x_test"]
y_te = conjuntos["y_test"].to_numpy()
clases = sorted(conjuntos["y_train"].unique())
mapa = {c: i for i, c in enumerate(clases)}
ytr_i, yva_i = conjuntos["y_train"].map(mapa).to_numpy(), conjuntos["y_val"].map(mapa).to_numpy()
print("train/val/test:", len(x_tr), len(x_va), len(x_te), "| features al modelo:", x_tr.shape[1], "| descartadas:", len(descartadas))
print("NaN maximo entre las 158:", round(float(X[prep.columnas_].isna().mean().max() * 100), 1))


def medir(nombre, z, pred, real):
    print(
        f"{nombre}: ARI {adjusted_rand_score(real, pred):.3f} | NMI {normalized_mutual_info_score(real, pred):.3f} | "
        f"sil grupos {silhouette_score(z, pred):.3f} | sil clases {silhouette_score(z, real):.3f}",
        flush=True,
    )


# ---------------------------------------------------------------- modelos (igual que el notebook)
ae, hist = entrenar_autoencoder(x_tr, x_va, z_dim=8, n_epocas=40, semilla=42)
print("AE: epocas", len(hist.val), "| mejor mse val", round(min(hist.val), 4))
z_tr, z_te = latente(ae, x_tr), latente(ae, x_te)
gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42, n_init=5).fit(z_tr)
pred_gmm = gmm.predict(z_te)
medir("GMM (z-score)", z_te, pred_gmm, y_te)
tabla_gmm = pd.crosstab(y_te, pred_gmm)
print(tabla_gmm)

rng = np.random.default_rng(42)
por_clase = [np.flatnonzero(ytr_i == k) for k in range(len(clases))]
n_max = max(len(ix) for ix in por_clase)
balanceado = np.concatenate([rng.choice(ix, size=n_max, replace=len(ix) < n_max) for ix in por_clase])
mcd, _ = entrenar_mcdsvdd(x_tr[balanceado], ytr_i[balanceado], x_va, yva_i, autoencoder=ae, z_dim=8, n_epocas=25, semilla=42)
z_mcd = latente(mcd, x_te)
pred_i = asignar_mcdsvdd(mcd, x_te)
pred_mcd = np.array(clases)[pred_i]
medir("MCDSVDD con sobremuestreo", z_mcd, pred_mcd, y_te)
tabla_mcd = pd.crosstab(y_te, pred_mcd)
print(tabla_mcd)

mcd0, _ = entrenar_mcdsvdd(x_tr, ytr_i, x_va, yva_i, autoencoder=ae, z_dim=8, n_epocas=25, semilla=42)
pred_mcd0 = np.array(clases)[asignar_mcdsvdd(mcd0, x_te)]
medir("MCDSVDD sin sobremuestreo", latente(mcd0, x_te), pred_mcd0, y_te)
acierto0 = pd.Series({c: float((pred_mcd0[y_te == c] == c).mean()) for c in clases})
acierto1 = pd.Series({c: float((pred_mcd[y_te == c] == c).mean()) for c in clases})
print("acierto por clase sin/con:", acierto0.round(3).to_dict(), acierto1.round(3).to_dict())

# escala anterior (IQR), mismo resto del pipeline, para la comparacion
conj_iqr, _, _ = prep_iqr.preparar_conjuntos(X, labels, oids)
assert (conj_iqr["oid_test"] == conjuntos["oid_test"]).all()
m2 = (conj_iqr["x_train"].astype(np.float64) ** 2).mean(axis=0)
acum = np.cumsum(np.sort(m2)[::-1]) / m2.sum()
print("IQR: features que juntan 50% / 90% de la magnitud:", int((acum < 0.5).sum() + 1), int((acum < 0.9).sum() + 1), "| max abs:", round(float(np.abs(conj_iqr["x_train"]).max()), 0))
ae_iqr, _ = entrenar_autoencoder(conj_iqr["x_train"], conj_iqr["x_val"], z_dim=8, n_epocas=40, semilla=42)
zi_tr, zi_te = latente(ae_iqr, conj_iqr["x_train"]), latente(ae_iqr, conj_iqr["x_test"])
gmm_iqr = GaussianMixture(n_components=5, covariance_type="full", random_state=42, n_init=5).fit(zi_tr)
medir("GMM (IQR)", zi_te, gmm_iqr.predict(zi_te), y_te)
mcd_iqr, _ = entrenar_mcdsvdd(conj_iqr["x_train"], ytr_i, conj_iqr["x_val"], yva_i, autoencoder=ae_iqr, z_dim=8, n_epocas=25, semilla=42)
medir("MCDSVDD (IQR, sin sobremuestreo)", latente(mcd_iqr, conj_iqr["x_test"]), np.array(clases)[asignar_mcdsvdd(mcd_iqr, conj_iqr["x_test"])], y_te)

# ---------------------------------------------------------------- UMAP
emb_ae = umap.UMAP(n_neighbors=VECINOS, min_dist=0.1, n_components=2, random_state=42).fit_transform(z_te)
emb_iqr = umap.UMAP(n_neighbors=VECINOS, min_dist=0.1, n_components=2, random_state=42).fit_transform(zi_te)
emb_mcd = umap.UMAP(n_neighbors=VECINOS, min_dist=0.1, n_components=2, random_state=42).fit_transform(z_mcd)

# figura: efecto de la escala
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55))
puntos_por_clase(axes[0], emb_iqr, y_te)
axes[0].set_title("Escala IQR")
puntos_por_clase(axes[1], emb_ae, y_te)
axes[1].set_title("Escala z-score")
leyenda_clases(fig, y=-0.07)
fig.tight_layout(w_pad=2.0)
guardar(fig, "escala_latente.png")

# figura: clustering GMM
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.02]})
puntos_por_clase(axes[0], emb_ae, y_te)
axes[0].set_title("Latente por clase")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, markerscale=1.6, handletextpad=0.0, columnspacing=0.5, fontsize=6.5)
for k in range(5):
    m = pred_gmm == k
    axes[1].scatter(emb_ae[m, 0], emb_ae[m, 1], s=5, alpha=0.6, c=COL_CLUSTER[k], linewidths=0, label=f"c{k}", rasterized=True)
sin_ejes(axes[1])
axes[1].set_title("Clustering GMM")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, markerscale=1.6, handletextpad=0.0, columnspacing=0.5, fontsize=6.5)
mapa_calor(axes[2], tabla_gmm, "Clase vs. componente", "componente GMM")
fig.tight_layout(w_pad=1.2)
guardar(fig, "gmm.png")

# figura: clustering MCDSVDD
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75), gridspec_kw={"width_ratios": [1.18, 1]})
puntos_por_clase(axes[0], emb_mcd, y_te, s=7)
axes[0].set_title("Clustering MCDSVDD")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, markerscale=1.5, handletextpad=0.0, columnspacing=0.7, fontsize=7.5)
mapa_calor(axes[1], tabla_mcd, "Clase vs. centro", "centro asignado")
fig.tight_layout(w_pad=2.2)
guardar(fig, "mcdsvdd.png")

# ---------------------------------------------------------------- curvas de luz
lcs = pd.read_parquet(
    ruta_datos / "lcs.parquet",
    columns=["oid", "mjd", "fid", "magpsf_corr", "corrected", "dubious"],
    filters=[("oid", "in", list(conjuntos["oid_test"]))],
)
limpias = lcs[lcs["corrected"] & ~lcs["dubious"] & (lcs["magpsf_corr"] < 30)]
curvas = dict(tuple(limpias.groupby("oid")))
oid_te = conjuntos["oid_test"].to_numpy()
per_te = conjuntos["periodo_test"].to_numpy()
n_det = limpias.groupby("oid").size().reindex(oid_te, fill_value=0).to_numpy()


def curva(ax, i, titulo, plegada=True, s=3.0):
    sub = curvas.get(oid_te[i])
    if sub is not None:
        for fid, color in BANDAS.items():
            b = sub[sub["fid"] == fid]
            mag = b["magpsf_corr"].to_numpy()
            if plegada:
                fase = fase_plegada(b["mjd"].to_numpy(), float(per_te[i]))
                ax.scatter(np.r_[fase, fase + 1], np.tile(mag, 2), s=s, alpha=0.5, c=color, linewidths=0, rasterized=True)
            else:
                ax.scatter(b["mjd"], mag, s=s, alpha=0.5, c=color, linewidths=0, rasterized=True)
        ax.invert_yaxis()
    ax.set_title(titulo, fontsize=7.5, pad=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#bbbbbb")


def mahalanobis2(z, modelo, k):
    d = z - modelo.means_[k]
    return np.einsum("ij,jk,ik->i", d, modelo.precisions_[k], d)


centros = mcd.centros.detach().cpu().numpy()
dist_centros = np.sqrt(((z_mcd[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2))
protos_gmm, protos_mcd = {}, {}
for k in range(5):
    del_k = np.flatnonzero(pred_gmm == k)
    protos_gmm[k] = del_k[np.argsort(mahalanobis2(z_te[del_k], gmm, k))[:3]]
    del_k = np.flatnonzero(pred_i == k)
    protos_mcd[clases[k]] = del_k[np.argsort(dist_centros[del_k, k])[:3]]
print("prototipos GMM:", {k: [(oid_te[i], y_te[i], round(float(per_te[i]), 4)) for i in v] for k, v in protos_gmm.items()})
print("prototipos MCD:", {k: [(oid_te[i], y_te[i], round(float(per_te[i]), 4)) for i in v] for k, v in protos_mcd.items()})


def per(p):
    txt = f"{p:.3f}" if p < 10 else f"{p:.1f}" if p < 100 else f"{p:.0f}"
    return "P = " + txt.replace(".", ",") + " d"


# figura: una curva tipica por clase (el prototipo MCDSVDD con mas detecciones entre los 3 mas centrales)
fig, axes = plt.subplots(1, 5, figsize=(6.9, 1.55))
nombres = {"CEP": "Cefeida", "DSCT": "Delta Scuti", "EB": "Binaria eclipsante", "LPV": "Período largo", "RRL": "RR Lyrae"}
for ax, c in zip(axes, clases):
    i = max(protos_mcd[c], key=lambda j: n_det[j])
    curva(ax, i, f"{nombres[c]} ({c})\n{per(per_te[i])}", s=2.5)
axes[0].set_ylabel("brillo", fontsize=7.5)
fig.supxlabel("fase (dos ciclos)", fontsize=7.5, y=0.02)
fig.tight_layout(w_pad=0.6)
guardar(fig, "clases_curvas.png")

# figura: prototipos, el mas central de cada grupo
fig, axes = plt.subplots(2, 5, figsize=(6.9, 2.75))
for k in range(5):
    i = protos_gmm[k][0]
    curva(axes[0, k], i, f"componente {k}: {y_te[i]}\n{per(per_te[i])}")
    c = clases[k]
    i = protos_mcd[c][0]
    curva(axes[1, k], i, f"centro {c}: {y_te[i]}\n{per(per_te[i])}")
axes[0, 0].set_ylabel("GMM", fontsize=9, fontweight="bold", color=AZUL)
axes[1, 0].set_ylabel("MCDSVDD", fontsize=9, fontweight="bold", color=AZUL)
fig.tight_layout(w_pad=0.6, h_pad=0.9)
guardar(fig, "prototipos.png")

# figura: outliers elegidos de las listas de los 6 mas extremos de cada modelo
loglik = gmm.score_samples(z_te).astype(float)
dist = dist_centros.min(axis=1)
top_gmm = list(oid_te[np.argsort(loglik)[:6]])
top_mcd = list(oid_te[np.argsort(dist)[::-1][:6]])
print("outliers GMM:", [(o, y_te[list(oid_te).index(o)], round(float(per_te[list(oid_te).index(o)]), 4), int(n_det[list(oid_te).index(o)])) for o in top_gmm])
print("outliers MCD:", [(o, y_te[list(oid_te).index(o)], round(float(per_te[list(oid_te).index(o)]), 4), int(n_det[list(oid_te).index(o)])) for o in top_mcd])
elegidos = [
    ("ZTF18aborpmv", "GMM: Mira real"),
    ("ZTF18aaywfxv", "GMM: período erróneo"),
    ("ZTF18abmefly", "MCDSVDD: período erróneo"),
    ("ZTF18abfxrti", "Ambos: 17 detecciones"),
]
for o, _ in elegidos:
    print("   ", o, "en GMM:", o in top_gmm, "| en MCD:", o in top_mcd)
fig, axes = plt.subplots(2, 4, figsize=(6.9, 2.75))
for j, (o, titulo) in enumerate(elegidos):
    i = list(oid_te).index(o)
    curva(axes[0, j], i, f"{titulo}\n{y_te[i]}, {per(per_te[i])}", plegada=False)
    curva(axes[1, j], i, "", plegada=True)
axes[0, 0].set_ylabel("en el tiempo", fontsize=8)
axes[1, 0].set_ylabel("plegada", fontsize=8)
fig.tight_layout(w_pad=0.6, h_pad=0.5)
guardar(fig, "outliers.png")

frec = 1 / per_te
alias = (np.round(frec) >= 1) & (np.abs(frec - np.round(frec)) < 0.01)
print("periodo en alias de 1 dia por clase (test):", pd.Series(alias).groupby(y_te).mean().round(3).to_dict())

# ---------------------------------------------------------------- transformaciones y recorte
partes = particionar(X, labels, oids)
spec = especificar_transformaciones(partes["x_train"])
print("columnas con log1p:", len(spec["log"]), "| con log10:", len(spec["periodo"]), "| con arcsinh:", len(spec["arcsinh"]))
xt = aplicar_transformaciones_elemento(partes["x_train"], spec)  # con logaritmos, antes del recorte
p01, p99 = xt.quantile(0.01), xt.quantile(0.99)
fuera = float(((xt < p01) | (xt > p99)).to_numpy().sum() / xt.notna().to_numpy().sum())
print("valores de train fuera del rango p1-p99:", round(100 * fuera, 2), "%")
ejemplo = "Harmonics_mag_1_g"
v = xt[ejemplo].dropna()
print(ejemplo, "tras log1p: p1", round(float(p01[ejemplo]), 3), "| mediana", round(float(v.median()), 3), "| p99", round(float(p99[ejemplo]), 3), "| max", round(float(v.max()), 2))

fig, axes = plt.subplots(2, 1, figsize=(3.9, 3.75))
ax = axes[0]
logp = np.log10(partes["x_train"]["Multiband_period_g_r"].clip(lower=1e-8)).to_numpy()
y_tr_txt = partes["y_train"].to_numpy()
bordes = np.linspace(-1.7, 3.2, 70)
for clase, color in COLORES.items():
    ax.hist(logp[y_tr_txt == clase], bins=bordes, histtype="stepfilled", alpha=0.5, color=color, label=clase, linewidth=0)
ax.set_xticks([-1, 0, 1, 2, 3], ["0,1", "1", "10", "100", "1000"])
ax.set_xlabel("período multibanda (días)")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_title("Período en escala logarítmica")
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.6, handlelength=0.9, handletextpad=0.25, fontsize=6.5)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)

ax = axes[1]
ax.hist(v, bins=70, color=AZUL, alpha=0.85)
ax.set_yscale("log")
lim = float(p99[ejemplo])
ax.axvline(lim, color="#d95f02", ls="--", lw=1.0)
ax.axvspan(lim, float(v.max()), color="#d95f02", alpha=0.12, lw=0)
ax.text(
    lim + 0.02 * float(v.max()),
    ax.get_ylim()[1] * 0.35,
    "sobre el percentil 99:" + chr(10) + "se reemplaza por el límite",
    fontsize=6.5,
    color="#a04000",
    va="top",
)
ax.set_xlabel("amplitud del primer armónico, banda g, tras log(1+x)")
ax.set_ylabel("objetos")
ax.set_title("Recorte de valores extremos")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
fig.tight_layout(h_pad=1.0)
guardar(fig, "transformaciones.png")

# ---------------------------------------------------------------- exploracion: UMAP de las features
orden = ["train", "val", "test"]
x_todo = np.vstack([conjuntos[f"x_{s}"] for s in orden])
y_todo = np.concatenate([conjuntos[f"y_{s}"].to_numpy() for s in orden])
emb_pre = umap.UMAP(n_neighbors=VECINOS, min_dist=0.1, n_components=2, random_state=42).fit_transform(x_todo)
x_crudo = X.fillna(X.median())
print("sin preprocesar, feature de mayor desviacion:", x_crudo.std().idxmax(), f"{x_crudo.std().max():.3g}", "| mediana de las desviaciones:", round(float(x_crudo.std().median()), 2))
emb_crudo = umap.UMAP(n_neighbors=VECINOS, min_dist=0.1, n_components=2, random_state=42).fit_transform(x_crudo)
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.55))
puntos_por_clase(axes[0], emb_crudo, labels.to_numpy(), s=2.5, alpha=0.45)
axes[0].set_title("Sin preprocesar")
puntos_por_clase(axes[1], emb_pre, y_todo, s=2.5, alpha=0.45)
axes[1].set_title("Con preprocesamiento")
leyenda_clases(fig, y=-0.08)
fig.tight_layout(w_pad=1.5)
guardar(fig, "umap_features.png")
print("listo")
