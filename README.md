# Clustering y anomalías en curvas de luz de estrellas periódicas

Proyecto del curso EL4106 Inteligencia Computacional, grupo 5B (Gabriel Fonseca y Tomás Olave).

Se usan las features de ALeRCE-ZTF de 8419 estrellas periódicas de cinco clases (CEP, DSCT, EB, LPV y RRL).
De las 178 features entran 140 (ver "Preprocesamiento" más abajo), que se preprocesan; un autoencoder las
comprime a un espacio latente de 10 dimensiones y sobre
ese espacio se agrupa con una mezcla de gaussianas (GMM, sin etiquetas) y con Multi-Class Deep SVDD
(MCDSVDD, con etiquetas). Después se revisan las curvas de luz de los prototipos y outliers de cada grupo,
los objetos que quedan entre dos componentes y los sub-clusters dentro de cada clase.

## Qué se entrega

| Ruta | Contenido |
|---|---|
| `notebooks/Datos_iniciales.ipynb` | Exploración de los datos: datos faltantes, escalas, descarte de features, transformaciones y UMAP |
| `notebooks/meeting2.ipynb` | Notebook principal: modelos, métricas, importancia de features y análisis de curvas de luz |
| `src/ic_project/` | Implementación: carga de datos, preprocesamiento, autoencoder, MCDSVDD, GMM, métricas y figuras |
| `scripts/run_meeting2.py` | Corre el pipeline del notebook desde la terminal, sin abrir Jupyter (ver más abajo) |
| `scripts/comparar_gmm.py` | Valida el GMM propio contra el de scikit-learn sobre el latente real |
| `scripts/run_dagmm.py` | Entrena DAGMM (autoencoder y GMM juntos, `src/ic_project/dagmm.py`) y lo compara con el pipeline actual |
| `scripts/run_protocolo_mcdsvdd.py` | Evalúa MCDSVDD con el protocolo del paper: cada clase como anomalía, AUROC en 5 folds (`src/ic_project/protocolo.py`) |
| `scripts/comparar_preprocesamiento.py` | Compara variantes de preprocesamiento en validación (pasos 1 a 7, ver "Preprocesamiento") |
| `tests/` | Pruebas automáticas |
| `presentacion/` | Fuente LaTeX de la presentación y el script que genera sus figuras |
| `data/raw/` | Lugar donde van los datos (no están en el repositorio) |

## Cómo revisarlo

Los notebooks se entregan con sus resultados guardados, así que se pueden leer sin ejecutarlos. Orden sugerido:

1. `notebooks/Datos_iniciales.ipynb`. Muestra por qué se descartan 20 de las 178 features (no-detecciones y
   colores WISE), por qué el período va en escala logarítmica y que sin preprocesar una sola feature domina
   las distancias.
2. `notebooks/meeting2.ipynb`, en este orden:
   - partición 70/15/15 estratificada, con recorte, imputación y escala z-score ajustados solo en train;
   - autoencoder con cuello de 10 dimensiones;
   - GMM de 5 componentes y MCDSVDD entrenado con el protocolo del paper, con ARI, NMI, silhouette y
     Davies-Bouldin en test;
   - qué features importan, con un random forest y la comparación entre 158 y 178 features;
   - análisis cualitativo del GMM: prototipos, outliers, objetos entre dos componentes y sub-clusters;
   - prototipos de MCDSVDD;
   - detección, caracterización y visualización de anomalías con MCDSVDD (`src/ic_project/anomalias.py`):
     umbral por centro calibrado con validación (percentil `PERCENTIL_ANOMALIA` = 99), categorías
     "candidato", "otra clase" y "posible error de datos", y las features en que cada anomalía se aleja de
     su clase;
   - los cuatro detectores (MCDSVDD, autoencoder, GMM y DAGMM) con el protocolo del paper (una clase fuera,
     AUROC).
3. `src/ic_project/`, si se quiere ver la implementación. Los archivos principales son
   `preprocesamiento.py` (el preprocesador se ajusta solo con train), `modelos.py` (autoencoder y MCDSVDD),
   `gmm.py` (mezcla de gaussianas con EM, implementación propia sin scikit-learn) y `cualitativo.py`
   (prototipos y outliers).

### GMM propio

`src/ic_project/gmm.py` implementa la mezcla de gaussianas de covarianza completa solo con NumPy:
inicialización con k-means++, pasos E y M de EM, criterio de convergencia, log-verosimilitud de cada objeto
y BIC. Conserva la interfaz de `sklearn.mixture.GaussianMixture` (`fit`, `predict`, `predict_proba`,
`score_samples`, `bic`, `means_`, `precisions_`), así que el resto del código no cambia.

Se validó contra scikit-learn (`tests/test_gmm.py` y `scripts/comparar_gmm.py`). Partiendo de los mismos
parámetros iniciales, los dos llegan a los mismos parámetros con diferencias de 1e-15 y en las mismas
iteraciones. Cada uno con su propia inicialización (k-means++ propio contra el KMeans de scikit-learn) pueden
terminar en óptimos locales distintos: con el latente de 10 dimensiones llegan a soluciones con casi la misma
log-verosimilitud (−12.6603 y −12.6606 por objeto) pero ARI distinto en test (0.28 y 0.19). EM maximiza la
verosimilitud, no el acuerdo con las clases, así que el ARI del GMM depende de la inicialización.

### MCDSVDD con el protocolo del paper

`modelos.entrenar_mcdsvdd` sigue a Pérez-Carrasco et al. (2023, sección 3) y a su código
([AnomalyALeRCE](https://github.com/mperezcarrasco/AnomalyALeRCE)): el encoder parte del autoencoder
preentrenado, los centros son la media del latente de cada clase y se minimiza la ecuación 5 (distancia media
dentro de cada clase, sumada sobre las clases, así que cada clase pesa lo mismo) con weight decay
λ = 0.5·10⁻⁶. Adam con lr 1e-4, batch 1024 y early stopping con paciencia 100 sobre la pérdida de validación.

`protocolo.py` implementa la evaluación de la sección 4: 80/20 estratificado, 5 folds dentro del 80%, cada
clase se saca del entrenamiento y se busca en un test con 10% de anomalías. AUROC en test, media ± desviación
sobre los 5 folds (`scripts/run_protocolo_mcdsvdd.py`, unos 30 minutos):

| Clase como anomalía | MCDSVDD | MCDSVDD en el paper | Autoencoder | Autoencoder en el paper |
|---|---|---|---|---|
| CEP | 0.855 ± 0.024 | 0.858 | 0.611 | 0.564 |
| DSCT | 0.768 ± 0.039 | 0.819 | 0.567 | 0.367 |
| EB | 0.784 ± 0.023 | 0.945 | 0.712 | 0.864 |
| LPV | 0.793 ± 0.018 | 0.953 | 0.968 | 0.996 |
| RRL | 0.803 ± 0.022 | 0.953 | 0.438 | 0.907 |

Como en el paper, MCDSVDD es mejor que el autoencoder para CEP, DSCT, EB y RRL, y el autoencoder es mejor
para LPV. La comparación con el paper no es directa: usan otros datos, escalan las features a [0, 1] y el
latente es más grande (aquí `Z_DIM = 10`, elegido para el GMM).

Los cuatro detectores con el mismo protocolo y las 140 features del pipeline
(`run_protocolo_mcdsvdd.py --features pipeline --con-gmm --con-dagmm`, cerca de una hora). El puntaje del GMM
es −log-verosimilitud sobre el latente del autoencoder y el de DAGMM es su energía:

| Clase como anomalía | MCDSVDD | Autoencoder | GMM | DAGMM |
|---|---|---|---|---|
| CEP | 0.860 | 0.624 | 0.495 | 0.548 |
| DSCT | 0.740 | 0.572 | 0.454 | 0.412 |
| EB | 0.812 | 0.734 | 0.609 | 0.650 |
| LPV | 0.799 | 0.969 | 0.965 | 0.956 |
| RRL | 0.827 | 0.436 | 0.445 | 0.391 |

### Hiperparámetros

Están en `src/ic_project/config.py` y los usan el notebook y los scripts. Se eligieron con el conjunto de
**validación**, nunca con test:

| Constante | Valor | Cómo se eligió |
|---|---|---|
| `Z_DIM` | 10 | Se probaron 4, 6, 8, 10, 12 y 16 con 3 semillas. Con 10, el ARI del GMM en validación fue 0.264 ± 0.028 (3 semillas del autoencoder × 8 del GMM), contra 0.216 ± 0.018 con 8, y MCDSVDD también quedó mejor (ARI 0.80 contra 0.79). |
| `N_COMPONENTES_GMM` | 5 | Se probaron 5, 8, 10, 12 y 15 con `Z_DIM = 10`. El ARI baja al agregar componentes (0.27 con 5, 0.24 con 10). |
| `EPOCAS_AE` | 100 | Es un tope. El early stopping con validación paró entre las épocas 28 y 64. |
| `FEATURES_EXCLUIDAS` | 18 features | Ver "Preprocesamiento". Con ellas fuera quedan 140. |

La semilla (42) no se ajusta: elegir la semilla que da mejor resultado es ajustar al azar. `Z_DIM` y
`N_COMPONENTES_GMM` se eligieron con las 158 features, antes de excluir las 18.

### Preprocesamiento

`scripts/comparar_preprocesamiento.py --paso N` compara variantes en **validación** (3 semillas del
autoencoder × 3 del GMM; ARI y NMI del GMM, F1 de un random forest y ARI de MCDSVDD):

| Paso | Qué se probó | Resultado |
|---|---|---|
| 1 | Quitar features de observación (muestreo, posición, magnitud aparente, SPM) | ninguna variante mejora |
| 2 | Imputación: KNN, iterativa, indicadores de faltante | ninguna mejora el GMM; KNN limpia los sub-clusters |
| 3 | Quitar redundancia, unificar bandas g y r, PCA con whitening | ninguna mejora; quitar redundancia empeora |
| 4 | Filtrar objetos como el paper (≥ 6 detecciones en g y r) | mejora solo porque se evalúa sobre objetos más fáciles |
| 5 | Quitar features con mucho NaN y poca importancia hasta 140/130/120 | **140 se adopta**: con 5 semillas, ARI 0.262 → 0.284 y NMI 0.307 → 0.339 |
| 6 | Recorte: sin recorte, percentiles 1/2.5/5, Tukey 1.5 y 3 IQR | el actual (percentiles 1 y 99) es el mejor |
| 7 | Escala: z-score, robusta, min-max, cuantiles a normal, Yeo-Johnson | el actual (z-score) es el mejor |

Las 18 excluidas (`FEATURES_EXCLUIDAS`) son las versiones de la banda r de features poco importantes, los
`MHPS_PN_flag`, `delta_period_r` y `SPM_chi_r`. El protocolo de MCDSVDD (`protocolo.py`) sigue usando las
158 features, porque replica al paper.

Resultados principales en test:

| Modelo | ARI | NMI |
|---|---|---|
| GMM | 0.330 | 0.372 |
| DAGMM | 0.023 | 0.085 |
| MCDSVDD | 0.775 | 0.724 |

Con las 158 features eran GMM 0.284 / 0.316 y MCDSVDD 0.770 / 0.714.

## Instalación

Requiere Python 3.11 o superior y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

Esto crea el entorno `.venv` con las versiones exactas de `uv.lock`. Para los notebooks, usar ese entorno
como kernel.

## Datos

Los datos no están en el repositorio porque pesan cerca de 590 MB (el `.gitignore` excluye los `.parquet`).
Hay que copiar estos tres archivos en `data/raw/`:

| Archivo | Contenido |
|---|---|
| `oids.parquet` | Clase ALeRCE de 9829 objetos |
| `features.parquet` | 178 features de 8419 objetos |
| `lcs.parquet` | Detecciones de las curvas de luz (5189550 filas) |

Los notebooks buscan los datos en `data/raw/` tanto si se abren desde `notebooks/` como desde la raíz del
repositorio.

## Cómo ejecutar

- **Notebooks.** Ejecutar `notebooks/meeting2.ipynb` de principio a fin toma unos 5 minutos en CPU (la
  búsqueda del random forest es lo más lento). Todas las semillas son 42.
- **Pruebas.**

  ```bash
  uv run pytest -q
  ```

  Verifican que no haya fuga de información entre train, val y test, que el pipeline funcione con datos
  sintéticos, que el GMM propio dé lo mismo que scikit-learn partiendo de los mismos parámetros, que la
  pérdida de MCDSVDD sea la ecuación 5 del paper y que en el protocolo la clase anómala no se filtre al
  entrenamiento.
- **Lint.**

  ```bash
  uv run ruff check .
  ```

- **Script.**

  ```bash
  uv run python scripts/run_meeting2.py
  ```

  Hace lo mismo que la primera parte de `notebooks/meeting2.ipynb` (158 features, autoencoder, GMM, MCDSVDD
  con el protocolo del paper e importancia de features) y entrega las mismas métricas en test. Guarda en
  `results/` las figuras, `metricas_test.csv`, `resumen_meeting2.json` y los modelos entrenados. Con
  `--sin-balancear` MCDSVDD promedia sobre objetos en vez de usar la ecuación 5. El análisis de curvas de
  luz (prototipos, outliers y sub-clusters) está solo en el notebook.

## Convenciones del curso

- El `Makefile` y `CONTRIBUTING.md` vienen de la plantilla del curso. En este proyecto se usan los comandos
  con `uv` de arriba.
- Ramas `feat/*`, `fix/*`, `exp/*` o `docs/*`, y commits con formato Conventional Commits.
