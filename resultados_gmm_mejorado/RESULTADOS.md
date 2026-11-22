# Resultados: GMM mejorado (GMM por clase y VaDE)

Datos: `features.parquet` y `oids.parquet` del proyecto (8419 estrellas: EB 2689, RRL 2652, LPV 1374,
CEP 1172, DSCT 532). 140 features (FEATURES_EXCLUIDAS), latente de 10 dimensiones, semilla 42.

**Cómo se corrió.** En una máquina sin PyTorch ni pyarrow. Por eso el autoencoder, MCDSVDD y VaDE se
reimplementaron en NumPy con backprop a mano (`scripts/sin_pytorch/`), con la misma arquitectura,
optimizador e hiperparámetros que `src/ic_project/modelos.py` y `vade.py`. Los gradientes se validaron
con diferencias finitas (`check_grad.py`). El GMM, el GMM por clase, el preprocesamiento, la partición y
las métricas son el código del repositorio. En VaDE sin clases, gamma = p(c|z) se trata como constante en
el gradiente (paso E); la pérdida es la misma.

**Validación contra los resultados de Gabriel (PyTorch).** MCDSVDD en test: ARI 0.774 (Gabriel 0.775).
AUROC de MCDSVDD en el protocolo: 0.871 / 0.756 / 0.814 / 0.773 / 0.804 (Gabriel 0.860 / 0.740 / 0.812 /
0.799 / 0.827). Los números cuadran; las diferencias son del orden de la variación entre folds.

## 1. Detección de anomalías: protocolo del paper

Una clase escondida, 5 folds, TS2 con 10% de anomalías. Media ± desviación sobre los folds.

### AUROC (azar = 0.5)

| Clase escondida | MCDSVDD | Autoencoder | GMM | GMM por clase (AE) | **GMM por clase (MCDSVDD)** | VaDE | VaDE con clases |
|---|---|---|---|---|---|---|---|
| CEP | **0.871 ± 0.015** | 0.625 ± 0.009 | 0.463 ± 0.015 | 0.492 ± 0.013 | 0.863 ± 0.033 | 0.616 ± 0.005 | 0.617 ± 0.004 |
| DSCT | **0.756 ± 0.025** | 0.566 ± 0.004 | 0.422 ± 0.019 | 0.483 ± 0.019 | 0.750 ± 0.040 | 0.551 ± 0.002 | 0.550 ± 0.002 |
| EB | **0.814 ± 0.014** | 0.738 ± 0.006 | 0.612 ± 0.024 | 0.656 ± 0.017 | 0.764 ± 0.027 | 0.729 ± 0.004 | 0.730 ± 0.005 |
| LPV | 0.773 ± 0.028 | 0.971 ± 0.006 | 0.965 ± 0.006 | 0.966 ± 0.004 | 0.894 ± 0.038 | **0.972 ± 0.004** | **0.972 ± 0.004** |
| RRL | **0.804 ± 0.024** | 0.445 ± 0.008 | 0.415 ± 0.023 | 0.438 ± 0.018 | 0.716 ± 0.018 | 0.430 ± 0.004 | 0.433 ± 0.005 |
| **Promedio** | **0.803** | 0.669 | 0.576 | 0.607 | **0.797** | 0.660 | 0.660 |

### AUPRC (azar = 0.1)

| Clase escondida | MCDSVDD | Autoencoder | GMM | GMM por clase (AE) | **GMM por clase (MCDSVDD)** | VaDE | VaDE con clases |
|---|---|---|---|---|---|---|---|
| CEP | **0.434** | 0.135 | 0.094 | 0.101 | 0.398 | 0.133 | 0.132 |
| DSCT | **0.191** | 0.113 | 0.081 | 0.091 | 0.184 | 0.109 | 0.108 |
| EB | **0.260** | 0.187 | 0.140 | 0.178 | 0.240 | 0.183 | 0.183 |
| LPV | 0.232 | 0.840 | **0.846** | 0.840 | 0.602 | 0.837 | 0.842 |
| RRL | **0.221** | 0.084 | 0.081 | 0.086 | 0.163 | 0.081 | 0.082 |
| **Promedio** | 0.268 | 0.271 | 0.248 | 0.259 | **0.317** | 0.268 | 0.270 |

Componentes elegidos por BIC (moda sobre folds), latente de MCDSVDD: CEP 2, DSCT 1–2, EB 2–3, LPV 2,
RRL 2–3. En el latente del autoencoder salen más (LPV 4, EB 3–4).

El puntaje "mezcla" (−log Σ_c π_c p(z|c)) da casi lo mismo que "max": AUROC promedio 0.798.

## 2. Clustering en test (split 70/15/15)

| Modelo | ARI | NMI | F1 macro | Silhouette (grupos) |
|---|---|---|---|---|
| GMM sin etiquetas | 0.297 | 0.307 | 0.396 | 0.115 |
| GMM por clase (latente AE) | 0.578 | 0.516 | 0.739 | 0.016 |
| **GMM por clase (latente MCDSVDD)** | **0.772** | **0.719** | **0.877** | 0.593 |
| MCDSVDD (centro más cercano) | 0.774 | 0.717 | 0.879 | 0.614 |
| VaDE sin etiquetas | 0.241 | 0.304 | 0.404 | 0.146 |
| VaDE con clases | 0.481 | 0.459 | 0.658 | 0.076 |

En los modelos sin etiquetas, el F1 asigna a cada grupo la clase mayoritaria de sus estrellas de train.

## 3. Lectura

1. **El GMM por clase sobre el latente de MCDSVDD arregla el GMM.** El AUROC promedio sube de 0.576 a
   0.797, casi igual que MCDSVDD (0.803), y nunca cae bajo el azar. Tiene el mejor AUPRC promedio de
   todos (0.317 contra 0.268 de MCDSVDD).
2. **Le gana a MCDSVDD en LPV** (0.894 contra 0.773, en los 5 folds). MCDSVDD es el detector más débil
   para LPV; el GMM por clase, al tener covarianza propia por clase, la reconoce mejor. Empata en CEP y
   DSCT (gana en 3 de 5 folds cada una) y pierde en EB (0.76 contra 0.81) y RRL (0.72 contra 0.80).
3. **Las etiquetas solas no bastan: importa el espacio.** El mismo GMM por clase sobre el latente del
   autoencoder casi no mejora (0.607). Ahí las clases se pisan y ni con etiquetas se separan. El
   latente de MCDSVDD sí agrupa por clase, y sobre él una mezcla gaussiana por clase es un buen modelo
   de densidad.
4. **VaDE se comporta como el autoencoder** (AUROC 0.660 contra 0.669, mismo patrón por clase). Su
   puntaje, −ELBO, queda dominado por el error de reconstrucción (140 features sumadas), así que
   hereda las fallas del autoencoder: ve normales a las RRL (0.43). Las clases no cambian esto, porque en
   test la clase no se conoce. Como clustering, VaDE con clases (ARI 0.48) queda bajo el GMM por clase.
5. **Recomendación:** usar el GMM por clase sobre el latente de MCDSVDD como la versión con GMM del
   detector. Iguala a MCDSVDD en promedio, lo complementa en LPV y da probabilidades en vez de
   distancias.
