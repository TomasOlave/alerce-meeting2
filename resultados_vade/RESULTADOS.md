# Resultados: VaDE, el modelo con GMM del repositorio del paper

**Qué es.** El repositorio del paper ([AnomalyALeRCE](https://github.com/mperezcarrasco/AnomalyALeRCE))
trae un solo modelo con mezcla de gaussianas: VaDE (`src/models/VaDE.py`). Es un autoencoder variacional
cuyo espacio latente sigue un GMM; la red y el GMM se entrenan juntos. Allá quedó incompleto (no tiene
puntaje de anomalía, usa un atributo que no existe y da 0/0 al calcular las pertenencias). Se completó y
corrigió en `src/ic_project/vade.py`, con su misma arquitectura y su misma pérdida (−ELBO).
Se probaron dos versiones: sin clases (5 componentes, como el GMM) y con un componente por clase conocida.

**Datos.** 8419 estrellas (EB 2689, RRL 2652, LPV 1374, CEP 1172, DSCT 532), 140 features, latente de 10
dimensiones, semilla 42.

**Cómo se corrió.** En una máquina sin PyTorch: el autoencoder, MCDSVDD y VaDE se reimplementaron en NumPy
(`scripts/sin_pytorch/`), con la misma arquitectura e hiperparámetros, y los gradientes se verificaron con
diferencias finitas. Cuadra con los resultados en PyTorch de Gabriel: MCDSVDD da ARI 0.774 en test (antes
0.775) y AUROC del protocolo 0.871 / 0.756 / 0.814 / 0.773 / 0.804 (antes 0.860 / 0.740 / 0.812 / 0.799 /
0.827).

## 1. Detección de anomalías: protocolo del paper

Una clase escondida, 5 folds, TS2 con 10% de anomalías. Media ± desviación sobre los folds.

### AUROC (azar = 0.5)

| Clase escondida | MCDSVDD | Autoencoder | GMM | VaDE | VaDE con clases |
|---|---|---|---|---|---|
| CEP | **0.871 ± 0.015** | 0.625 ± 0.009 | 0.463 ± 0.015 | 0.616 ± 0.005 | 0.617 ± 0.004 |
| DSCT | **0.756 ± 0.025** | 0.566 ± 0.004 | 0.422 ± 0.019 | 0.551 ± 0.002 | 0.550 ± 0.002 |
| EB | **0.814 ± 0.014** | 0.738 ± 0.006 | 0.612 ± 0.024 | 0.729 ± 0.004 | 0.730 ± 0.005 |
| LPV | 0.773 ± 0.028 | 0.971 ± 0.006 | 0.965 ± 0.006 | **0.972 ± 0.004** | **0.972 ± 0.004** |
| RRL | **0.804 ± 0.024** | 0.445 ± 0.008 | 0.415 ± 0.023 | 0.430 ± 0.004 | 0.433 ± 0.005 |
| **Promedio** | **0.803** | 0.669 | 0.576 | 0.660 | 0.660 |

### AUPRC (azar = 0.1)

| Clase escondida | MCDSVDD | Autoencoder | GMM | VaDE | VaDE con clases |
|---|---|---|---|---|---|
| CEP | **0.434** | 0.135 | 0.094 | 0.133 | 0.132 |
| DSCT | **0.191** | 0.113 | 0.081 | 0.109 | 0.108 |
| EB | **0.260** | 0.187 | 0.140 | 0.183 | 0.183 |
| LPV | 0.232 | 0.840 | **0.846** | 0.837 | 0.842 |
| RRL | **0.221** | 0.084 | 0.081 | 0.081 | 0.082 |
| **Promedio** | 0.268 | **0.271** | 0.248 | 0.268 | 0.270 |

## 2. Clustering en test (split 70/15/15)

| Modelo | ARI | NMI | F1 macro |
|---|---|---|---|
| GMM | 0.297 | 0.307 | 0.396 |
| VaDE | 0.241 | 0.304 | 0.404 |
| VaDE con clases* | 0.481 | 0.459 | 0.658 |
| MCDSVDD* (referencia) | 0.774 | 0.717 | 0.879 |

\* Usan las clases al entrenar; no son clustering sin etiquetas. En los modelos sin etiquetas, el F1
asigna a cada grupo la clase mayoritaria de sus estrellas de train.

## 3. Lectura

1. **VaDE mejora al GMM como detector de anomalías**: AUROC promedio 0.660 contra 0.576, y sube en las
   cinco clases (CEP 0.46 → 0.62, DSCT 0.42 → 0.55, EB 0.61 → 0.73).
2. **Pero no supera al autoencoder** (0.669) y se comporta igual que él, clase por clase. Su puntaje
   (−ELBO) suma el error de reconstrucción de 140 features, y ese término domina. Hereda la falla del
   autoencoder: ve más normales a las RRL que a las conocidas (0.43).
3. **Usar las clases en VaDE no cambia nada** (0.660 en las dos versiones), porque el puntaje lo domina la
   reconstrucción y en test la clase no se conoce.
4. **Como clustering sin etiquetas, VaDE no mejora al GMM** (ARI 0.24 contra 0.30).
5. **MCDSVDD sigue siendo el mejor detector** (0.803), salvo en LPV, que los modelos de reconstrucción y
   densidad detectan casi perfecto (~0.97).

**Siguiente paso posible:** usar como puntaje de VaDE solo la parte del GMM (−log p(z) en el latente, ya
calculada en `salidas_vade` como `energia_latente`), sin la reconstrucción.
