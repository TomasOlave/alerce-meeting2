"""Selección / ranking de features con Random Forest (solo train)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def importancia_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    nombres: list[str],
    semilla: int = 42,
    n_estimadores: int = 200,
) -> pd.DataFrame:
    clf = RandomForestClassifier(
        n_estimators=n_estimadores,
        max_depth=16,
        class_weight="balanced",
        random_state=semilla,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    tabla = pd.DataFrame({"feature": nombres, "importancia": clf.feature_importances_}).sort_values(
        "importancia", ascending=False
    )
    return tabla.reset_index(drop=True)
