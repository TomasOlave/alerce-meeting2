"""Carga los datos sin pyarrow (lector propio en parquet_puro.py) y deja src/ en el path."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ic_project.datos import unir_features_etiquetas  # noqa: E402
from parquet_puro import leer_parquet  # noqa: E402


def cargar():
    datos = RAIZ / "data" / "raw"
    f = leer_parquet(datos / "features.parquet")
    o = leer_parquet(datos / "oids.parquet")
    return unir_features_etiquetas(f, o)
