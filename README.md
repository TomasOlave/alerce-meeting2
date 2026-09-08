# Plantilla de repositorio — Curso de Inteligencia Computacional
Plantilla base para tareas y proyectos del curso usando Python 3.11+ y `uv`.

## Estructura del repositorio
| Ruta | Propósito |
|---|---|
| `README.md` | Guía de uso y convenciones |
| `pyproject.toml` | Dependencias y configuración de herramientas |
| `.gitignore` | Exclusiones de Git |
| `.env.example` | Variables de entorno de ejemplo |
| `src/` | Código fuente y lógica del curso |
| `scripts/` | Scripts CLI que **solo orquestan** |
| `notebooks/` | Exploración y reportes reproducibles |
| `tests/` | Pruebas automáticas |
| `data/` | Datos crudos/procesados (no versionados) |
| `results/` | Salidas de experimentos (figuras, métricas, modelos) |

## Regla principal: `src/` vs `scripts/` vs `notebooks/`
- `src/`: implementación real (algoritmos, utilidades, carga/procesamiento).
- `scripts/`: entrada por CLI con `argparse`; llaman funciones de `src/`.
- `notebooks/`: análisis interactivo; no contienen lógica crítica duplicada.

## Setup rápido con `uv`
1. Crear entorno e instalar dependencias:
   - `uv sync --dev`
2. Copiar variables de entorno:
   - `cp .env.example .env`
3. (Opcional) activar entorno:
   - `source .venv/bin/activate`

## Cómo correr
- Verificar configuración:
  - `uv run python scripts/check_setup.py`
- Ejecutar ejemplo de pipeline:
  - `uv run python scripts/run_experiment.py --name baseline`

## Calidad y pruebas
- Lint:
  - `uv run ruff check .`
- Formato (si aplica):
  - `uv run ruff format .`
- Tests:
  - `uv run pytest -q`

## Convenciones de código
- Python 3.11+.
- `line-length = 100`.
- Imports ordenados y tipado explícito cuando aporte claridad.
- Evitar lógica en scripts; todo va a `src/`.
- Configuración centralizada en `src/ic_project/config.py`.

## Flujo de Git del curso
- Ramas permitidas: `feat/*`, `fix/*`, `exp/*`, `docs/*`.
- Commits con estilo Conventional Commits (ej: `feat: agrega experimento base`).
- Toda contribución entra por PR revisado.
- Nunca commitear directo a `main`.

## Checklist de entrega
- [ ] `uv sync --dev` ejecuta sin errores.
- [ ] `.env` creado desde `.env.example`.
- [ ] `uv run ruff check .` en verde.
- [ ] `uv run pytest -q` en verde.
- [ ] Scripts en `scripts/` solo orquestan.
- [ ] Lógica implementada en `src/`.
- [ ] PR abierto desde rama `feat|fix|exp|docs`.
