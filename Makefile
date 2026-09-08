# Atajos del proyecto. Ejecuta `make ayuda` para ver los objetivos disponibles.
.DEFAULT_GOAL := ayuda
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: ayuda instalar entorno pruebas cobertura lint formato tipos auditoria calidad notebook limpiar

ayuda: ## Muestra esta ayuda
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

entorno: ## Crea el entorno virtual .venv
	$(PY) -m venv $(VENV)

instalar: entorno ## Instala el paquete en modo editable con extras de desarrollo
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev,notebooks]"
	$(BIN)/pre-commit install

pruebas: ## Ejecuta la bateria de pruebas
	$(BIN)/pytest

cobertura: ## Pruebas con reporte de cobertura
	$(BIN)/pytest --cov=icomp --cov-report=term-missing --cov-report=html

lint: ## Revisa estilo y errores estaticos
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

formato: ## Formatea y arregla lo que se pueda automaticamente
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/ruff format src tests scripts

tipos: ## Verifica anotaciones de tipo
	$(BIN)/mypy

auditoria: ## Revisa vulnerabilidades conocidas en las dependencias
	$(BIN)/pip-audit --skip-editable

calidad: lint tipos pruebas ## Todo lo anterior (lo mismo que corre la CI)

notebook: ## Abre JupyterLab
	$(BIN)/jupyter lab

limpiar: ## Borra artefactos de compilacion y caches
	rm -rf build dist htmlcov .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
