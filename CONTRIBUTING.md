# Cómo trabajar en este repositorio

## Flujo de trabajo

```bash
git switch -c step-03-red-neuronal   # una rama por tarea o experimento
# ... trabajas ...
make calidad                            # lint + tipos + pruebas
git add -p && git commit
git push -u origin step-03-red-neuronal 
```

Abre un pull request usando la plantilla. Si trabajas en pareja, cada uno revisa el PR del otro antes de mezclar: es la forma más rápida de encontrar errores.

## Mensajes de commit

Formato [Conventional Commits](https://www.conventionalcommits.org/), en
imperativo y en español:

```text
feat(difusa): agregar defuzzificacion por bisector
fix(mlp): corregir el gradiente de la capa de salida en multiclase
test(pso): cubrir el caso de limites invertidos
docs(readme): documentar el comando de auditoria
refactor(utils): extraer la validacion de vectores
```

Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
El alcance (`difusa`, `mlp`, `pso`, ...) es el módulo tocado.

## Estilo de código

Lo automático lo resuelven Ruff y `pre-commit`; lo que sigue es lo que revisa una
persona:

- **Nombres en español**, consistentes con los apuntes: `ajustar`, `predecir`,
  `puntuar`, `tasa_aprendizaje`, `n_epocas`. Los atributos aprendidos terminan en
  guion bajo (`pesos_`, `errores_`), como en scikit-learn.
- **Anotaciones de tipo** en todo lo público. `mypy` corre sobre `src/`.
- **Docstrings** con una línea de resumen y, cuando aporte, secciones `Args`,
  `Returns` y `Ejemplo`. Documenta los supuestos: forma de los arreglos, rangos
  válidos, qué se asume de `y`.
- **Comentarios que expliquen el por qué**, no el qué. Si el comentario repite el
  código, sobra.
- **Errores explícitos y temprano.** Valida los hiperparámetros en `__init__` con
  `ValueError` y un mensaje que diga qué se esperaba.
- **Nada de `np.random.seed`.** Pide un generador con
  `icomp.config.generador(semilla)`.
- **Rutas desde `icomp.config`**, nunca rutas absolutas ni relativas al cwd.

## Pruebas

- Una prueba por comportamiento, con nombre que describa la expectativa:
  `test_no_resuelve_compuertas_no_separables`.
- Cubre el camino feliz, los casos borde y los errores esperados
  (`pytest.raises`).
- Las pruebas deben ser deterministas: semilla fija siempre.
- Marca lo que tarde más de un segundo con `@pytest.mark.lento`.
- Contra el azar, prueba propiedades (monotonía, cotas, invariantes) en lugar de
  números exactos que dependan de la versión de NumPy.

```bash
pytest                              # todo
pytest tests/test_difusa.py -v      # un archivo
pytest -k "perceptron and no"       # por nombre
pytest -m "not lento"               # sin las lentas
make cobertura                      # reporte en htmlcov/index.html
```

## Agregar un algoritmo nuevo

1. Ubícalo en el subpaquete de su unidad (o crea uno nuevo con `__init__.py`).
2. Expórtalo en el `__all__` del subpaquete.
3. Escribe las pruebas en `tests/test_<unidad>.py`.
4. Agrega una demostración en `icomp/cli.py` (`DEMOS`) si es un algoritmo central.
5. Menciónalo en `docs/programa.md` y en la tabla del `README.md`.

## Seguridad y datos

- **Nunca** subas contraseñas, tokens, claves de API ni archivos `.env`. Si se
  filtra algo, avisa: rotar la credencial es lo primero, borrar el commit no basta.
- No subas datos personales, de clientes ni de terceros. Los ejemplos usan datos
  sintéticos y ficticios.
- Antes de agregar una dependencia: ¿está mantenida (commits recientes, releases,
  issues atendidos)? ¿se puede resolver con la librería estándar o con NumPy?
  Justifícalo en el PR y corre `make auditoria`.
- No deserialices `pickle` de origen desconocido: ejecuta código al cargarse.

## Uso de asistentes de IA

Están permitidos como herramienta de apoyo, con dos condiciones:

1. **Entiendes lo que entregas.** Debes poder explicar cada línea en la defensa.
2. **Lo declaras.** Indica en el informe qué parte se apoyó en un asistente y
   cómo verificaste el resultado.

Copiar una solución sin entenderla es lo mismo que copiarla de un compañero.

## Revisión de un PR

- ¿Las pruebas cubren el cambio y fallarían si se revierte?
- ¿Los nombres dicen lo que hacen?
- ¿Hay validación de entradas y mensajes de error útiles?
- ¿La documentación quedó al día (README, programa, docstrings)?
- ¿Se puede reproducir el resultado con la semilla indicada?
