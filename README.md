# AnimaFace — Proyecto Anima

Espacio de rasgos faciales (PCA sobre el Basel Face Model) + puntaje
poligénico sintético + BRIM, con un visor 3D interactivo en el navegador.

Este repo es solo del módulo Rostro.

## Estado del proyecto

### Punto 1 — Espacio de rasgos faciales (COMPLETO)

PCA sobre `model2019_face12.h5` (recorte solo a la región facial: sin
cráneo, sin cuello, sin orejas — 27.657 vértices). 28 componentes retenidas
(95% de la varianza total de forma). Normalización de escala: no se aplica
Procrustes adicional — limitación conocida y documentada, no pendiente de
resolver (ver `CLAUDE.md` para el detalle de la evidencia empírica sobre esa
limitación).

- `espacio_rasgos_rostro.py`: genera/regenera `espacio_rasgos_rostro.npz`.
- `reconstruir_cara(pesos, media, base_pca, varianza)`: reconstruye una cara
  a partir de un vector de pesos en desvíos estándar por componente.
- `visor_rostro_3d.html`: visor 3D interactivo (Three.js), autocontenido.

### Punto 2 — Puntaje poligénico (COMPLETO)

`puntaje_poligenico.py`. En vez de simular las 253 variantes individuales
de Xiong et al. (2025) variante por variante, simula directamente, por cada
una de las 28 componentes PCA, un puntaje poligénico con media 0 y varianza
anclada en la heredabilidad publicada (hasta 7,9% de la varianza del
componente). Decisión de alcance documentada, no un atajo silencioso.

- `generar_genotipo_sintetico(n_personas, heredabilidad_por_componente)`:
  genética por componente (n_personas, 28), en las mismas unidades que
  `pesos` en `reconstruir_cara()`.
- `puntaje_escalar(genetica)`: colapsa la genética por componente en un
  puntaje poligénico escalar por persona (suma con pesos iguales,
  estandarizada). Se agregó para que el Punto 3 (BRIM) tenga un predictor
  escalar — no existía antes.
- `cara_observada(genetica, heredabilidad_por_componente)`: genética + ruido
  ambiental, con varianza total 1 por componente. Se agregó por el mismo
  motivo — es la "cara observada" que BRIM necesita como respuesta.
- Integración en el visor: botón **"Generar genotipo aleatorio"** en
  `visor_rostro_3d.html` (muestra el mismo criterio de
  `generar_genotipo_sintetico`, implementado en JS).

### Punto 3 — BRIM (COMPLETO)

`brim_rostro.py`. Conecta el puntaje poligénico (Punto 2) con el espacio de
rasgos (Punto 1): BRIM (Bootstrapped Response-based Imputation Modeling,
Claes & Shriver 2014, PLOS Genetics) calcula, por persona, el RIP
(Response-based Imputed Predictor) — "cuánto puntaje genético sugiere su
cara" — sin usar el propio dato de esa persona (leave-one-out), con
incertidumbre estimada por bootstrap.

**Cohorte oficial:** `cohorte_oficial()` arma `(puntaje, pesos)` con la
salida real del Punto 2: `generar_genotipo_sintetico()` da la genética,
`puntaje_escalar()` da el predictor y `cara_observada()` da la respuesta.
`simular_cohorte()` (ya existía en el diseño original de BRIM) queda solo
como cohorte **alternativa para pruebas** que la cohorte oficial no puede
producir limpio:
- el control "sin señal genética" (con h²=0, `generar_genotipo_sintetico()`
  da genética idénticamente cero, y `puntaje_escalar()` no puede
  estandarizar una suma constante);
- el barrido de sensibilidad a techos de heredabilidad por encima del 7,9%
  (ese techo es una validación del Punto 2, no una limitación de BRIM).

**Pipeline:** PLSR (cantidad de componentes elegida por validación cruzada
de 10 pliegues) → RIP leave-one-out → bootstrap de 300 remuestreos
(estabilidad de pesos + incertidumbre del RIP por persona) → R² fuera de
muestra con IC95% bootstrap.

#### El R² varía de corrida a corrida — número reproducible

Una sola semilla no alcanza para citar un R² en la tesis. `barrido_semillas_
brim.py` corre la corrida principal y el control de permutación con 20
semillas distintas:

| | media | mínimo | máximo |
|---|---|---|---|
| R² honesto (LOO), corrida principal | 2,25% | -2,81% | 10,04% |
| R² con el puntaje mezclado (permutación) | -5,41% | -7,60% | -1,67% |

(n = 503, h² = 7,9% en las 28 componentes; medido el 2026-10-12.)

#### Controles de sanidad (`tests/test_brim.py`, pytest)

- `puntaje_escalar` queda estandarizado exacto (media 0, desvío 1).
- `cara_observada` da varianza ~1 por componente.
- Con el puntaje mezclado entre personas, R² LOO < 3%.
- R² "con trampa" (sin leave-one-out) > R² honesto.
- R² honesto de la corrida principal entre -5% y 10%.
- Sin señal genética (`simular_cohorte(sin_senal=True)`), R² LOO < 5%.

Tolerancias anchas a propósito, por la variabilidad entre semillas de la
tabla anterior — estos tests verifican el patrón esperado, no un número
exacto de una sola corrida.

#### Parámetros editables (`brim_rostro.py`)

- `N_PERSONAS` (503, subpoblación europea de 1000 Genomes). Subirlo (1000,
  2504, 5000...) recupera más señal — es decisión del proyecto/tutor, no se
  cambia sin consultar.
- `R2_TOTAL`: techo de varianza para `simular_cohorte()` (barrido de
  sensibilidad), independiente del h² de Punto 2.
- `N_BOOTSTRAP` (300): remuestreos para estabilidad de coeficientes e
  incertidumbre del RIP.
- `MAX_COMP_PLS` (10): máximo de componentes PLS a probar en la validación
  cruzada.

#### Limitaciones documentadas (no pendientes de resolver)

- Las componentes PLS se eligen por validación cruzada sobre **todos** los
  datos (leve optimismo posible en el número elegido, no en el R² final que
  sí es leave-one-out).
- No se controla por sexo ni ancestría.
- `puntaje_escalar()` usa pesos iguales entre las 28 componentes — decisión
  de diseño documentada, no un valor medido empíricamente.
- Aplicar el techo de heredabilidad (7,9%, Xiong et al. 2025) a las 28
  componentes a la vez es generoso: en el paper es el máximo reportado para
  UN rasgo, no el valor de todos simultáneamente. El parámetro
  `heredabilidad_por_componente` acepta un array con un h² distinto por
  componente cuando se tengan esos valores.

## Cómo correr

### El visor

`visor_rostro_3d.html` es autocontenido (datos embebidos en base64, Three.js
desde CDN) — no necesita servidor ni backend. Se abre directamente con
doble click o "Abrir con" el navegador.

### Pipeline completo (Puntos 1, 2 y 3 en un solo comando)

```
pip install -r requirements.txt

python correr_pipeline.py          # barrido de 20 semillas (el que se cita)
python correr_pipeline.py --rapido # barrido de 5 semillas, para desarrollo
```

`correr_pipeline.py` corre, en orden, las verificaciones de Punto 1, la
cohorte de Punto 2, la corrida de BRIM (Punto 3), el barrido de semillas y
los tests -- reutilizando las funciones de cada módulo, sin reimplementar
nada. Si una etapa falla, se marca y el pipeline sigue con las demás. Si
`model2019_face12.h5` no está en la carpeta (no lo está en este repo), lo
avisa y verifica solo contra el `.npz` -- sin el `.h5` no se puede comprobar
el "95% de la varianza total" porque el `.npz` solo guarda las 28
componentes ya retenidas, no el total del modelo.

Salida:
- consola, con marca `OK` / `REVISAR` / `N/D` / `FALLÓ` por verificación;
- `reporte_pipeline.txt`: el mismo resumen, con fecha de la corrida;
- `reporte_pipeline.png`: panel de 4 gráficos (varianza acumulada, puntaje
  escalar de la cohorte, RIP vs. puntaje real, barrido de semillas).

Tiene lugares reservados (sin implementar) para el Punto 4 (calibración) y
el Punto 5 del proyecto.

### BRIM (Punto 3) por separado

```
python brim_rostro.py              # corrida principal + controles de sanidad
                                    # exporta brim_rostro_resultados.npz y brim_rostro.png

python barrido_semillas_brim.py    # tabla de R² en 20 semillas

python -m pytest tests/            # tests de sanidad
```

`espacio_rasgos_rostro.py` y `puntaje_poligenico.py` no se ejecutan en
tiempo de uso del visor ni de BRIM; son los scripts que generaron/generan
los datos (el `.npz` de Punto 1) y quedan como referencia.

## Requisitos

Python 3.11 (ver `.python-version`). Dependencias en `requirements.txt`:
numpy, scikit-learn, h5py, matplotlib, pytest.
