# AnimaFace — Proyecto Anima

Espacio de rasgos faciales (PCA sobre el Basel Face Model) + puntaje
poligénico sintético, con un visor 3D interactivo en el navegador.

## Estado del proyecto

### Punto 1 — Espacio de rasgos faciales (COMPLETO)

- Archivos: `espacio_rasgos_rostro.py`, `espacio_rasgos_rostro.npz`, `visor_rostro_3d.html`
- PCA sobre `model2019_face12.h5` (recorte solo a la región facial: sin
  cráneo, sin cuello, sin orejas — más alineado con la hipótesis del
  proyecto sobre morfología facial específicamente). 27.657 vértices.
- 28 componentes retenidas (siguen cubriendo el 95% de la varianza total de
  forma; el número cambia respecto del recorte anterior por la nueva malla).
- **Decisión corregida:** antes se usaba `model2019_bfm.h5` (47.439 vértices)
  con 31 componentes. Se cambió a face12 por alineación con la hipótesis
  del proyecto (morfología facial específica).
- Normalización de escala: no se aplica Procrustes adicional sobre el
  espacio del modelo — limitación conocida y documentada, no pendiente de
  resolver.
- **Evidencia empírica sobre esa limitación:** la dominancia de la primera
  componente (~48% de la varianza) es prácticamente idéntica entre fullHead,
  bfm y face12. O sea que NO se debe a incluir cráneo/cuello/orejas, sino a
  que el modelo no normaliza escala antes del PCA: es una propiedad del
  modelo, no un artefacto de la máscara elegida.
- `reconstruir_cara(pesos, media, base_pca, varianza)`: reconstruye una cara
  a partir de un vector de pesos, interpretados en desvíos estándar de cada
  componente (pesos=0 en todas las componentes devuelve la cara promedio).
- `visor_rostro_3d.html`: visor 3D interactivo (Three.js) con los datos
  reales del modelo embebidos en base64 (media, base_pca, varianza y
  topología de la malla). Sliders manuales de -3 a +3 desvíos estándar por
  componente.

### Punto 2 — Puntaje poligénico (COMPLETO)

- Archivo: `puntaje_poligenico.py`
- **Decisión de alcance documentada:** en vez de simular las 253 variantes
  individuales de Xiong et al. (2025), se simula directamente, por cada una
  de las 28 componentes PCA, un puntaje poligénico como variable aleatoria
  con media 0 y varianza anclada en la heredabilidad ya publicada (hasta
  7,9% de la varianza total del componente, según el paper). Esto es una
  simplificación metodológica documentada, no un error — reemplaza la
  simulación SNP por SNP porque no tenemos datos reales emparejados a nivel
  individual, igual que ya está justificado en el resto del proyecto.
- `generar_genotipo_sintetico(n_personas, heredabilidad_por_componente)`
  devuelve, por persona simulada, un puntaje poligénico por componente, en
  las mismas unidades que `pesos` en `reconstruir_cara()` (desvíos estándar
  del componente) — se puede usar directamente sin reescalar.
- Integración en el visor: botón **"Generar genotipo aleatorio"** en
  `visor_rostro_3d.html`, al lado del botón de reset. Samplea un puntaje
  poligénico por componente (mismo criterio que
  `generar_genotipo_sintetico`, implementado en JS para no depender de un
  backend) y mueve los sliders automáticamente. Los sliders manuales siguen
  funcionando igual que antes — esto se sumó como capa nueva, sin tocar la
  lógica del Punto 1 (PCA, reconstrucción, verificación).
- **Completado para conectar con el Punto 3 (BRIM):** `generar_genotipo_
  sintetico()` por sí solo no alcanza para BRIM — da la genética por
  componente (n, 28), pero BRIM necesita además un puntaje poligénico
  ESCALAR y una cara OBSERVADA (genética + ruido ambiental). Se agregaron
  dos funciones:
  - `puntaje_escalar(genetica)`: suma con pesos iguales entre componentes,
    estandarizada (media 0, desvío 1) — el puntaje poligénico escalar por
    persona.
  - `cara_observada(genetica, heredabilidad_por_componente)`: genética +
    ruido ambiental, con varianza total 1 por componente (misma convención
    que `pesos` en `reconstruir_cara()`).

### Punto 3 — BRIM (COMPLETO)

- Archivos: `brim_rostro.py`, `barrido_semillas_brim.py`, `tests/test_brim.py`
- Referencia: Claes, P., & Shriver, M. D. (2014), PLOS Genetics.
- Conecta el puntaje poligénico (Punto 2) con el espacio de rasgos (Punto 1):
  por cada persona calcula el RIP (Response-based Imputed Predictor),
  "cuánto puntaje genético sugiere su cara", con PLSR + leave-one-out +
  bootstrap (300 remuestreos).
- **Cohorte oficial:** `cohorte_oficial()` en `brim_rostro.py` arma
  `(puntaje, pesos)` con la salida real del Punto 2 (`generar_genotipo_
  sintetico()` + `puntaje_escalar()` + `cara_observada()`), no con datos
  inventados para BRIM.
- **`simular_cohorte()` queda solo como cohorte alternativa para pruebas**
  que la cohorte oficial no puede producir limpio: el control "sin señal
  genética" (h²=0 da genética idénticamente cero, y `puntaje_escalar()` no
  puede estandarizar una suma constante) y el barrido de sensibilidad a
  techos de heredabilidad por encima del 7,9% (ese techo es una validación
  del Punto 2, no de BRIM).
- Controles de sanidad (`tests/test_brim.py`, pytest): R² con el puntaje
  mezclado < 3%; R² "con trampa" (sin leave-one-out) mayor que el honesto;
  R² honesto de la corrida principal entre -5% y 10%; sin señal genética,
  R² < 5%; `puntaje_escalar` estandarizado exacto; `cara_observada` con
  varianza ~1 por componente. Tolerancias anchas a propósito — ver el punto
  siguiente.
- **El R² honesto varía de corrida a corrida según la semilla** (en 20
  semillas: de -2,8% a 10,0%, media 2,25%; la permutación varió de -7,6% a
  -1,7%, media -5,41%). `barrido_semillas_brim.py` mide esto con 20 semillas
  y es el número que se cita en la tesis — no el de una semilla suelta.
- Parámetros editables en `brim_rostro.py`: `N_PERSONAS` (503, subpoblación
  europea de 1000 Genomes — cambiarlo es decisión del proyecto/tutor, no se
  cambia solo), `R2_TOTAL` (techo para `simular_cohorte()`), `N_BOOTSTRAP`,
  `MAX_COMP_PLS`.
- **Limitaciones documentadas, no pendientes de resolver:**
  - Las componentes PLS se eligen por validación cruzada sobre todos los
    datos (leve optimismo posible en el número elegido).
  - No se controla por sexo ni ancestría.
  - El puntaje escalar (`puntaje_escalar()`) usa pesos iguales entre las 28
    componentes — decisión de diseño documentada, no un valor medido.
  - Aplicar el techo de heredabilidad (7,9%, Xiong et al. 2025) a las 28
    componentes a la vez es generoso: en el paper es el máximo reportado
    para UN rasgo, no el valor de todos simultáneamente.

## Cómo correr el visor

`visor_rostro_3d.html` es autocontenido (datos embebidos en base64, Three.js
desde CDN) — no necesita servidor ni backend. Se abre directamente con
doble click o "Abrir con" el navegador.

`espacio_rasgos_rostro.py` y `puntaje_poligenico.py` no se ejecutan en tiempo
de uso del visor; son los scripts que generaron/generan los datos y quedan
como referencia y para regenerar el `.npz` si cambia el modelo o la cantidad
de componentes retenidas.

## Cómo correr BRIM (Punto 3)

```
pip install -r requirements.txt
python brim_rostro.py              # corrida principal + controles de sanidad
python barrido_semillas_brim.py    # R² honesto/permutación en 20 semillas
python -m pytest tests/            # tests de sanidad
```

`brim_rostro.py` exporta `brim_rostro_resultados.npz` y `brim_rostro.png` en
la raíz del repo.
