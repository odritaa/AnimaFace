# Punto 1 — Espacio de rasgos faciales (PCA sobre el Basel Face Model)

Este es el Paso 1 del pipeline del módulo Rostro: construir un espacio de
rasgos faciales de baja dimensión, sobre el que después se van a apoyar el
puntaje poligénico (Paso 2), BRIM (Paso 3) y la calibración por conformal
prediction (Paso 4). Es estadística pura (PCA) — ningún LLM interviene en
este paso ni en ninguno de los pasos 1 a 5 del pipeline.

## Archivos

- `espacio_rasgos_rostro.py` — funciones para explorar el archivo `.h5` del
  Basel Face Model, cargarlo, reconstruir una cara desde un vector de pesos,
  verificar esa reconstrucción, y exportar la versión liviana en `.npz`.
  También incluye una demo con datos sintéticos (`demo_sintetica`), usada
  para validar la lógica de PCA antes de tener el archivo real del modelo.
- `espacio_rasgos_rostro.npz` — el espacio de rasgos ya exportado, listo
  para que el resto del pipeline lo cargue sin depender del `.h5` completo.
  Contiene:
  - `media`: cara promedio del modelo, `(142317,)` = 47439 puntos × 3 (x, y, z).
  - `base_pca`: base de componentes principales, `(142317, 31)`.
  - `varianza`: varianza de cada una de las 31 componentes.
  - `n_componentes`: `31`.

## Decisiones tomadas

**Archivo base: `model2019_bfm.h5`.**
Se usó la máscara clásica del Basel Face Model 2019, no la versión
full-head ni la face12. Esto evita modelar cuello y parte trasera de la
cabeza con heurísticas del propio modelo (zonas que el BFM no captura desde
escaneos reales), y evita sumar un archivo adicional al proyecto sin una
necesidad concreta para el pipeline.

**Componentes retenidas: 31.**
Cubren el 95% de la varianza total de forma del modelo. Es el criterio de
corte estándar para retener un espacio PCA que sea manejable en los pasos
siguientes (puntaje poligénico, BRIM) sin perder la mayor parte de la
información morfológica.

**Normalización de escala: no aplicada.**
No se corre un Procrustes adicional sobre el espacio del modelo — se usa
tal como viene el BFM. Esto es una limitación conocida y documentada, no un
problema pendiente de resolver: el propio marco teórico de la tesis ya
registra esta decisión en la sección de Limitaciones. Se deja constancia
acá para que quede visible junto al código que la implementa.

## Verificación de la reconstrucción

`verificar_reconstruccion()` corre dos chequeos de sanidad sobre el espacio
exportado, antes de que el resto del pipeline confíe en él:

1. Pesos en cero deben devolver exactamente la cara promedio (`media`).
2. Una variación moderada (2 desvíos estándar en PC1) debe producir un
   desplazamiento de unos pocos milímetros, no algo desproporcionado.

## Qué no es este módulo (todavía)

Este paso no calcula puntaje poligénico ni corre BRIM ni calibración —
esos son los Pasos 2 a 4, pendientes. Tampoco hay ningún LLM leyendo ni
escribiendo nada acá: cuando el pipeline llegue al Paso 5 (salida
calibrada), un LLM podría eventualmente redactar esa salida en lenguaje
natural, pero siempre desde afuera, leyendo un JSON ya calculado — nunca
tocando el espacio de rasgos ni ningún cálculo de los pasos 1 a 5.
