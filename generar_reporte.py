"""
Módulo Rostro — Generador del reporte HTML (Puntos 1 a 3 en una sola página)
=============================================================================

Arma reporte/reporte_rostro.html a partir de la plantilla reporte/plantilla.html,
reemplazando sus dos marcadores (ROSTRO_DATA, RESULTADOS) con datos reales:
no reimplementa nada de los Puntos 1, 2 o 3 -- reutiliza las funciones ya
existentes de espacio_rasgos_rostro.py, puntaje_poligenico.py y brim_rostro.py.

El resultado es un único archivo HTML autocontenido (Three.js se carga desde
CDN, los datos van embebidos en base64), para abrir con doble click, sin
servidor ni imágenes sueltas.

ROSTRO_DATA
-----------
Sale de espacio_rasgos_rostro.npz (media + 28 componentes PCA ya retenidas)
y de model2019_face12.h5 (triángulos de la malla y varianza de las 199
componentes del modelo completo -- el .npz solo guarda las 28 ya retenidas,
no alcanza para saber qué % del total representan ni para dibujar la malla).
La base PCA se cuantiza a int16 por componente (igual que ya hacía
visor_rostro_3d.html) para que las 28 componentes x 27.657 vértices entren
en un HTML liviano; el error que introduce se verifica más abajo contra
reconstruir_cara().

RESULTADOS
----------
Sale de correr una corrida real del pipeline: cohorte_oficial() + correr_brim()
(Punto 3) sobre la cohorte real del Punto 2, más el control de permutación y
el barrido de 20 semillas (barrido_semillas(), mismas reglas que
barrido_semillas_brim.py). La dirección del RIP en la cara (w, mm_prom,
mm_max) sale de direccion_rip_en_cara(coef, ...), ya existente en
brim_rostro.py.

Uso
---
    python generar_reporte.py                  # semilla 42 (la de referencia del proyecto)
    python generar_reporte.py --semilla 7
    python generar_reporte.py --abrir           # además abre el HTML generado
"""

import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

import h5py
import numpy as np

import brim_rostro as p3
from espacio_rasgos_rostro import reconstruir_cara
from puntaje_poligenico import HEREDABILIDAD_MAX

# La consola de Windows por defecto no es UTF-8 (cp1252) y este script
# imprime acentos y símbolos (², á, ó...) -- sin esto, print() con esos
# caracteres se ve mal (mismo ajuste que ya usa correr_pipeline.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_H5 = RUTA_PROYECTO / "model2019_face12.h5"
RUTA_PLANTILLA = RUTA_PROYECTO / "reporte" / "plantilla.html"
RUTA_SALIDA = RUTA_PROYECTO / "reporte" / "reporte_rostro.html"

# Error máximo aceptado entre reconstruir_cara() (Punto 1, float32 completo) y
# la fórmula cuantizada a int16 que usa el visor en JS -- si se pasa de esto,
# algo en la cuantización está mal y no hay que seguir generando el reporte.
TOLERANCIA_MM = 0.01


def _b64_float32(array):
    return base64.b64encode(np.asarray(array, dtype=np.float32).tobytes()).decode("ascii")


def _b64_int16(array):
    return base64.b64encode(np.asarray(array, dtype=np.int16).tobytes()).decode("ascii")


def _b64_uint16(array):
    return base64.b64encode(np.asarray(array, dtype=np.uint16).tobytes()).decode("ascii")


def verificar_cuantizacion(media, base_pca, varianza, base_q, escala, n_componentes, semilla=0):
    """Chequea que media + Σ peso·desvío·escala·baseQ (la fórmula que arma el
    JS de la plantilla a partir de ROSTRO_DATA) reproduce reconstruir_cara()
    con error menor a TOLERANCIA_MM, para varios vectores de pesos al azar."""
    rng = np.random.default_rng(semilla)
    desvio = np.sqrt(varianza)
    peor = 0.0
    for _ in range(20):
        pesos = rng.uniform(-3, 3, size=n_componentes)
        cara_real = reconstruir_cara(pesos, media, base_pca, varianza, n_componentes)
        cara_cuantizada = media + (base_q * escala[None, :]) @ (pesos * desvio)
        peor = max(peor, float(np.abs(cara_real - cara_cuantizada).max()))
    if peor >= TOLERANCIA_MM:
        raise AssertionError(
            f"La reconstrucción cuantizada (la que usa el visor en JS) se aparta de "
            f"reconstruir_cara() por {peor:.4f} mm, por encima de la tolerancia de "
            f"{TOLERANCIA_MM} mm. No se generó el reporte."
        )
    return peor


def armar_rostro_data():
    """Arma ROSTRO_DATA: el espacio de rasgos real (Punto 1) más la topología
    y la varianza total del modelo completo (model2019_face12.h5). Devuelve
    también la varianza de las 199 componentes, para que armar_resultados()
    pueda calcular la curva de varianza acumulada sin volver a abrir el .h5."""
    if not RUTA_H5.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_H5.name} en la raíz del proyecto. Hace falta para "
            f"los triángulos de la malla y la varianza de las 199 componentes del "
            f"modelo completo (espacio_rasgos_rostro.npz solo guarda las 28 ya "
            f"retenidas, no el total)."
        )

    media, base_pca, varianza, n_componentes = p3.cargar_espacio()

    with h5py.File(RUTA_H5, "r") as f:
        varianza_total_199 = f["shape/model/pcaVariance"][:]
        cells = f["shape/representer/cells"][:]  # (3, n_triangulos), vértices ya indexados desde 0

    n_vertices = media.size // 3
    n_triangulos = cells.shape[1]
    # cells es (3, n_triangulos) -- un triángulo por columna. La plantilla
    # espera [t0v0, t0v1, t0v2, t1v0, ...], o sea un triángulo por fila.
    cells_flat = cells.T.reshape(-1)

    # Cuantización a int16 por componente (una escala distinta por cada una
    # de las 28, no una global) para que la malla completa entre en un HTML
    # liviano -- el error que introduce se verifica debajo.
    escala = np.abs(base_pca).max(axis=0) / 32767.0
    base_q = np.round(base_pca / escala[None, :]).astype(np.int16)

    error_cuantizacion = verificar_cuantizacion(
        media, base_pca, varianza, base_q, escala, n_componentes
    )
    print(f"Verificación de cuantización (Punto 1): error máximo {error_cuantizacion:.4f} mm "
          f"(tolerancia {TOLERANCIA_MM} mm).")

    # % de la varianza TOTAL del modelo (199 componentes), no de las 28 retenidas.
    pct_total = varianza / varianza_total_199.sum() * 100

    rostro_data = {
        "n_vertices": int(n_vertices),
        "n_triangles": int(n_triangulos),
        "n_componentes": int(n_componentes),
        "media_b64": _b64_float32(media),
        "cells_b64": _b64_uint16(cells_flat),
        "base_q_b64": _b64_int16(base_q.reshape(-1)),
        "escala_b64": _b64_float32(escala),
        "varianza_b64": _b64_float32(varianza),
        "pct_total_b64": _b64_float32(pct_total),
    }
    return rostro_data, varianza_total_199


def armar_resultados(semilla, varianza_total_199):
    """Arma RESULTADOS: una corrida real del pipeline (cohorte oficial del
    Punto 2 + BRIM del Punto 3), con su control de permutación y el barrido
    de 20 semillas -- los mismos números que da brim_rostro.py."""
    media, base_pca, varianza, n_componentes = p3.cargar_espacio()

    puntaje, pesos, _ = p3.cohorte_oficial(semilla=semilla)
    res = p3.correr_brim(pesos, puntaje, verbose=False)

    rng_perm = np.random.default_rng(semilla + 99)
    res_perm = p3.correr_brim(pesos, rng_perm.permutation(puntaje), verbose=False)

    # Dirección del RIP en la cara: la misma función que ya usa brim_rostro.py
    # al final de su programa principal.
    w, _, _, desp = p3.direccion_rip_en_cara(res["coef"], media, base_pca, varianza)

    # Valor teórico "verdadero" del coeficiente por componente, para la línea
    # de referencia del gráfico de pesos de BRIM: con heredabilidad h² pareja
    # en las 28 componentes (cohorte_oficial) y componentes de la cara casi
    # sin correlación entre sí (cada una es su propio genotipo + su propio
    # ruido ambiental), la pendiente esperada de puntaje contra cada
    # componente es Cov/Var = sqrt(h²/28)/1 = sqrt(h²/28) -- igual en las 28,
    # por simetría. No es el valor que estima BRIM (eso es 'coef'); es la
    # referencia teórica contra la que se lo compara.
    beta_verdadero = float(np.sqrt(HEREDABILIDAD_MAX / n_componentes))

    barrido = p3.barrido_semillas(n_semillas=20)

    cum = np.cumsum(varianza_total_199) / varianza_total_199.sum() * 100

    resultados = {
        "n": int(p3.N_PERSONAS),
        "h2": float(HEREDABILIDAD_MAX),
        "n_pls": int(res["n_pls"]),
        "r2": float(res["r2_loo"] * 100),
        "r2_ic": [float(v * 100) for v in res["r2_loo_ic"]],
        "trampa": float(res["r2_con_trampa"] * 100),
        "perm": float(res_perm["r2_loo"] * 100),
        "perm_ic": [float(v * 100) for v in res_perm["r2_loo_ic"]],
        "mm_prom": float(desp.mean()),
        "mm_max": float(desp.max()),
        "puntaje": [float(v) for v in puntaje],
        "rip": [float(v) for v in res["rip"]],
        "coef": [float(v) for v in res["coef"]],
        "coef_lo": [float(v) for v in res["coef_ic"][0]],
        "coef_hi": [float(v) for v in res["coef_ic"][1]],
        "beta_verdadero": beta_verdadero,
        "w": [float(v) for v in w],
        "barrido_honesto": [float(v * 100) for v in barrido["r2_principal"]],
        "barrido_perm": [float(v * 100) for v in barrido["r2_permutacion"]],
        "cum": [float(v) for v in cum],
        "cum28": float(cum[n_componentes - 1]),
    }
    return resultados


def generar_reporte(semilla=42, abrir=False):
    rostro_data, varianza_total_199 = armar_rostro_data()
    resultados = armar_resultados(semilla, varianza_total_199)

    plantilla = RUTA_PLANTILLA.read_text(encoding="utf-8")
    html = plantilla.replace(
        "/*__ROSTRO_DATA__*/null", json.dumps(rostro_data)
    ).replace(
        "/*__RESULTADOS__*/null", json.dumps(resultados)
    )

    RUTA_SALIDA.parent.mkdir(exist_ok=True)
    RUTA_SALIDA.write_text(html, encoding="utf-8")

    peso_mb = RUTA_SALIDA.stat().st_size / (1024 * 1024)
    print(f"Reporte generado en: {RUTA_SALIDA} ({peso_mb:.1f} MB)")

    if abrir:
        webbrowser.open(RUTA_SALIDA.resolve().as_uri())

    return RUTA_SALIDA


def main():
    parser = argparse.ArgumentParser(
        description="Genera reporte/reporte_rostro.html (Puntos 1 a 3 en una sola página)."
    )
    parser.add_argument("--semilla", type=int, default=42,
                        help="semilla de la cohorte oficial (default 42, la de referencia del proyecto)")
    parser.add_argument("--abrir", action="store_true",
                        help="abrir el HTML generado en el navegador al terminar")
    args = parser.parse_args()
    generar_reporte(semilla=args.semilla, abrir=args.abrir)


if __name__ == "__main__":
    main()
