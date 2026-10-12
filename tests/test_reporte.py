"""
Tests del generador de reporte (generar_reporte.py).

Genera reporte/reporte_rostro.html con la semilla 42 (la de referencia del
proyecto) una sola vez para todo el módulo, y chequea su forma, tamaño y los
números clave contra los valores citados en CLAUDE.md / barrido_semillas_brim.py.
Las tolerancias son las mismas que ya usa tests/test_brim.py: anchas a
propósito, porque lo que se verifica es que el reporte reproduce el patrón
esperado, no que una sola corrida dé un número exacto.
"""

import json
import re

import numpy as np
import pytest

from generar_reporte import generar_reporte

TOL_PCT = 0.5   # puntos porcentuales, para R²
TOL_MM = 0.05   # milímetros, para la distancia RIP-2/RIP+2


def _extraer_json(html, nombre_variable):
    patron = r"const " + nombre_variable + r" = (\{.*?\});"
    coincidencia = re.search(patron, html, re.S)
    assert coincidencia, f"no se encontró 'const {nombre_variable} = {{...}};' en el HTML generado"
    return json.loads(coincidencia.group(1))


@pytest.fixture(scope="module")
def reporte():
    ruta = generar_reporte(semilla=42, abrir=False)
    html = ruta.read_text(encoding="utf-8")
    rostro_data = _extraer_json(html, "ROSTRO_DATA")
    resultados = _extraer_json(html, "RESULTADOS")
    return ruta, rostro_data, resultados


def test_archivo_generado_y_liviano(reporte):
    ruta, _, _ = reporte
    assert ruta.exists()
    assert ruta.stat().st_size < 16 * 1024 * 1024


def test_rostro_data_tiene_las_claves_esperadas(reporte):
    _, rostro_data, _ = reporte
    claves = {
        "n_vertices", "n_triangles", "n_componentes", "media_b64", "cells_b64",
        "base_q_b64", "escala_b64", "varianza_b64", "pct_total_b64",
    }
    assert claves <= rostro_data.keys()
    assert rostro_data["n_vertices"] == 27657
    assert rostro_data["n_triangles"] == 55040
    assert rostro_data["n_componentes"] == 28


def test_resultados_tiene_las_claves_esperadas(reporte):
    _, _, r = reporte
    claves = {
        "n", "h2", "n_pls", "r2", "r2_ic", "trampa", "perm", "perm_ic", "mm_prom",
        "mm_max", "puntaje", "rip", "coef", "coef_lo", "coef_hi", "beta_verdadero",
        "w", "barrido_honesto", "barrido_perm", "cum", "cum28",
    }
    assert claves <= r.keys()
    assert len(r["puntaje"]) == r["n"] == 503
    assert len(r["rip"]) == r["n"]
    assert len(r["coef"]) == len(r["coef_lo"]) == len(r["coef_hi"]) == len(r["w"]) == 28
    assert len(r["barrido_honesto"]) == len(r["barrido_perm"]) == 20


def test_sin_nan(reporte):
    _, _, r = reporte
    listas = (r["puntaje"], r["rip"], r["coef"], r["coef_lo"], r["coef_hi"], r["w"],
              r["barrido_honesto"], r["barrido_perm"], r["cum"], r["r2_ic"], r["perm_ic"])
    for lista in listas:
        assert not any(np.isnan(v) for v in lista)
    escalares = (r["r2"], r["trampa"], r["perm"], r["mm_prom"], r["mm_max"],
                 r["beta_verdadero"], r["cum28"])
    assert not any(np.isnan(v) for v in escalares)


def test_numeros_de_referencia_semilla_42(reporte):
    _, _, r = reporte
    # Valores citados en CLAUDE.md / reproducidos por brim_rostro.py con semilla 42.
    assert r["r2"] == pytest.approx(2.89, abs=TOL_PCT)
    assert r["perm"] == pytest.approx(-1.79, abs=TOL_PCT)
    assert r["trampa"] == pytest.approx(11.45, abs=TOL_PCT)
    assert r["mm_prom"] == pytest.approx(2.32, abs=TOL_MM)
    assert r["mm_max"] == pytest.approx(4.20, abs=TOL_MM)
    assert r["cum28"] == pytest.approx(95.07, abs=0.05)
    # Barrido de 20 semillas (mismo criterio que barrido_semillas_brim.py).
    assert float(np.mean(r["barrido_honesto"])) == pytest.approx(2.25, abs=TOL_PCT)
    assert float(np.mean(r["barrido_perm"])) == pytest.approx(-5.41, abs=TOL_PCT)
