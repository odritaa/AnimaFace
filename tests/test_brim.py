"""
Tests del Punto 3 (BRIM) -- conexión con la salida real del Punto 2 y
controles de sanidad.

Las tolerancias son deliberadamente anchas: el R² honesto varía de corrida a
corrida según la semilla (ver barrido_semillas_brim.py, que lo mide con 20
semillas distintas). Estos tests verifican el patrón esperado -- no un
número exacto de una sola semilla.
"""

import numpy as np
import pytest

from puntaje_poligenico import (
    HEREDABILIDAD_MAX,
    cara_observada,
    generar_genotipo_sintetico,
    puntaje_escalar,
)
from brim_rostro import (
    SEMILLA,
    cargar_espacio,
    cohorte_oficial,
    correr_brim,
    simular_cohorte,
)

# Bootstrap chico para estos tests: solo se mira r2_loo y r2_con_trampa, que
# no dependen de n_boot (lo que depende de n_boot es coef_ic/rip_se, que
# estos tests no usan) -- bajarlo acelera la corrida sin cambiar lo que se
# verifica.
N_BOOT_TEST = 5


@pytest.fixture(scope="module")
def espacio():
    return cargar_espacio()


@pytest.fixture(scope="module")
def cohorte():
    """Cohorte oficial (Punto 2 real: puntaje_poligenico.py), semilla del proyecto."""
    return cohorte_oficial()


def test_puntaje_escalar_estandarizado():
    genetica = generar_genotipo_sintetico(
        n_personas=2000, heredabilidad_por_componente=HEREDABILIDAD_MAX, semilla=1
    )
    puntaje = puntaje_escalar(genetica)
    assert puntaje.mean() == pytest.approx(0.0, abs=1e-9)
    assert puntaje.std() == pytest.approx(1.0, abs=1e-9)


def test_cara_observada_varianza_por_componente():
    genetica = generar_genotipo_sintetico(
        n_personas=5000, heredabilidad_por_componente=HEREDABILIDAD_MAX, semilla=2
    )
    pesos = cara_observada(genetica, HEREDABILIDAD_MAX, semilla=3)
    # cada componente debe tener varianza total ~1 (genética + ruido ambiental)
    assert np.allclose(pesos.var(axis=0), 1.0, atol=0.1)


def test_r2_con_trampa_mayor_que_honesto(cohorte):
    puntaje, pesos, _ = cohorte
    res = correr_brim(pesos, puntaje, n_boot=N_BOOT_TEST, verbose=False)
    assert res["r2_con_trampa"] > res["r2_loo"]


def test_corrida_principal_r2_honesto_en_rango(cohorte):
    puntaje, pesos, _ = cohorte
    res = correr_brim(pesos, puntaje, n_boot=N_BOOT_TEST, verbose=False)
    # rango observado en el barrido de 20 semillas (barrido_semillas_brim.py):
    # media ≈ 2.3 %, entre -2.8 % y 10.0 %. Se deja margen extra.
    assert -0.05 <= res["r2_loo"] <= 0.10


def test_permutacion_r2_loo_menor_a_3_por_ciento(cohorte):
    puntaje, pesos, _ = cohorte
    rng = np.random.default_rng(SEMILLA + 99)
    res_perm = correr_brim(pesos, rng.permutation(puntaje), n_boot=N_BOOT_TEST, verbose=False)
    # con el puntaje mezclado no debería quedar señal real: cercano a 0 o
    # negativo. En el barrido de 20 semillas varió entre -7.6 % y -1.7 %.
    assert res_perm["r2_loo"] < 0.03


def test_sin_senal_r2_loo_cercano_a_cero(espacio):
    _, _, varianza, _ = espacio
    puntaje, pesos, _ = simular_cohorte(varianza, semilla=SEMILLA + 7, sin_senal=True)
    res = correr_brim(pesos, puntaje, n_boot=N_BOOT_TEST, verbose=False)
    # sin señal genética, BRIM no debería "inventar" una señal positiva grande
    assert res["r2_loo"] < 0.05
