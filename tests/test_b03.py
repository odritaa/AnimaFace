"""
Tests del criterio B-03: diferencia pareada honesto - permutación por
semilla, con IC 95 % bootstrap. Aprueba si el límite inferior del IC es > 0.

Los dos primeros tests usan datos sintéticos conocidos (verifican la
función). El último corre el barrido real de 20 semillas (mismas semillas
que barrido_semillas_brim.py) y verifica el criterio.
"""

import numpy as np
import pytest

from brim_rostro import barrido_semillas, diferencia_pareada


def test_diferencias_constantes_dan_ic_degenerado():
    # todas las diferencias valen 5 -> cualquier remuestreo da media 5
    honesto = np.full(20, 7.0)
    permutacion = np.full(20, 2.0)
    dif = diferencia_pareada(honesto, permutacion)
    assert dif["media"] == pytest.approx(5.0)
    assert dif["ic95"][0] == pytest.approx(5.0)
    assert dif["ic95"][1] == pytest.approx(5.0)
    assert dif["n_positivas"] == 20


def test_diferencias_simetricas_no_cumplen_b03():
    # diferencias simétricas alrededor de 0 -> el IC tiene que incluir el 0
    d = np.concatenate([np.linspace(1, 10, 10), -np.linspace(1, 10, 10)])
    dif = diferencia_pareada(d, np.zeros(20))
    assert dif["media"] == pytest.approx(0.0, abs=1e-12)
    assert dif["ic95"][0] < 0       # el criterio NO se cumple
    assert dif["n_positivas"] == 10


def test_b03_barrido_real():
    resultado = barrido_semillas(n_semillas=20)
    dif = diferencia_pareada(resultado["r2_principal"], resultado["r2_permutacion"])
    ic_inf, ic_sup = dif["ic95"]
    print(f"\nB-03 -- media: {dif['media']*100:.2f} pp   "
          f"IC 95 %: [{ic_inf*100:.2f} ; {ic_sup*100:.2f}] pp   "
          f"positivas: {dif['n_positivas']} de {dif['n']}")
    assert ic_inf > 0
