"""
Módulo Rostro — Punto 3 (BRIM): barrido de semillas
====================================================

Corre la corrida principal de brim_rostro.py (cohorte oficial) y el control
de permutación con 20 semillas distintas, y reporta media/mínimo/máximo del
R² leave-one-out de cada una. Una sola semilla no alcanza para citar el R²
en la tesis -- esta tabla es el número reproducible.

También calcula el criterio B-03: la diferencia pareada honesto - permutación
semilla por semilla, con IC 95 % bootstrap (tiene que quedar entero > 0).

Uso
---
    python barrido_semillas_brim.py
"""

from brim_rostro import (
    barrido_semillas,
    diferencia_pareada,
    imprimir_barrido_semillas,
    imprimir_diferencia_pareada,
)

if __name__ == "__main__":
    resultado = barrido_semillas(n_semillas=20)
    imprimir_barrido_semillas(resultado)
    dif = diferencia_pareada(resultado["r2_principal"], resultado["r2_permutacion"])
    imprimir_diferencia_pareada(dif)
