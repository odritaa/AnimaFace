"""
Módulo Rostro — Punto 3 (BRIM): barrido de semillas
====================================================

Corre la corrida principal de brim_rostro.py (cohorte oficial) y el control
de permutación con 20 semillas distintas, y reporta media/mínimo/máximo del
R² leave-one-out de cada una. Una sola semilla no alcanza para citar el R²
en la tesis -- esta tabla es el número reproducible.

Uso
---
    python barrido_semillas_brim.py
"""

from brim_rostro import barrido_semillas, imprimir_barrido_semillas

if __name__ == "__main__":
    resultado = barrido_semillas(n_semillas=20)
    imprimir_barrido_semillas(resultado)
