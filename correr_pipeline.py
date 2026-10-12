"""
Módulo Rostro — Pipeline completo (Puntos 1, 2 y 3 en un solo comando)
========================================================================

Qué hace este script
---------------------
Corre, en orden, las verificaciones y resultados de los tres puntos ya
completos del proyecto, reutilizando las funciones de cada módulo -- no
reimplementa ninguna lógica de Punto 1, 2 o 3:

  1. Punto 1 (espacio de rasgos): carga espacio_rasgos_rostro.npz y corre
     las verificaciones de espacio_rasgos_rostro.py (28 componentes,
     27.657 vértices, pesos=0 -> cara promedio, desplazamiento con PC1
     +2 desvíos). Si model2019_face12.h5 está en la carpeta, verifica
     además contra el archivo original (varianza total, componente 1). Si
     no está, lo avisa y sigue solo con el .npz -- sin el .h5 no se puede
     comprobar de verdad el "95% de la varianza total" (el .npz solo
     guarda las 28 componentes ya retenidas, no el total del modelo).
  2. Punto 2 (puntaje poligénico): genera la cohorte con las funciones de
     puntaje_poligenico.py (generar_genotipo_sintetico, puntaje_escalar,
     cara_observada) y resume desvío por componente, puntaje escalar y
     varianza de la cara observada.
  3. Punto 3 (BRIM): corre brim_rostro.cohorte_oficial() + correr_brim(),
     con control de permutación y R² "con trampa".
  4. Barrido de semillas: brim_rostro.barrido_semillas() (20 semillas, o 5
     con --rapido).
  5. Tests: corre pytest sobre tests/ y resume cuántos pasan.

Cada etapa corre dentro de un try/except en main(): si una falla con una
excepción, se marca FALLÓ y el pipeline sigue con la siguiente -- ninguna
etapa depende de que otra haya corrido antes (cada una genera su propia
cohorte).

Lugar reservado para etapas futuras del proyecto (no implementadas acá,
ver el final del archivo): Punto 4 (calibración) y Punto 5.

Salida
------
  - Resumen en consola, por etapa, con marca OK / REVISAR / N/D / FALLÓ.
  - reporte_pipeline.txt: el mismo resumen + fecha de la corrida.
  - reporte_pipeline.png: panel de 4 gráficos (varianza acumulada del
    Punto 1, puntaje escalar del Punto 2, RIP vs. puntaje real del Punto 3,
    barrido de semillas).

Uso
---
    python correr_pipeline.py             # barrido de 20 semillas
    python correr_pipeline.py --rapido    # barrido de 5 semillas (desarrollo)
"""

import argparse
import io
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np

import espacio_rasgos_rostro as p1
import puntaje_poligenico as p2
import brim_rostro as p3

# La consola de Windows por defecto no es UTF-8 (cp1252) y este script
# imprime acentos y símbolos (², ≈, á, ó...) -- sin esto, cualquier print()
# con esos caracteres tira UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROJO, DORADO, GRIS = "#B7362D", "#C9962E", "#6E6860"

RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_H5 = RUTA_PROYECTO / "model2019_face12.h5"
RUTA_REPORTE_TXT = RUTA_PROYECTO / "reporte_pipeline.txt"
RUTA_REPORTE_PNG = RUTA_PROYECTO / "reporte_pipeline.png"


# ---------------------------------------------------------------------------
# Reporte: junta lo que se imprime en consola para después guardarlo igual
# en reporte_pipeline.txt
# ---------------------------------------------------------------------------
class Reporte:
    def __init__(self):
        self.lineas = []

    def linea(self, texto=""):
        print(texto)
        self.lineas.append(texto)

    def titulo(self, texto):
        self.linea("")
        self.linea("=" * 70)
        self.linea(texto)
        self.linea("=" * 70)

    def marca(self, ok, etiqueta, detalle=""):
        estado = "OK" if ok else "REVISAR"
        texto = f"  [{estado:7s}] {etiqueta}"
        if detalle:
            texto += f" -- {detalle}"
        self.linea(texto)
        return ok

    def capturar(self, func, *args, **kwargs):
        """Corre func (una función ya existente de Punto 1/2/3 que imprime
        con print()) capturando su salida, para que quede también en el
        reporte -- sin reescribir lo que esa función ya hace."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            resultado = func(*args, **kwargs)
        texto = buf.getvalue()
        print(texto, end="")
        self.lineas.extend(texto.splitlines())
        return resultado

    def texto(self):
        return "\n".join(self.lineas)


# ---------------------------------------------------------------------------
# Etapa 1 -- Punto 1 (espacio de rasgos)
# ---------------------------------------------------------------------------
def etapa_punto1(reporte):
    reporte.titulo("PUNTO 1 -- Espacio de rasgos faciales (Basel Face Model)")

    # espacio_rasgos_rostro.py no tiene un loader propio del .npz (solo lo
    # exporta) -- se reutiliza el de brim_rostro.py, que es el mismo archivo.
    media, base_pca, varianza, n_comp = p3.cargar_espacio()
    n_vertices = media.size // 3

    reporte.marca(n_comp == 28, "28 componentes retenidas", f"n_componentes={n_comp}")
    reporte.marca(n_vertices == 27657, "27.657 vértices", f"vértices={n_vertices}")

    ok_media = reporte.capturar(p1.verificar_reconstruccion, media, base_pca, varianza, n_comp)
    reporte.marca(ok_media, "pesos=0 reproduce la cara promedio exacta")

    pesos_pc1 = np.zeros(n_comp)
    pesos_pc1[0] = 2.0
    cara_pc1 = p1.reconstruir_cara(pesos_pc1, media, base_pca, varianza, n_comp)
    desplazamiento = np.linalg.norm((cara_pc1 - media).reshape(-1, 3), axis=1)
    # sanity: un desplazamiento de unos pocos mm, no algo absurdo (ni cero)
    desplazamiento_razonable = 0 < desplazamiento.mean() < 50
    reporte.marca(desplazamiento_razonable, "desplazamiento con PC1 +2 desvíos no es absurdo",
                  f"{desplazamiento.mean():.2f} mm promedio, {desplazamiento.max():.2f} mm máximo")

    var_acum_h5 = None
    if RUTA_H5.exists():
        try:
            _, _, varianza_h5 = p1.cargar_bfm_real(str(RUTA_H5))
            var_exp_h5, var_acum_h5 = reporte.capturar(p1.analizar_varianza_bfm, varianza_h5)
            reporte.marca(var_acum_h5[n_comp - 1] >= 0.95,
                          "varianza retenida >= 95% del total (contra .h5)",
                          f"{var_acum_h5[n_comp - 1] * 100:.2f}% con las primeras {n_comp} componentes")
            reporte.marca(np.isclose(varianza_h5[0], varianza[0]),
                          "componente 1 del .npz coincide con el .h5",
                          f".npz={varianza[0]:.1f}  .h5={varianza_h5[0]:.1f}")
        except Exception as exc:
            reporte.linea(f"  [REVISAR] no se pudo leer model2019_face12.h5 con la estructura "
                          f"esperada ({exc!r}) -- se sigue solo con el .npz.")
    else:
        reporte.linea("  [N/D    ] model2019_face12.h5 no está en la carpeta -- no se puede "
                      "comprobar la varianza total contra el modelo original (el .npz solo "
                      "guarda las 28 componentes ya retenidas, no el total). Se sigue con el .npz.")

    return {"varianza": varianza, "n_comp": n_comp, "var_acum_h5": var_acum_h5}


# ---------------------------------------------------------------------------
# Etapa 2 -- Punto 2 (puntaje poligénico)
# ---------------------------------------------------------------------------
def etapa_punto2(reporte, n_personas=p3.N_PERSONAS, semilla=p3.SEMILLA):
    reporte.titulo("PUNTO 2 -- Puntaje poligénico (puntaje_poligenico.py)")

    genetica = p2.generar_genotipo_sintetico(n_personas, p2.HEREDABILIDAD_MAX, semilla=semilla)
    puntaje = p2.puntaje_escalar(genetica)
    pesos = p2.cara_observada(genetica, p2.HEREDABILIDAD_MAX, semilla=semilla + 1000)

    desvio = genetica.std(axis=0)
    reporte.linea(f"  Desvío por componente (genética) -- mín/media/máx: "
                  f"{desvio.min():.4f} / {desvio.mean():.4f} / {desvio.max():.4f}  "
                  f"(esperado ≈ {np.sqrt(p2.HEREDABILIDAD_MAX):.4f})")

    reporte.marca(np.isclose(puntaje.mean(), 0, atol=1e-6) and np.isclose(puntaje.std(), 1, atol=1e-6),
                  "puntaje escalar estandarizado (media 0, desvío 1)",
                  f"media={puntaje.mean():.2e}  desvío={puntaje.std():.4f}")

    varianzas = pesos.var(axis=0)
    reporte.marca(np.allclose(varianzas, 1.0, atol=0.15),
                  "varianza de la cara observada ≈ 1 por componente",
                  f"mín={varianzas.min():.3f}  máx={varianzas.max():.3f}")

    return {"genetica": genetica, "puntaje": puntaje, "pesos": pesos}


# ---------------------------------------------------------------------------
# Etapa 3 -- Punto 3 (BRIM)
# ---------------------------------------------------------------------------
def etapa_punto3(reporte, semilla=p3.SEMILLA):
    reporte.titulo("PUNTO 3 -- BRIM (brim_rostro.py)")

    puntaje, pesos, _ = p3.cohorte_oficial(semilla=semilla)
    res = p3.correr_brim(pesos, puntaje, verbose=False)
    reporte.capturar(p3.imprimir_resumen, res, "Corrida principal (cohorte oficial)")

    rng = np.random.default_rng(semilla + 99)
    res_perm = p3.correr_brim(pesos, rng.permutation(puntaje), verbose=False)
    reporte.capturar(p3.imprimir_resumen, res_perm, "Control de permutación")

    reporte.marca(res["r2_con_trampa"] > res["r2_loo"], "R² 'con trampa' > R² honesto",
                  f"{res['r2_con_trampa'] * 100:.2f}% > {res['r2_loo'] * 100:.2f}%")
    reporte.marca(res_perm["r2_loo"] < 0.03, "R² con el puntaje mezclado < 3%",
                  f"{res_perm['r2_loo'] * 100:.2f}%")

    return {"puntaje": puntaje, "pesos": pesos, "res": res, "res_perm": res_perm}


# ---------------------------------------------------------------------------
# Etapa 4 -- Barrido de semillas
# ---------------------------------------------------------------------------
def etapa_barrido(reporte, n_semillas=20):
    reporte.titulo(f"BARRIDO DE SEMILLAS -- {n_semillas} semillas")
    resultado = p3.barrido_semillas(n_semillas=n_semillas)
    reporte.capturar(p3.imprimir_barrido_semillas, resultado)
    return resultado


# ---------------------------------------------------------------------------
# Etapa 5 -- Tests
# ---------------------------------------------------------------------------
def etapa_tests(reporte):
    reporte.titulo("TESTS -- pytest tests/")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=RUTA_PROYECTO, capture_output=True, text=True,
    )
    salida = (proc.stdout + proc.stderr).strip()
    for linea in salida.splitlines():
        reporte.linea(f"  {linea}")
    ok = proc.returncode == 0
    ultima_linea = salida.splitlines()[-1] if salida else ""
    reporte.marca(ok, "pytest tests/", ultima_linea)
    return {"returncode": proc.returncode, "salida": salida}


# ---------------------------------------------------------------------------
# Panel de gráficos
# ---------------------------------------------------------------------------
def graficar_panel(datos, path_salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) varianza acumulada
    d1 = datos.get("punto1")
    if d1 is not None:
        if d1.get("var_acum_h5") is not None:
            n_mostrar = min(60, len(d1["var_acum_h5"]))
            var_acum = d1["var_acum_h5"][:n_mostrar] * 100
            subtitulo = "contra la varianza total del .h5"
        else:
            varianza = d1["varianza"]
            var_acum = np.cumsum(varianza) / varianza.sum() * 100
            subtitulo = "sobre las 28 retenidas -- sin .h5 no se ve el total del modelo"
        ax[0, 0].plot(range(1, len(var_acum) + 1), var_acum, marker="o", markersize=3,
                      linewidth=1.5, color=ROJO)
        ax[0, 0].axvline(d1["n_comp"], color=GRIS, linestyle=":", linewidth=1)
        ax[0, 0].axhline(95, color=DORADO, linestyle="--", linewidth=1)
        ax[0, 0].set_title(f"(a) Varianza acumulada\n{subtitulo}", fontsize=9)
        ax[0, 0].set_xlabel("Componente")
        ax[0, 0].set_ylabel("% acumulado")
        ax[0, 0].grid(alpha=0.25)
    else:
        ax[0, 0].axis("off")
        ax[0, 0].set_title("(a) Punto 1: sin datos (etapa falló)")

    # (b) puntaje escalar de la cohorte
    d2 = datos.get("punto2")
    if d2 is not None:
        ax[0, 1].hist(d2["puntaje"], bins=30, color=DORADO, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax[0, 1].set_title("(b) Puntaje poligénico escalar\n(cohorte del Punto 2)", fontsize=9)
        ax[0, 1].set_xlabel("Puntaje (estandarizado)")
        ax[0, 1].set_ylabel("Personas")
        ax[0, 1].grid(alpha=0.25)
    else:
        ax[0, 1].axis("off")
        ax[0, 1].set_title("(b) Punto 2: sin datos (etapa falló)")

    # (c) RIP vs puntaje real
    d3 = datos.get("punto3")
    if d3 is not None:
        res, puntaje = d3["res"], d3["puntaje"]
        ax[1, 0].scatter(puntaje, res["rip"], s=10, alpha=0.5, color=DORADO)
        lim = [min(puntaje.min(), res["rip"].min()), max(puntaje.max(), res["rip"].max())]
        ax[1, 0].plot(lim, lim, color=GRIS, lw=1, ls="--")
        ax[1, 0].set_title(f"(c) RIP vs. puntaje real\nR² honesto = {res['r2_loo'] * 100:.1f}%", fontsize=9)
        ax[1, 0].set_xlabel("Puntaje poligénico real")
        ax[1, 0].set_ylabel("RIP (leave-one-out)")
        ax[1, 0].grid(alpha=0.25)
    else:
        ax[1, 0].axis("off")
        ax[1, 0].set_title("(c) Punto 3: sin datos (etapa falló)")

    # (d) barrido de semillas
    d4 = datos.get("barrido")
    if d4 is not None:
        x = range(1, len(d4["r2_principal"]) + 1)
        ax[1, 1].plot(x, np.array(d4["r2_principal"]) * 100, "o-", color=ROJO, label="R² honesto", ms=4)
        ax[1, 1].plot(x, np.array(d4["r2_permutacion"]) * 100, "o-", color=DORADO, label="R² permutación", ms=4)
        ax[1, 1].axhline(0, color=GRIS, lw=0.8)
        ax[1, 1].set_title(f"(d) Barrido de {len(d4['r2_principal'])} semillas", fontsize=9)
        ax[1, 1].set_xlabel("Semilla (índice)")
        ax[1, 1].set_ylabel("R² LOO (%)")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].grid(alpha=0.25)
    else:
        ax[1, 1].axis("off")
        ax[1, 1].set_title("(d) Barrido de semillas: sin datos (etapa falló)")

    plt.tight_layout()
    plt.savefig(path_salida, dpi=150)
    print(f"Panel guardado en: {path_salida}")


# ---------------------------------------------------------------------------
# Lugar reservado para etapas futuras del proyecto -- NO implementadas acá.
# Cuando se definan, agregarlas a la lista ETAPAS en main() como las demás.
# ---------------------------------------------------------------------------
def etapa_calibracion(reporte):
    """Punto 4 del proyecto (calibración). Todavía no implementado."""
    raise NotImplementedError("Punto 4 (calibración) todavía no está implementado.")


def etapa_punto5(reporte):
    """Punto 5 del proyecto. Todavía no implementado ni definido."""
    raise NotImplementedError("Punto 5 todavía no está implementado.")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo del módulo Rostro (Puntos 1, 2 y 3)."
    )
    parser.add_argument("--rapido", action="store_true",
                        help="usa 5 semillas en el barrido en vez de 20 (para desarrollo)")
    args = parser.parse_args()
    n_semillas = 5 if args.rapido else 20

    reporte = Reporte()
    reporte.linea(f"Pipeline Rostro -- corrida del {datetime.now():%Y-%m-%d %H:%M}")
    if args.rapido:
        reporte.linea("Modo --rapido: barrido de 5 semillas (no 20).")

    etapas = [
        ("punto1", etapa_punto1, {}),
        ("punto2", etapa_punto2, {}),
        ("punto3", etapa_punto3, {}),
        ("barrido", etapa_barrido, {"n_semillas": n_semillas}),
        ("tests", etapa_tests, {}),
        # Etapas futuras del proyecto (Punto 4: calibración: Punto 5): no
        # implementadas todavía -- agregar acá cuando corresponda:
        #   ("calibracion", etapa_calibracion, {}),
        #   ("punto5", etapa_punto5, {}),
    ]

    datos = {}
    for nombre, fn, kwargs in etapas:
        try:
            datos[nombre] = fn(reporte, **kwargs)
        except Exception as exc:
            reporte.titulo(f"ETAPA '{nombre}' FALLÓ")
            reporte.linea(f"  [FALLÓ  ] {exc!r}")
            datos[nombre] = None

    RUTA_REPORTE_TXT.write_text(reporte.texto(), encoding="utf-8")
    print(f"\nReporte de texto guardado en: {RUTA_REPORTE_TXT}")

    try:
        graficar_panel(datos, RUTA_REPORTE_PNG)
    except Exception as exc:
        print(f"No se pudo generar el panel de gráficos: {exc!r}")

    try:
        from generar_reporte import generar_reporte
        ruta_html = generar_reporte(semilla=p3.SEMILLA)
        print(f"\nReporte HTML (visor 3D + resultados) generado en: {ruta_html}")
    except Exception as exc:
        print(f"No se pudo generar el reporte HTML: {exc!r}")


if __name__ == "__main__":
    main()
