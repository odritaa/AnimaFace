"""
Módulo Rostro — Punto 3: BRIM (Bootstrapped Response-based Imputation Modeling)
================================================================================

Qué hace este script
--------------------
Conecta el puntaje poligénico (Punto 2) con el espacio de rasgos faciales
(Punto 1). Para cada persona devuelve UN número, el RIP (Response-based
Imputed Predictor): "cuánto puntaje genético sugiere su cara", calculado SIN
usar el propio dato de esa persona (leave-one-out) y con incertidumbre
estimada por bootstrap.

Referencia: Claes, P., & Shriver, M. D. (2014). PLOS Genetics.

Flujo oficial (cohorte real del Punto 2)
-----------------------------------------
BRIM necesita dos piezas separadas por persona: un puntaje poligénico
ESCALAR (un solo número) y una cara OBSERVADA (vector de 28 pesos). El
Punto 2 original (puntaje_poligenico.py) solo daba la genética por
componente (n_personas, 28), pensada para usarse directamente como
'pesos' -- no daba ninguna de esas dos piezas. Se completó el Punto 2 con
dos funciones nuevas (no una decisión de BRIM):

  - puntaje_escalar(genetica)                -> puntaje poligénico (n,)
  - cara_observada(genetica, h²)             -> cara observada (n, 28)
                                                 (genética + ruido ambiental,
                                                  varianza total 1 por
                                                  componente)

cohorte_oficial() en este archivo junta esas dos piezas con
generar_genotipo_sintetico() para dar (puntaje, pesos) tal como los espera
correr_brim(). Es el flujo que usa la corrida principal de este script.

simular_cohorte() (más abajo) queda SOLO como cohorte alternativa para
pruebas de sanidad que la cohorte oficial no puede producir limpio (el
control "sin señal genética": con h²=0, generar_genotipo_sintetico() da
genética idénticamente cero y puntaje_escalar() no puede estandarizar una
suma constante) y para el barrido de sensibilidad general de BRIM (que
necesita techos de varianza por encima del 7,9 % que generar_genotipo_
sintetico() no acepta, porque ese techo es una validación del Punto 2, no
de BRIM).

Flujo interno (igual para cualquier cohorte)
---------------------------------------------
1. Se cargan las 28 componentes (espacio_rasgos_rostro.npz, Punto 1).
2. Se obtiene una cohorte: puntaje poligénico (n,) + pesos faciales (n, 28).
3. PLSR (mínimos cuadrados parciales) combina las 28 componentes en una sola
   dirección que predice el puntaje.
4. Leave-one-out: el RIP de cada persona sale de un modelo entrenado con
   todas las demás.
5. Bootstrap: repite el ajuste con remuestreos para medir qué tan estable es
   cada componente y cuánta incertidumbre tiene el RIP de cada persona.
6. Pruebas de sanidad: control nulo (puntaje mezclado => R² ~ 0) y
   comparación contra el R² "con trampa" (sin leave-one-out).

Uso
---
    python brim_rostro.py            # corre todo con la cohorte oficial
"""

from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from puntaje_poligenico import (
    HEREDABILIDAD_MAX,
    N_COMPONENTES_DEFAULT,
    cara_observada,
    generar_genotipo_sintetico,
    puntaje_escalar,
)

# ---------------------------------------------------------------------------
# Parámetros (todos modificables; cada uno es una decisión metodológica)
# ---------------------------------------------------------------------------
SEMILLA = 42
N_PERSONAS = 503          # tamaño de la subpoblación europea de 1000 Genomes;
                          # usar más personas (1000, 2504, 5000...) es una
                          # decisión del proyecto/tutor, no se cambia acá.
R2_TOTAL = 0.079          # varianza facial total atribuible al genotipo, para
                          # simular_cohorte() (barrido de sensibilidad); techo
                          # anclado en Xiong et al. (2025), hasta 7,9 %
N_BOOTSTRAP = 300
MAX_COMP_PLS = 10         # máximo de componentes PLS a probar

RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_NPZ = RUTA_PROYECTO / "espacio_rasgos_rostro.npz"
RUTA_SALIDA_NPZ = RUTA_PROYECTO / "brim_rostro_resultados.npz"
RUTA_SALIDA_PNG = RUTA_PROYECTO / "brim_rostro.png"


# ---------------------------------------------------------------------------
# Carga del espacio de rasgos (Punto 1)
# ---------------------------------------------------------------------------
def cargar_espacio(path=RUTA_NPZ):
    d = np.load(path)
    return d["media"], d["base_pca"], d["varianza"], int(d["n_componentes"])


# ---------------------------------------------------------------------------
# Cohorte oficial (Punto 2 real: puntaje_poligenico.py)
# ---------------------------------------------------------------------------
def cohorte_oficial(n_personas=N_PERSONAS, heredabilidad_por_componente=HEREDABILIDAD_MAX,
                     n_componentes=N_COMPONENTES_DEFAULT, semilla=SEMILLA):
    """
    Devuelve (puntaje, pesos, genetica) construidos con la salida real del
    Punto 2, en vez de datos simulados ad hoc para BRIM.

    puntaje  : (n,)   puntaje_escalar(genetica) -- puntaje poligénico
                       estandarizado (media 0, desvío 1).
    pesos    : (n, k) cara_observada(genetica, h²) -- genética + ruido
                       ambiental, varianza total 1 por componente (misma
                       convención que 'pesos' en reconstruir_cara()).
    genetica : (n, k) generar_genotipo_sintetico(...) -- solo la parte
                       genética, se devuelve por si hace falta inspeccionarla.

    Nota de potencia (no es algo a corregir, es para documentar): aplicar
    heredabilidad_por_componente=HEREDABILIDAD_MAX (7,9 %) a las 28
    componentes a la vez es generoso -- en Xiong et al. (2025) ese 7,9 % es
    el máximo reportado para UN rasgo, no el valor de todos a la vez. El
    parámetro queda listo para recibir un array con un h² distinto por
    componente cuando se tengan esos valores.
    """
    genetica = generar_genotipo_sintetico(n_personas, heredabilidad_por_componente,
                                           n_componentes, semilla)
    semilla_ruido = None if semilla is None else semilla + 1000
    pesos = cara_observada(genetica, heredabilidad_por_componente, semilla=semilla_ruido)
    puntaje = puntaje_escalar(genetica)
    return puntaje, pesos, genetica


# ---------------------------------------------------------------------------
# Cohorte simulada -- SOLO como alternativa para pruebas (ver docstring del
# módulo): el control "sin señal genética" y el barrido de sensibilidad.
# ---------------------------------------------------------------------------
def simular_cohorte(varianza, n_personas=N_PERSONAS, r2_total=R2_TOTAL,
                    semilla=SEMILLA, sin_senal=False):
    """
    Devuelve (puntaje, pesos, b_verdadero).

    puntaje : (n,)   puntaje poligénico estandarizado (media 0, desvío 1).
    pesos   : (n, k) posición de cada cara en las k componentes, en desvíos
                     estándar (misma convención que reconstruir_cara).
    b_verdadero : (k,) cuánto empuja el puntaje a cada componente. Sólo se
                  conoce porque la cohorte es simulada; sirve para chequear
                  que BRIM recupera lo que realmente se puso.

    A diferencia de cohorte_oficial(), acá el puntaje empuja a la cara en UNA
    dirección arbitraria del espacio de rasgos (un patrón de pesos desiguales
    entre las 28 componentes, con un techo de varianza total `r2_total`).
    Sirve para probar casos que cohorte_oficial() no puede producir (sin
    señal, techos por encima del 7,9 %), no para la corrida principal.
    """
    rng = np.random.default_rng(semilla)
    n_comp = len(varianza)
    puntaje = rng.standard_normal(n_personas)

    if sin_senal:
        b = np.zeros(n_comp)
    else:
        u = rng.uniform(size=n_comp) ** 2 * rng.choice([-1.0, 1.0], size=n_comp)
        b = np.sqrt(r2_total) * u / np.linalg.norm(u)   # sum(b_j^2) = r2_total

    ruido = rng.standard_normal((n_personas, n_comp))
    pesos = puntaje[:, None] * b[None, :] + ruido * np.sqrt(1 - b**2)[None, :]
    return puntaje, pesos, b


# ---------------------------------------------------------------------------
# Piezas de BRIM
# ---------------------------------------------------------------------------
def r2_fuera_de_muestra(y, y_pred):
    """R² = 1 - SSE/SST usando predicciones que NO vieron a esa persona."""
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)


def elegir_n_pls(pesos, puntaje, max_comp=MAX_COMP_PLS, k_folds=10, semilla=SEMILLA):
    """Cuántas componentes PLS usar: la que mejor predice en validación
    cruzada de 10 pliegues."""
    kf = KFold(k_folds, shuffle=True, random_state=semilla)
    r2 = []
    for k in range(1, max_comp + 1):
        pred = np.zeros_like(puntaje)
        for tr, te in kf.split(pesos):
            m = PLSRegression(n_components=k, scale=False).fit(pesos[tr], puntaje[tr])
            pred[te] = m.predict(pesos[te]).ravel()
        r2.append(r2_fuera_de_muestra(puntaje, pred))
    r2 = np.array(r2)
    return int(np.argmax(r2) + 1), r2


def rip_leave_one_out(pesos, puntaje, n_pls):
    """RIP de cada persona, calculado con un modelo que no la incluye."""
    n = len(puntaje)
    rip = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        m = PLSRegression(n_components=n_pls, scale=False).fit(pesos[mask], puntaje[mask])
        rip[i] = m.predict(pesos[i:i + 1]).item()
    return rip


def bootstrap_brim(pesos, puntaje, n_pls, n_boot=N_BOOTSTRAP, semilla=SEMILLA):
    """
    Remuestrea personas con reposición y reajusta el modelo cada vez.
    Devuelve:
      coefs  (n_boot, k)  coeficiente de cada componente en cada remuestreo
      rip_oob (n_boot, n) predicción para las personas que quedaron FUERA
                          de ese remuestreo (NaN para las que entraron)
    """
    rng = np.random.default_rng(semilla + 1)
    n, k = pesos.shape
    coefs = np.zeros((n_boot, k))
    rip_oob = np.full((n_boot, n), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        fuera = np.setdiff1d(np.arange(n), idx)
        m = PLSRegression(n_components=n_pls, scale=False).fit(pesos[idx], puntaje[idx])
        coefs[b] = m.coef_.ravel()
        rip_oob[b, fuera] = m.predict(pesos[fuera]).ravel()
    return coefs, rip_oob


def ic_bootstrap_r2(puntaje, rip, n_boot=2000, semilla=SEMILLA):
    """Intervalo de confianza 95 % del R² fuera de muestra (percentiles)."""
    rng = np.random.default_rng(semilla + 2)
    n = len(puntaje)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = r2_fuera_de_muestra(puntaje[idx], rip[idx])
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def correr_brim(pesos, puntaje, n_boot=N_BOOTSTRAP, verbose=True):
    n_pls, r2_cv = elegir_n_pls(pesos, puntaje)
    rip = rip_leave_one_out(pesos, puntaje, n_pls)
    coefs, rip_oob = bootstrap_brim(pesos, puntaje, n_pls, n_boot)

    modelo_final = PLSRegression(n_components=n_pls, scale=False).fit(pesos, puntaje)
    coef = modelo_final.coef_.ravel()

    res = {
        "n_pls": n_pls,
        "r2_cv_por_k": r2_cv,
        "rip": rip,
        "rip_se": np.nanstd(rip_oob, axis=0),
        "coef": coef,
        "coef_ic": np.percentile(coefs, [2.5, 97.5], axis=0),
        "r2_loo": r2_fuera_de_muestra(puntaje, rip),
        "r2_loo_ic": ic_bootstrap_r2(puntaje, rip),
        "corr_loo": np.corrcoef(puntaje, rip)[0, 1],
        "r2_con_trampa": modelo_final.score(pesos, puntaje),  # sin leave-one-out
        "r2_mejor_comp_sola": max(np.corrcoef(pesos[:, j], puntaje)[0, 1] ** 2
                                  for j in range(pesos.shape[1])),
    }
    if verbose:
        imprimir_resumen(res)
    return res


def imprimir_resumen(res, titulo=""):
    lo, hi = res["r2_loo_ic"]
    print(f"\n{titulo}")
    print(f"  Componentes PLS elegidas (validación cruzada): {res['n_pls']}")
    print(f"  R² leave-one-out (honesto):      {res['r2_loo']*100:6.2f} %   "
          f"IC95 % bootstrap [{lo*100:.2f} ; {hi*100:.2f}]")
    print(f"  Correlación RIP vs puntaje:      {res['corr_loo']:6.3f}")
    print(f"  R² 'con trampa' (sin LOO):       {res['r2_con_trampa']*100:6.2f} %")
    print(f"  Mejor componente sola (R²):      {res['r2_mejor_comp_sola']*100:6.2f} %")


# ---------------------------------------------------------------------------
# Barrido de semillas: cuánto varía el R² de corrida a corrida
# ---------------------------------------------------------------------------
# Una sola semilla no alcanza para citar el R² en la tesis -- el control de
# permutación en particular varía bastante de semilla a semilla (se observó
# entre -1 % y -6 % en corridas sueltas). Esta función corre la corrida
# principal (cohorte oficial) y la permutación con varias semillas distintas
# para reportar media/mínimo/máximo, que es el número reproducible.
def barrido_semillas(n_semillas=20, semilla_inicial=SEMILLA, n_boot=50):
    """
    n_boot se baja de N_BOOTSTRAP (300) a un valor menor acá: el bootstrap de
    coeficientes (coef_ic, rip_se) no hace falta para esta tabla, solo r2_loo
    de cada corrida -- bajarlo acelera las n_semillas corridas sin cambiar el
    número que se reporta.
    """
    r2_principal = np.empty(n_semillas)
    r2_permutacion = np.empty(n_semillas)
    semillas = [semilla_inicial + i for i in range(n_semillas)]

    for i, s in enumerate(semillas):
        puntaje, pesos, _ = cohorte_oficial(semilla=s)
        res = correr_brim(pesos, puntaje, n_boot=n_boot, verbose=False)
        r2_principal[i] = res["r2_loo"]

        rng = np.random.default_rng(s + 99)
        res_perm = correr_brim(pesos, rng.permutation(puntaje), n_boot=n_boot, verbose=False)
        r2_permutacion[i] = res_perm["r2_loo"]

    return {"semillas": semillas, "r2_principal": r2_principal, "r2_permutacion": r2_permutacion}


def imprimir_barrido_semillas(resultado):
    rp, rpm = resultado["r2_principal"], resultado["r2_permutacion"]
    n = len(rp)
    print(f"\n=== BARRIDO DE {n} SEMILLAS (corrida principal vs permutación, R² LOO) ===")
    print(f"  R² honesto     -- media: {rp.mean()*100:6.2f} %   "
          f"mín: {rp.min()*100:6.2f} %   máx: {rp.max()*100:6.2f} %")
    print(f"  R² permutación -- media: {rpm.mean()*100:6.2f} %   "
          f"mín: {rpm.min()*100:6.2f} %   máx: {rpm.max()*100:6.2f} %")


# ---------------------------------------------------------------------------
# Dirección del RIP en el espacio facial
# ---------------------------------------------------------------------------
def direccion_rip_en_cara(coef, media, base_pca, varianza, n_desvios=2.0):
    """
    Traduce el vector `coef` (qué componentes pesan y con qué signo) a dos
    caras: "RIP bajo" y "RIP alto". Devuelve también el desplazamiento
    promedio entre ambas, en las unidades del modelo (mm).
    """
    w = coef / np.linalg.norm(coef)
    desvio = np.sqrt(varianza)
    cara_alta = media + base_pca @ (w * n_desvios * desvio)
    cara_baja = media - base_pca @ (w * n_desvios * desvio)
    desp = np.linalg.norm((cara_alta - cara_baja).reshape(-1, 3), axis=1)
    return w, cara_baja, cara_alta, desp


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def graficar(puntaje, res, b_verdadero, path_salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROJO, DORADO, GRIS = "#B7362D", "#C9962E", "#6E6860"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    ax[0].scatter(puntaje, res["rip"], s=10, alpha=0.5, color=DORADO)
    lim = [min(puntaje.min(), res["rip"].min()), max(puntaje.max(), res["rip"].max())]
    ax[0].plot(lim, lim, color=GRIS, lw=1, ls="--")
    ax[0].set_xlabel("Puntaje poligénico real")
    ax[0].set_ylabel("RIP (leave-one-out)")
    ax[0].set_title(f"R² honesto = {res['r2_loo']*100:.1f} %")
    ax[0].grid(alpha=0.25)

    k = len(res["coef"])
    x = np.arange(1, k + 1)
    c = res["coef"] / np.abs(res["coef"]).max()
    ic = res["coef_ic"] / np.abs(res["coef"]).max()
    ax[1].bar(x, c, color=ROJO, alpha=0.8, label="Peso estimado por BRIM")
    ax[1].errorbar(x, c, yerr=[c - ic[0], ic[1] - c], fmt="none", ecolor=GRIS, lw=0.8)
    if b_verdadero is not None:
        ax[1].plot(x, b_verdadero / np.abs(b_verdadero).max(), "o", ms=3.5,
                   color=DORADO, label="Peso verdadero (simulado)")
    ax[1].axhline(0, color=GRIS, lw=0.6)
    ax[1].set_xlabel("Componente facial")
    ax[1].set_ylabel("Peso relativo")
    ax[1].set_title("Qué componentes usa el RIP (IC95 % bootstrap)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(path_salida, dpi=150)
    print(f"Gráfico guardado en: {path_salida}")


# ---------------------------------------------------------------------------
# Programa principal: corrida + pruebas de sanidad
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    media, base_pca, varianza, k = cargar_espacio()
    print(f"Espacio de rasgos: {k} componentes, {media.size // 3} vértices")

    # --- 1. Corrida principal (cohorte oficial: Punto 2 real) ---------------
    puntaje, pesos, genetica = cohorte_oficial()
    res = correr_brim(pesos, puntaje, verbose=False)
    imprimir_resumen(res, "=== CORRIDA PRINCIPAL (cohorte oficial, puntaje_poligenico.py, n=%d) ===" % N_PERSONAS)

    # --- 2. Control nulo: sin señal genética ---------------------------------
    # La cohorte oficial no puede producir este caso limpio (h²=0 da genética
    # idénticamente cero y puntaje_escalar() no puede estandarizar una suma
    # constante), así que usa simular_cohorte() (alternativa de prueba).
    p0, w0, _ = simular_cohorte(varianza, semilla=SEMILLA + 7, sin_senal=True)
    res0 = correr_brim(w0, p0, verbose=False)
    imprimir_resumen(res0, "=== CONTROL NULO (sin señal genética, simular_cohorte) ===")

    # --- 3. Control de permutación sobre la cohorte oficial -------------------
    rng = np.random.default_rng(SEMILLA + 99)
    res_perm = correr_brim(pesos, rng.permutation(puntaje), verbose=False)
    imprimir_resumen(res_perm, "=== PERMUTACIÓN (puntaje mezclado entre personas, cohorte oficial) ===")

    # --- 4. Sensibilidad al tamaño de la señal (simular_cohorte) -------------
    # Barrido general del comportamiento de BRIM, incluyendo techos por
    # encima del 7,9 % que generar_genotipo_sintetico() no acepta (ese techo
    # es una validación del Punto 2, no una limitación de BRIM en sí).
    print("\n=== SENSIBILIDAD: varianza facial atribuida al genotipo vs R² recuperado ===")
    for techo in [0.01, 0.03, 0.079, 0.15]:
        p, w, _ = simular_cohorte(varianza, r2_total=techo, semilla=SEMILLA + 3)
        r = correr_brim(w, p, n_boot=50, verbose=False)
        print(f"  genotipo explica {techo*100:5.1f} %  ->  R² LOO = {r['r2_loo']*100:6.2f} %"
              f"  (PLS={r['n_pls']})")

    # --- 5. Dirección del RIP en la cara (cohorte oficial) --------------------
    w, cara_baja, cara_alta, desp = direccion_rip_en_cara(res["coef"], media, base_pca, varianza)
    print(f"\nDiferencia de forma entre RIP -2 y RIP +2: "
          f"{desp.mean():.2f} mm promedio, {desp.max():.2f} mm máximo")

    # --- 6. Exportar ------------------------------------------------------------
    np.savez(RUTA_SALIDA_NPZ,
             puntaje=puntaje, rip_loo=res["rip"], rip_se=res["rip_se"],
             coef=res["coef"], coef_ic=res["coef_ic"], direccion=w,
             n_pls=res["n_pls"], r2_loo=res["r2_loo"], r2_loo_ic=res["r2_loo_ic"])
    print(f"Resultados exportados a: {RUTA_SALIDA_NPZ}")
    # Sin b_verdadero: la cohorte oficial no tiene una única dirección "real"
    # fija como simular_cohorte() (cada componente recibe su propio sorteo
    # genético independiente; lo único "esperado" es que, con heredabilidad
    # pareja en las 28 componentes, el peso recuperado sea parejo también).
    graficar(puntaje, res, None, RUTA_SALIDA_PNG)
