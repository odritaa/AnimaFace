"""
Módulo Rostro — Puntaje poligénico sintético (Basel Face Model + PCA)
=======================================================================

Este módulo es la capa del Punto 2 del proyecto: el puntaje poligénico por
componente. Se apoya en el espacio de rasgos ya construido en
espacio_rasgos_rostro.py (28 componentes PCA del Basel Face Model 2019, recorte face12) pero
no lo importa ni lo modifica -- es una capa nueva y separada.

DECISIÓN DE ALCANCE (documentada, no es un atajo silencioso):
  En vez de simular las 253 variantes genéticas individuales de Xiong et al.
  (2025) variante por variante, acá se simula directamente, por cada una de
  las 28 componentes PCA, un puntaje poligénico como variable aleatoria con
  media 0 y varianza anclada en la heredabilidad ya publicada (hasta 7,9% de
  la varianza total del componente, según el paper). Es una simplificación
  metodológica documentada: reemplaza la simulación SNP por SNP porque no
  tenemos datos reales emparejados a nivel individual (genotipo + escaneo 3D
  de la misma persona), igual que ya está justificado en el resto del
  proyecto (ver espacio_rasgos_rostro.py, sección de decisiones documentadas).

UNIDADES: reconstruir_cara() en espacio_rasgos_rostro.py interpreta 'pesos'
en desvíos estándar de cada componente (pesos[k]=2 -> "+2 desvíos estándar en
la componente k"). h² se define acá como fracción de la varianza
*estandarizada* (=1) de cada componente explicada por el genotipo, así que su
desvío estándar es directamente sqrt(h²) en esas mismas unidades -- el
puntaje poligénico que devuelve esta función se puede usar tal cual como
'pesos' en reconstruir_cara(), sin reescalar por la varianza real (mm²) del
.npz.
"""

import numpy as np

# Techo documentado en Xiong et al. (2025): hasta 7.9% de la varianza de un
# componente de forma facial explicada por el puntaje poligénico de esa
# componente. Se usa como cota de validación, no como valor fijo obligatorio.
HEREDABILIDAD_MAX = 0.079

N_COMPONENTES_DEFAULT = 28 # igual a N_COMPONENTES en espacio_rasgos_rostro.py


def generar_genotipo_sintetico(n_personas, heredabilidad_por_componente,
                                n_componentes=N_COMPONENTES_DEFAULT, semilla=None):
    """
    Simula un puntaje poligénico por componente PCA, para n_personas
    "personas" sintéticas.

    Parameters
    ----------
    n_personas : int
        Cantidad de personas a simular.
    heredabilidad_por_componente : float o array-like
        h² por componente (fracción de varianza estandarizada explicada por
        el genotipo), entre 0 y HEREDABILIDAD_MAX. Puede pasarse:
          - un único float: se aplica igual a las n_componentes componentes.
          - un array de longitud n_componentes: un h² distinto por componente
            (por ejemplo, si más adelante se afinan valores por componente
            según el paper en vez de usar el techo parejo).
    n_componentes : int
        Cantidad de componentes PCA. Se ignora si heredabilidad_por_componente
        ya viene como array (se usa su longitud). Default 28, igual al
        espacio de rasgos de espacio_rasgos_rostro.py.
    semilla : int o None
        Semilla del generador aleatorio, para resultados reproducibles.

    Returns
    -------
    puntajes : np.ndarray, shape (n_personas, n_componentes)
        Puntaje poligénico de cada persona, por componente, en desvíos
        estándar del componente (mismas unidades que 'pesos' en
        reconstruir_cara()).
    """
    h2 = np.asarray(heredabilidad_por_componente, dtype=float)
    if h2.ndim == 0:
        h2 = np.full(n_componentes, float(h2))
    else:
        n_componentes = h2.shape[0]

    if np.any(h2 < 0) or np.any(h2 > HEREDABILIDAD_MAX):
        raise ValueError(
            f"heredabilidad_por_componente debe estar entre 0 y {HEREDABILIDAD_MAX} "
            f"(techo documentado en Xiong et al. 2025); recibido: "
            f"min={h2.min():.4f} max={h2.max():.4f}"
        )

    rng = np.random.default_rng(semilla)
    desvio = np.sqrt(h2)  # (n_componentes,)
    puntajes = rng.normal(loc=0.0, scale=1.0, size=(n_personas, n_componentes)) * desvio
    return puntajes


# ---------------------------------------------------------------------------
# Capa de conexión con el Punto 3 (BRIM)
# ---------------------------------------------------------------------------
# generar_genotipo_sintetico() da la parte genética por componente, pero BRIM
# (brim_rostro.py) necesita dos piezas separadas por persona: un puntaje
# poligénico ESCALAR (un solo número) y una cara OBSERVADA (genética + ruido
# ambiental, con varianza total 1 por componente, para que siga siendo
# compatible con reconstruir_cara()). Ninguna de las dos existía todavía en
# este módulo -- esto completa el Punto 2, no es una decisión del Punto 3.

def puntaje_escalar(genetica):
    """
    Colapsa la genética por componente (n_personas, k) en un único puntaje
    poligénico por persona (n_personas,): suma con pesos iguales entre
    componentes, estandarizada (media 0, desvío 1).

    Pesos iguales es una decisión de diseño documentada -- no hay en Xiong
    et al. (2025) una ponderación relativa entre componentes que justifique
    otra cosa.
    """
    suma = genetica.sum(axis=1)
    return (suma - suma.mean()) / suma.std()


def cara_observada(genetica, heredabilidad_por_componente, semilla=None):
    """
    Cara observada por persona (n_personas, k): la parte genética que ya
    simuló generar_genotipo_sintetico() más ruido ambiental independiente,
    de manera que cada componente quede con varianza total 1 (misma
    convención que 'pesos' en reconstruir_cara(): desvíos estándar del
    componente). El ruido ambiental es lo que la genética por sí sola no
    tiene -- generar_genotipo_sintetico() devuelve solo la señal genética,
    sin ese componente ambiental.
    """
    h2 = np.broadcast_to(
        np.asarray(heredabilidad_por_componente, dtype=float), (genetica.shape[1],)
    )
    rng = np.random.default_rng(semilla)
    return genetica + rng.standard_normal(genetica.shape) * np.sqrt(1 - h2)


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO — puntaje poligénico sintético por componente PCA")
    print("=" * 70)

    puntajes = generar_genotipo_sintetico(
        n_personas=5,
        heredabilidad_por_componente=HEREDABILIDAD_MAX,
        semilla=42,
    )
    print(f"\nForma del resultado: {puntajes.shape} (n_personas x n_componentes)")
    print(f"Desvío estándar esperado por componente: {np.sqrt(HEREDABILIDAD_MAX):.4f}")
    print(f"Desvío estándar observado (columna 0, 5 personas): {puntajes[:, 0].std():.4f}")
    print("\nPuntajes de la primera persona simulada (primeras 8 componentes):")
    print(np.round(puntajes[0, :8], 3))
