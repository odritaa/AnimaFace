"""
Módulo Rostro — Espacio de rasgos faciales (PCA sobre el Basel Face Model)
============================================================================

Este script hace dos cosas:

1. explorar_bfm(path)      -> abre el archivo .h5 real del Basel Face Model
                              y muestra su estructura interna, para saber
                              exactamente dónde están la cara promedio,
                              los componentes principales y la varianza
                              de cada uno (el formato cambia un poco entre
                              versiones del modelo, por eso primero se explora
                              antes de asumir nombres de variables).

2. demo_sintetica()        -> genera datos falsos con la misma forma que
                              tendría el modelo real (una cara promedio +
                              variaciones), corre PCA sobre ellos, y grafica
                              la varianza acumulada — así podés ver la lógica
                              completa funcionando HOY, sin esperar la descarga.

Cuando subas el archivo real del Basel Face Model, corré explorar_bfm()
primero para ver su estructura, y ajustamos cargar_bfm_real() a los nombres
exactos que use esa versión.
"""

import numpy as np
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# PARTE 1 — para cuando tengas el archivo real
# ---------------------------------------------------------------------------

def explorar_bfm(path):
    """
    Recorre el archivo .h5 del Basel Face Model y muestra todos los grupos
    y datasets que contiene, con su forma (shape). Esto es siempre el primer
    paso con un archivo .h5 nuevo: no adivinamos la estructura, la miramos.
    """
    print(f"Explorando: {path}\n")

    def visitar(nombre, objeto):
        if isinstance(objeto, h5py.Dataset):
            print(f"  dataset: {nombre:50s} forma={objeto.shape}  tipo={objeto.dtype}")
        else:
            print(f"  grupo:   {nombre}")

    with h5py.File(path, "r") as f:
        f.visititems(visitar)


def cargar_bfm_real(path, grupo_shape="shape/model"):
    """
    Carga la cara promedio, los componentes principales (PCA basis) y la
    varianza de cada componente, desde el archivo real del Basel Face Model.

    IMPORTANTE: los nombres exactos de los datasets (mean, pcaBasis,
    pcaVariance) dependen de la versión del modelo (2009 / 2017 / 2019).
    Corré explorar_bfm(path) primero y ajustá 'grupo_shape' según lo que
    encuentres ahí.
    """
    with h5py.File(path, "r") as f:
        media = f[f"{grupo_shape}/mean"][:]
        base_pca = f[f"{grupo_shape}/pcaBasis"][:]
        varianza = f[f"{grupo_shape}/pcaVariance"][:]

    return media, base_pca, varianza


# ---------------------------------------------------------------------------
# PARTE 2 — demo con datos sintéticos, para validar la lógica hoy mismo
# ---------------------------------------------------------------------------

def demo_sintetica(n_caras=300, n_puntos=7150, semilla=42):
    """
    Simula un dataset con la MISMA FORMA que tendría el Basel Face Model:
    n_caras "personas", cada una descrita por n_puntos*3 coordenadas (x,y,z)
    de una malla facial. No son caras reales, son números aleatorios con
    una estructura de variación razonable — sirven solo para probar que el
    PCA y el análisis de varianza funcionan como esperamos.
    """
    rng = np.random.default_rng(semilla)

    # Simulamos unos pocos "modos de variación" reales ocultos (por ejemplo,
    # imaginate: ancho de cara, largo de nariz, prominencia del mentón...)
    # y generamos las caras como combinaciones de esos modos + ruido.
    n_modos_reales = 12
    modos_ocultos = rng.normal(size=(n_modos_reales, n_puntos * 3))
    pesos = rng.normal(size=(n_caras, n_modos_reales))

    datos = pesos @ modos_ocultos
    datos += rng.normal(scale=0.5, size=datos.shape)  # ruido de medición

    print(f"Datos sintéticos generados: {n_caras} caras × {n_puntos*3} coordenadas")
    return datos


def analizar_varianza(datos, max_componentes=50):
    """
    Corre PCA sobre los datos y muestra cuánta varianza explica cada
    componente y cuánta se acumula — esto es exactamente lo que vas a
    necesitar para decidir cuántas componentes retener del espacio de
    rasgos, con un criterio y no "porque sí".
    """
    pca = PCA(n_components=max_componentes)
    pca.fit(datos)

    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    print("\nComponente | Varianza individual | Varianza acumulada")
    for i in range(min(15, max_componentes)):
        print(f"  PC{i+1:<8} {varianza_explicada[i]*100:6.2f}%             {varianza_acumulada[i]*100:6.2f}%")

    # Componentes necesarios para llegar a distintos umbrales típicos
    for umbral in [0.90, 0.95, 0.98, 0.99]:
        alcanza = varianza_acumulada >= umbral
        if alcanza.any():
            n_necesarias = np.argmax(alcanza) + 1
            print(f"\nPara cubrir el {umbral*100:.0f}% de la varianza total, "
                  f"hacen falta {n_necesarias} componentes.")
        else:
            print(f"\nCon las {max_componentes} componentes calculadas no se llega "
                  f"al {umbral*100:.0f}% (se queda en {varianza_acumulada[-1]*100:.1f}%). "
                  f"Habría que calcular más componentes.")

    return pca, varianza_explicada, varianza_acumulada


def analizar_varianza_bfm(varianza):
    """
    Analiza la varianza YA CALCULADA que viene en el archivo del Basel Face
    Model (pcaVariance). A diferencia de analizar_varianza(), acá no corremos
    PCA de nuevo -- no tenemos ni tendríamos por qué tener los escaneos 3D
    originales, el modelo ya nos da los autovalores resultantes.
    """
    varianza_explicada = varianza / varianza.sum()
    varianza_acumulada = np.cumsum(varianza_explicada)

    print(f"\nComponentes disponibles en el modelo: {len(varianza)}")
    print("\nComponente | Varianza individual | Varianza acumulada")
    for i in range(15):
        print(f"  PC{i+1:<8} {varianza_explicada[i]*100:6.2f}%             {varianza_acumulada[i]*100:6.2f}%")

    for umbral in [0.90, 0.95, 0.98, 0.99]:
        alcanza = varianza_acumulada >= umbral
        if alcanza.any():
            n_necesarias = np.argmax(alcanza) + 1
            print(f"\nPara cubrir el {umbral*100:.0f}% de la varianza total, "
                  f"hacen falta {n_necesarias} componentes.")
        else:
            print(f"\nCon las {len(varianza)} componentes del modelo no se llega "
                  f"al {umbral*100:.0f}% (se queda en {varianza_acumulada[-1]*100:.1f}%).")

    return varianza_explicada, varianza_acumulada


def graficar_varianza(varianza_acumulada, path_salida, titulo="Varianza acumulada"):
    """Grafica la curva de varianza acumulada (el 'codo' que ayuda a decidir
    cuántas componentes retener)."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada * 100,
              marker="o", markersize=3, linewidth=1.5, color="#B7362D")
    for umbral in [90, 95, 98]:
        plt.axhline(umbral, color="#C9962E", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Cantidad de componentes (PCs)")
    plt.ylabel("Varianza acumulada explicada (%)")
    plt.title(titulo)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_salida, dpi=150)
    print(f"\nGráfico guardado en: {path_salida}")


# ---------------------------------------------------------------------------
# PARTE 3 — Punto 1 cerrado: decisiones tomadas y función de reconstrucción
# ---------------------------------------------------------------------------
#
# DECISIONES DOCUMENTADAS (para el Marco Teórico / Desarrollo de la tesis):
#   - Archivo usado: model2019_bfm.h5 (máscara clásica del Basel Face Model,
#     no la versión full-head ni la face12 -- evita el cuello/parte trasera
#     de la cabeza modelados con heurísticas, sin agregar otro archivo más).
#   - Componentes retenidas: 31 (cubren el 95% de la varianza total de forma).
#   - Normalización de escala: NO se aplica Procrustes adicional sobre el
#     espacio del modelo. Se declara como limitación conocida, no como
#     problema a resolver -- ver texto de Limitaciones ya redactado.

N_COMPONENTES = 31

def reconstruir_cara(pesos, media, base_pca, varianza, n_componentes=N_COMPONENTES):
    """
    Reconstruye una cara (vector de puntos 3D) a partir de un vector de
    pesos en el espacio de rasgos. pesos=0 en todas las componentes
    devuelve exactamente la cara promedio del modelo.

    'pesos' se interpreta en desvíos estándar de cada componente (por
    ejemplo, pesos[0]=2 significa "+2 desvíos estándar en la componente 1"),
    que es la convención estándar para moverse en un espacio PCA.
    """
    base_trunc = base_pca[:, :n_componentes]
    var_trunc = varianza[:n_componentes]
    desvio = np.sqrt(var_trunc)
    return media + base_trunc @ (pesos * desvio)


def verificar_reconstruccion(media, base_pca, varianza, n_componentes=N_COMPONENTES):
    """Chequeo de sanidad: pesos=0 debe devolver exactamente la media, y una
    variación moderada (2 desvíos en PC1) debe dar un desplazamiento de unos
    pocos milímetros, no algo absurdo. Se corre siempre antes de confiar en
    el espacio de rasgos para pasos posteriores del pipeline."""
    pesos_cero = np.zeros(n_componentes)
    cara_cero = reconstruir_cara(pesos_cero, media, base_pca, varianza, n_componentes)
    ok_media = np.allclose(cara_cero, media)

    pesos_pc1 = np.zeros(n_componentes)
    pesos_pc1[0] = 2.0
    cara_pc1 = reconstruir_cara(pesos_pc1, media, base_pca, varianza, n_componentes)
    desplazamiento = np.linalg.norm((cara_pc1 - media).reshape(-1, 3), axis=1)

    print(f"Pesos=0 reproduce la media exacta: {ok_media}")
    print(f"Desplazamiento promedio (PC1 +2 desvíos): {desplazamiento.mean():.2f} mm")
    print(f"Desplazamiento máximo: {desplazamiento.max():.2f} mm")
    return ok_media


def exportar_espacio_rasgos(media, base_pca, varianza, n_componentes, path_salida):
    """Guarda solo lo necesario (media + N componentes + su varianza) en un
    .npz liviano, para que el resto del pipeline (puntaje poligénico, BRIM,
    calibración) no tenga que cargar el archivo .h5 completo cada vez."""
    np.savez(path_salida,
              media=media,
              base_pca=base_pca[:, :n_componentes],
              varianza=varianza[:n_componentes],
              n_componentes=n_componentes)
    print(f"Espacio de rasgos exportado a: {path_salida}")


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO CON DATOS SINTÉTICOS (mientras llega el Basel Face Model real)")
    print("=" * 70)

    datos = demo_sintetica()
    pca, var_exp, var_acum = analizar_varianza(datos)
    graficar_varianza(var_acum, "/home/claude/varianza_acumulada.png")

    print("\n" + "=" * 70)
    print("Cuando tengas el archivo real, corré en su lugar:")
    print("  explorar_bfm('model2019_fullHead.h5')")
    print("  media, base_pca, varianza = cargar_bfm_real('model2019_fullHead.h5')")
    print("=" * 70)
