import sys
from pathlib import Path

# Los módulos del proyecto (brim_rostro.py, puntaje_poligenico.py) viven en la
# raíz del repo, un nivel arriba de tests/ -- hace falta en sys.path para
# poder importarlos desde aquí.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
