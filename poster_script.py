# 1. Librerias
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Rutas relativas
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos" / "cafe_arabica.xlsx"
RESULTADOS = RAIZ / "resultados"  
RESULTADOS.mkdir(exist_ok=True)  

# 3. Datos cargados
cafe = pd.read_excel(DATOS, sheet_name="cafe_arabica", index_col="fecha", parse_dates=True)
cafe = cafe.asfreq("MS") 
print(cafe.tail())
print("Observaciones:", len(cafe))
