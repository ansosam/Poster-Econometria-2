
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

# 4. Gráficos de la serie temporal 
sns.lineplot(x=cafe.index, y=cafe["precio"])
plt.title("Precio internacional del café arábigo (USD/kg, nominal)")
plt.xlabel("Fecha")
plt.ylabel("USD por kilo")
plt.savefig(RESULTADOS / "serie_original.png", dpi=300)
plt.show()
