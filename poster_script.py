#%% Importar paquetes

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss




#%% 2. Rutas relativas
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
ruta_cafe = DATOS / "cafe_arabica.csv" #ruta al archivo de datos   
RESULTADOS = RAIZ / "resultados"  #carpeta para guardar resultados
RESULTADOS.mkdir(exist_ok=True)  

#%% Cargar Datos
# 3. Datos cargados
cafe = pd.read_csv(ruta_cafe,
                    sep=";",
                    index_col="fecha",
                    parse_dates=True,
                    date_format="%Y-%m"
)
cafe = cafe.asfreq("MS") 
print(cafe.tail())
print("Observaciones:", len(cafe))

# %% Creación de la serie de tiempo de "exportaciones"

# La nueva serie de tiempo se va a llamar "cafe_serie" y va a tener valores numéricos
cafe_serie = cafe["precio"].copy()
cafe_serie = pd.to_numeric(cafe_serie, errors="coerce")
cafe_serie = cafe_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(cafe_serie.head())
print(cafe_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(cafe_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(cafe_serie.describe())

# %%METODOLOGÍA BOX- JENKINS
#1. Identificación
##############################
#Gráficos de la serie temporal 
plt.figure(figsize=(12, 6))
plt.plot(cafe_serie, linewidth=2)
plt.title("Precio internacional del café arábigo (USD/kg, nominal)")
plt.xlabel("Fecha")
plt.ylabel("USD por kilo")
plt.grid(True)
plt.savefig(RESULTADOS / "serie_original.png", dpi=300)
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    cafe_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC Precios Históricos del Café Arábigo")

plot_pacf(
    cafe_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP Precios Históricos del Café Arábigo")

plt.tight_layout()
plt.show()
# %%
# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(cafe_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(cafe_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie.
# %% Diferenciación de la serie (pues resultó NO estacionaria)
cafe_serie_diff = cafe_serie.diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_diff)
plt.title("Serie diferenciada")
plt.xlabel("Año")
plt.ylabel("Precio")
plt.grid(True)
plt.show()
# %% Diferencia del logaritmo

cafe_serie_log_diff = np.log(cafe_serie).diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia del logaritmo de exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_log_diff)
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Año")
plt.ylabel("Precio")
plt.grid(True)
plt.show()

# %% FAC y FACP de la diferencia del logaritmo

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    cafe_serie_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(
    cafe_serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la diferencia del logaritmo")

plt.tight_layout()
plt.show()

# %% Identificación del modelo usando FAC y FACP

# La FAC muestra que solo la primera autocorrelación es significativa.
# La FACP decae rápidamente.
# Por tanto, se propone inicialmente un modelo AR(1) sobre la serie
# transformada, equivalente a un ARIMA(1, 1, ) sobre log(cafe_serie).
#%% Identificación mediante criterios de Información
# 
candidatos = [(1, 1, 0), (0, 1, 1), (0, 1, 2), (1, 1, 1)]
filas = []
for orden in candidatos:
    est = SARIMAX(np.log(cafe_serie), order=orden, trend="n").fit(disp=False)
    res = est.resid.iloc[1:]
    lb = acorr_ljungbox(res, lags=[12, 24], model_df=orden[0] + orden[2], return_df=True)
    filas.append({
        "modelo": f"ARIMA{orden}",
        "AIC": round(est.aic, 1),
        "BIC": round(est.bic, 1),
        "LB p (12)": round(lb["lb_pvalue"].iloc[0], 3),
        "LB p (24)": round(lb["lb_pvalue"].iloc[1], 3),
    })
print(pd.DataFrame(filas).sort_values("BIC"))
#%% Identificación mediante criterios de Información
#El modelo ARIMA(1, 1, 0) es el que tiene el menor BIC y AIC
#Sin embargo, ningún modelo pasa la prueba de Ljung-Box a 12 y 24 rezagos.
# %% =========================
# Paso 2: Estimación
# ============================
modelo_ar1 = SARIMAX(
    np.log(cafe_serie),
    order=(1, 1, 0),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Objeto tipo "SARIMAX" del paquete statmodels
type(modelo_ar1)

#  Acá es donde se realiza la estimación del modelo AR(1)
estimacion_ar1 = modelo_ar1.fit(disp=False)

# Nota: El método de estimación es máxima verosimilitud 
#       sobre la representación del modelo en un espacio de estados

# Objeto tipo "SARIMAXResultsWrapper"
type(estimacion_ar1)

# Se imprimen los resultados principales de la estimación
# E.g. Coeficientes y significancia de los coeficientes
print(estimacion_ar1.summary())
# %%# =========================
# Paso 3: Validación de supuestos
# ============================

# Residuales del modelo
residuales = estimacion_ar1.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:-1])
plt.title("Residuales del modelo AR(1)")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales.describe())

# %%
