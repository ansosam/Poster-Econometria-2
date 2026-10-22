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

import statsmodels.formula.api as smf
from scipy.stats import kruskal
from statsmodels.tsa.stattools import acf

#%% 2. Rutas relativas
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
ruta_cafe = DATOS / "cafe_arabica.csv" #ruta al archivo de datos   
RESULTADOS = RAIZ / "resultados"  #carpeta para guardar resultados
RESULTADOS.mkdir(exist_ok=True)  

#%% Paleta de colores para las gráficas
COLORES = {
    "original": "#6F4E37",    # café
    "diferencia": "#2E86AB",  # azul
    "log_diff": "#3B8B5A",    # verde
    "fac": "#C0392B",         # rojo
    "facp": "#8E44AD",        # morado
    "residuales": "#E67E22",  # naranja
    "estacional": "#16A085",  # verde azulado
    "banda": "#7F8C8D",       # gris para bandas de confianza
}


def colorear_correlograma(ax, color):
    """Aplica un color a las barras, marcadores y banda de confianza de un FAC/FACP."""
    for linea in ax.lines:
        if linea.get_marker() not in (None, "None", ""):
            linea.set_color(color)
            linea.set_markerfacecolor(color)
            linea.set_markeredgecolor(color)
        else:  # línea horizontal en cero
            linea.set_color("black")
            linea.set_linewidth(0.8)
    for coleccion in ax.collections:
        if coleccion.__class__.__name__ == "LineCollection":
            coleccion.set_color(color)
        else:  # banda de confianza
            coleccion.set_facecolor(COLORES["banda"])
            coleccion.set_alpha(0.2)

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

# Serie de tiempo con el nombre "cafe_serie" 
cafe_serie = cafe["precio"].copy()
cafe_serie = pd.to_numeric(cafe_serie, errors="coerce")
cafe_serie = cafe_serie.dropna() # Borrar missing values

# Principio y final de la serie de tiempo
print(cafe_serie.head())
print(cafe_serie.tail())

# # El tipo de objeto de la base de datos se transforma en un Pandas.Series
print(type(cafe_serie))

# Estadísticas descriptivas de la serie de tiempo 
print(cafe_serie.describe())

# %% 4. METODOLOGÍA BOX- JENKINS
#1. Identificación
##############################
#Gráficos de la serie temporal 
plt.figure(figsize=(12, 6))
plt.plot(cafe_serie, linewidth=2, color=COLORES["original"])
plt.title("Precio internacional del café arábigo (USD/kg, nominal)")
plt.xlabel("Fecha")
plt.ylabel("USD por kilo")
plt.grid(True, alpha=0.3)
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

colorear_correlograma(axes[0], COLORES["fac"])
axes[0].set_title("FAC Precios Históricos del Café Arábigo")

plot_pacf(
    cafe_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

colorear_correlograma(axes[1], COLORES["facp"])
axes[1].set_title("FACP Precios Históricos del Café Arábigo")

plt.tight_layout()
plt.show()
# %%
# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(cafe_serie) # Se utiliza el Test de Dickey Fuller sin intercepto y tendencia determínistica

# En el test de ADF, si no se rechaza la H0, entonces la serie no es estacionaria
#       y si se rechaza la H0 la serie es estacionaria. 

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

# En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no se rechaza la H0, la serie es estacionaria
#       y si se rechaza la H0, la serie es no estacionaria. 

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

# Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie.
# %% Diferenciación de la serie (pues resultó NO estacionaria)
cafe_serie_diff = cafe_serie.diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_diff, color=COLORES["diferencia"])
plt.title("Serie diferenciada")
plt.xlabel("Año")
plt.ylabel("Precio")
plt.grid(True, alpha=0.3)
plt.show()
# %% Diferencia del logaritmo

cafe_serie_log_diff = np.log(cafe_serie).diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia del logaritmo de exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_log_diff, color=COLORES["log_diff"])
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Año")
plt.ylabel("Precio")
plt.grid(True, alpha=0.3)
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

colorear_correlograma(axes[0], COLORES["fac"])
axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(
    cafe_serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

colorear_correlograma(axes[1], COLORES["facp"])
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
#A 12 rezagos, tres de los cuatro modelos pasan la prueba de Ljung-Box (todos menos ARIMA(0, 1, 1)).
#Sin embargo, ningún modelo pasa la prueba de Ljung-Box a 24 rezagos.

# %% =========================
# 5. Prueba de estacionalidad
# ============================

# Variación logarítmica mensual (serie estacionaria)
dlog = np.log(cafe_serie).diff().dropna()

meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# %% Gráfico por mes: distribución de la variación en cada mes del año

grupos = [dlog[dlog.index.month == m] for m in range(1, 13)]

plt.figure(figsize=(10, 5))
plt.boxplot(
    grupos,
    patch_artist=True,
    boxprops=dict(facecolor=COLORES["estacional"], alpha=0.6, edgecolor=COLORES["estacional"]),
    medianprops=dict(color="black"),
    whiskerprops=dict(color=COLORES["estacional"]),
    capprops=dict(color=COLORES["estacional"]),
    flierprops=dict(markeredgecolor=COLORES["estacional"]),
)
plt.xticks(range(1, 13), meses)
plt.axhline(0, color="gray", linewidth=1)
plt.title(
    "Estacionalidad del precio del café arábigo (1960–2026)\n"
    "Variación logarítmica mensual agrupada por mes del año",
    fontsize=12,
)
plt.xlabel("Mes")
plt.ylabel("Δ log(precio)")
plt.grid(True, alpha=0.3)
plt.show()

# Promedio de la variación por mes
print(dlog.groupby(dlog.index.month).mean().set_axis(meses).round(4))

# %% Autocorrelaciones en los rezagos estacionales

fac = acf(dlog, nlags=36)
print("FAC en rezagos estacionales:")
for rezago in [12, 24, 36]:
    print(f"  Rezago {rezago}: {fac[rezago]:.3f}")
print(f"  Banda aproximada al 95%: ±{1.96 / np.sqrt(len(dlog)):.3f}")

# Si hubiera una raíz unitaria estacional, la FAC decaería lentamente
#    en los rezagos 12, 24, 36... Pero los valores pequeños indican que no hace
#    falta una diferencia estacional (D = 0).

# %% Prueba F: regresión de la variación sobre dummies mensuales

datos = pd.DataFrame({"dlog": dlog, "mes": dlog.index.month})

# Errores estándar HAC (Newey-West): robustos a autocorrelación y a
# heterocedasticidad (la prueba ARCH mostró volatilidad agrupada)
reg_mes = smf.ols("dlog ~ C(mes)", data=datos).fit(
    cov_type="HAC", cov_kwds={"maxlags": 12}
)

# H0: todos los efectos de los meses son cero (no hay estacionalidad)
R = np.eye(len(reg_mes.params))[1:]   # todos los coeficientes excepto el intercepto
prueba_f = reg_mes.f_test(R)

print("=== Prueba F de dummies mensuales ===")
print(prueba_f)

if float(prueba_f.pvalue) < 0.05:
    print("Se rechaza H0: hay evidencia de estacionalidad.")
else:
    print("No se rechaza H0: no hay evidencia de estacionalidad.")

# %% Prueba de Kruskal-Wallis (no paramétrica)

# No supone normalidad, lo cual es útil porque Jarque-Bera la rechazó.
# H0: la variación tiene la misma distribución en todos los meses.
kw = kruskal(*grupos)

print("=== Prueba de Kruskal-Wallis ===")
print("Estadístico H:", kw.statistic)
print("p-valor:", kw.pvalue)

if kw.pvalue < 0.05:
    print("Se rechaza H0: hay evidencia de estacionalidad.")
else:
    print("No se rechaza H0: no hay evidencia de estacionalidad.")

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

#  Se realiza la estimación del modelo AR(1)
estimacion_ar1 = modelo_ar1.fit(disp=False)

# El método de estimación es máxima verosimilitud 
#       sobre la representación del modelo en un espacio de estados

# Objeto tipo "SARIMAXResultsWrapper"
type(estimacion_ar1)

# Se imprimen los resultados principales de la estimación
# Como lo son los coeficientes y significancia de los coeficientes
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
plt.plot(residuales.iloc[1:-1], color=COLORES["residuales"])
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Residuales del modelo AR(1)")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.grid(True, alpha=0.3)
plt.show()

# Descripción de los residuales 
print(residuales.describe())

# %%# %% =========================
# PASO 3: Pronóstico
# ============================

# A partir de la estimación del AR(1) se realizó el pronóstico

# Protonóstico 12 pasos adelante
pronostico_log = estimacion_ar1.get_forecast(steps=12)

# Pronóstico puntual e intervalos de predicción del logaritmo de las exportaciones tradicionales
pronostico_puntual_log = pronostico_log.predicted_mean
varianza_pronostico_log = pronostico_log.var_pred_mean
intervalos_log = pronostico_log.conf_int()

# Al retransformar desde logaritmos, exp(E[log(y)]) corresponde a la mediana.
# Para obtener la media en niveles se usa la corrección por sesgo:
# E[y] = exp(mu + 0.5 * sigma^2), bajo normalidad en la escala logarítmica.
pronostico_nivel = np.exp(pronostico_puntual_log + 0.5 * varianza_pronostico_log)
intervalos_nivel = np.exp(intervalos_log)

tabla_pronostico = pd.DataFrame({
    "pronostico": pronostico_nivel,
    "limite_inferior": intervalos_nivel.iloc[:, 0],
    "limite_superior": intervalos_nivel.iloc[:, 1]
})

print(tabla_pronostico)

# Gráfica del pronóstico
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie, label="Datos históricos")
plt.plot(pronostico_nivel, label="Pronóstico", color="orange")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza"
)
