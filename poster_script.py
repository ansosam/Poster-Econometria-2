#%% Importar paquetes

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

import statsmodels.formula.api as smf
from scipy.stats import kruskal
from statsmodels.tsa.stattools import acf

#%% Rutas relativas
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
# Datos cargados
cafe = pd.read_csv(ruta_cafe,
                    sep=";",
                    index_col="fecha",
                    parse_dates=True,
                    date_format="%Y-%m"
)
cafe = cafe.asfreq("MS") 
print(cafe.tail())
print("Observaciones:", len(cafe))

# %% Creación de la serie de tiempo del precio del café

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

# %% METODOLOGÍA BOX-JENKINS
# =========================
# Paso 1: Identificación
# ============================
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
plt.savefig(RESULTADOS / "fac_facp_niveles.png", dpi=300)
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(cafe_serie) # Test de Dickey Fuller con intercepto y sin tendencia determinística

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

# En prueba KPSS se interpreta al contrario que una prueba ADF.
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
    print("KPSS: no rechazamos H0. Según el test, la serie es estacionaria.")

# Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie.
# %% Diferenciación de la serie (pues resultó NO estacionaria)
cafe_serie_diff = cafe_serie.diff().dropna()

# Gráfica de la diferencia del precio del café
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_diff, color=COLORES["diferencia"])
plt.title("Serie diferenciada")
plt.xlabel("Año")
plt.ylabel("Cambio en USD/kg")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "serie_diferenciada.png", dpi=300)
plt.show()
# %% Diferencia del logaritmo

cafe_serie_log_diff = np.log(cafe_serie).diff().dropna()

# Gráfica de la diferencia del logaritmo del precio del café
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_log_diff, color=COLORES["log_diff"])
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Año")
plt.ylabel("Δ log(precio)")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "log_diferencia.png", dpi=300)
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
plt.savefig(RESULTADOS / "fac_facp_log_diferencia.png", dpi=300)
plt.show()

# %% Identificación del modelo usando FAC y FACP

# La FACP solo tiene significativo el primer rezago y luego se corta.
# La FAC decae gradualmente.
# Por tanto, se propone inicialmente un modelo AR(1) sobre la serie
# transformada, equivalente a un ARIMA(1, 1, 0) sobre log(cafe_serie).
#%% Identificación mediante criterios de Información
# 
candidatos = [(1, 1, 0), (0, 1, 1), (0, 1, 2), (1, 1, 1)]
filas = []
for orden in candidatos:
    est = SARIMAX(
        np.log(cafe_serie),
        order=orden,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
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
#Los cuatro modelos pasan la prueba de Ljung-Box a 12 y 24 rezagos (p > 0.05).

# %% =========================
# Paso 1 (continuación): Prueba de estacionalidad
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
    "Estacionalidad del precio del café arábigo (1990–2026)\n"
    "Variación logarítmica mensual agrupada por mes del año",
    fontsize=12,
)
plt.xlabel("Mes")
plt.ylabel("Δ log(precio)")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "estacionalidad_boxplot.png", dpi=300)
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
# heterocedasticidad (la prueba ARCH del Paso 3 muestra volatilidad agrupada)
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

# No supone normalidad, lo cual es útil porque Jarque-Bera (Paso 3) la rechaza.
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

# Objeto tipo "SARIMAX" del paquete statsmodels
print(type(modelo_ar1))

#  Se realiza la estimación del modelo AR(1)
estimacion_ar1 = modelo_ar1.fit(disp=False)

# El método de estimación es máxima verosimilitud 
#       sobre la representación del modelo en un espacio de estados

# Objeto tipo "SARIMAXResultsWrapper"
print(type(estimacion_ar1))

# Se imprimen los resultados principales de la estimación
# Como lo son los coeficientes y significancia de los coeficientes
print(estimacion_ar1.summary())
# %% =========================
# Paso 3: Validación de supuestos
# ============================

# Residuales del modelo
residuales = estimacion_ar1.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
print(type(residuales))

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:], color=COLORES["residuales"])  # Se quita el primer residual (inicialización)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Residuales del modelo ARIMA(1,1,0)")
plt.xlabel("Fecha")
plt.ylabel("Residual de Δ log(precio)")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "residuales.png", dpi=300)
plt.show()

# Descripción de los residuales
print(residuales.describe())

# %% FAC y FACP de los residuales

# Si el modelo capturó toda la dinámica, ninguna barra debería salir
# de la banda de confianza (los residuales serían ruido blanco).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales.iloc[1:],  # Se quita el primer residual (inicialización)
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

colorear_correlograma(axes[0], COLORES["fac"])
axes[0].set_title("FAC de los residuales del ARIMA(1,1,0)")

plot_pacf(
    residuales.iloc[1:],
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

colorear_correlograma(axes[1], COLORES["facp"])
axes[1].set_title("FACP de los residuales del ARIMA(1,1,0)")

plt.tight_layout()
plt.savefig(RESULTADOS / "fac_facp_residuales.png", dpi=300)
plt.show()

# Casi todas las barras están dentro de la banda (±0.093). Solo los rezagos 16
#       (FAC y FACP) y 17 (FACP) la tocan apenas, algo esperable por azar con
#       24 rezagos al 5%. La prueba de Ljung-Box confirma que no hay
#       autocorrelación conjunta.

# %% Pruebas sobre los residuales

res_validos = residuales.iloc[1:]  # Se quita el primer residual (inicialización)

# Prueba de Ljung-Box (autocorrelación)
# H0: los residuales no tienen autocorrelación (ruido blanco).
ljung_box = acorr_ljungbox(res_validos, lags=[12, 24], model_df=1, return_df=True)

print("=== Prueba de Ljung-Box sobre los residuales ===")
print(ljung_box)

if (ljung_box["lb_pvalue"] < 0.05).any():
    print("Se rechaza H0: los residuales tienen autocorrelación.")
else:
    print("No se rechaza H0: los residuales no tienen autocorrelación.")

# Prueba de Jarque-Bera (normalidad)
# H0: los residuales son normales.
jb = jarque_bera(res_validos)

print("\n=== Prueba de Jarque-Bera sobre los residuales ===")
print("Estadístico JB:", jb.statistic)
print("p-valor:", jb.pvalue)

if jb.pvalue < 0.05:
    print("Se rechaza H0: los residuales no son normales.")
else:
    print("No se rechaza H0: los residuales son normales.")

# Prueba ARCH-LM (heterocedasticidad condicional)
# H0: no hay efectos ARCH (la varianza no depende de los choques pasados).
arch_lm, arch_p, _, _ = het_arch(res_validos, nlags=12)

print("\n=== Prueba ARCH-LM sobre los residuales (12 rezagos) ===")
print("Estadístico LM:", arch_lm)
print("p-valor:", arch_p)

if arch_p < 0.05:
    print("Se rechaza H0: hay efectos ARCH (volatilidad agrupada).")
else:
    print("No se rechaza H0: no hay efectos ARCH.")

# %% =========================
# Paso 4: Pronóstico
# ============================

# A partir de la estimación del AR(1) se realizó el pronóstico

# Pronóstico 12 pasos adelante
pronostico_log = estimacion_ar1.get_forecast(steps=12)

# Pronóstico puntual e intervalos de predicción del logaritmo del precio del café
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
plt.plot(cafe_serie, label="Datos históricos", color=COLORES["original"])
plt.plot(pronostico_nivel, label="Pronóstico", color=COLORES["diferencia"])
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color=COLORES["diferencia"],
    alpha=0.3,
    label="Intervalo de confianza"
)
plt.title("Pronóstico del precio del café arábigo a 12 meses (ARIMA(1,1,0))")
plt.xlabel("Fecha")
plt.ylabel("USD por kilo")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "pronostico.png", dpi=300)
plt.show()

# %% =========================
# Intervalos de predicción por bootstrap
# ============================

# Los intervalos anteriores suponen errores normales. Como la prueba de
# Jarque-Bera rechaza la normalidad, se construyen
# intervalos remuestreando los residuales del propio modelo
#   1. Se toman los residuales estimados y se centran en cero.
#   2. Se simulan B trayectorias futuras del AR(1) sobre la variación logarítmica:
#      r_{T+h} = phi * r_{T+h-1} + e*, con e* sacado con reemplazo de los residuales.
#   3. Se acumulan las variaciones sobre el último log(precio) observado.
#   4. Los percentiles 2.5% y 97.5% de las trayectorias son el intervalo del 95%.

# Este bootstrap mantiene fijo el coeficiente estimado (phi), así que
#       no incluye la incertidumbre de la estimación de los parámetros.

pasos = len(pronostico_nivel)  # Mismo horizonte que el pronóstico anterior
B_pron = 5000  # Número de trayectorias simuladas
rng_pron = np.random.default_rng(2026)  # Semilla para que el resultado sea reproducible

phi_pron = estimacion_ar1.params["ar.L1"]
residuos_boot = residuales.iloc[1:].values  # Se quita el primer residual (inicialización)
residuos_boot = residuos_boot - residuos_boot.mean()

log_precio = np.log(cafe_serie)
ultimo_log = log_precio.iloc[-1]
ultima_variacion = log_precio.diff().iloc[-1]

# Cada fila es una trayectoria y cada columna un mes del horizonte
choques = rng_pron.choice(residuos_boot, size=(B_pron, pasos), replace=True)
variaciones = np.empty((B_pron, pasos))
variaciones[:, 0] = phi_pron * ultima_variacion + choques[:, 0]
for h in range(1, pasos):
    variaciones[:, h] = phi_pron * variaciones[:, h - 1] + choques[:, h]

trayectorias = np.exp(ultimo_log + np.cumsum(variaciones, axis=1))

pronostico_boot = pd.Series(trayectorias.mean(axis=0), index=pronostico_nivel.index)
intervalos_boot = pd.DataFrame({
    "inferior": np.percentile(trayectorias, 2.5, axis=0),
    "superior": np.percentile(trayectorias, 97.5, axis=0),
}, index=pronostico_nivel.index)

tabla_bootstrap = pd.DataFrame({
    "pronostico_normal": pronostico_nivel,
    "inferior_normal": intervalos_nivel.iloc[:, 0],
    "superior_normal": intervalos_nivel.iloc[:, 1],
    "pronostico_bootstrap": pronostico_boot,
    "inferior_bootstrap": intervalos_boot["inferior"],
    "superior_bootstrap": intervalos_boot["superior"],
}).round(2)
tabla_bootstrap.index = tabla_bootstrap.index.strftime("%Y-%m")

print("=== Pronóstico con intervalos normales y por bootstrap (USD/kg) ===")
print(tabla_bootstrap.to_string())

# Gráfica: últimos 10 años + pronóstico con los dos intervalos
historia_reciente = cafe_serie.loc[cafe_serie.index >= cafe_serie.index[-1] - pd.DateOffset(years=10)]

plt.figure(figsize=(12, 6))
plt.plot(historia_reciente, color=COLORES["original"], linewidth=2, label="Precio observado")
plt.plot(pronostico_nivel, color="black", linewidth=2, linestyle="--", label="Pronóstico")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color=COLORES["banda"],
    alpha=0.25,
    label="IC 95% normal",
)
plt.plot(intervalos_boot["inferior"], color=COLORES["fac"], linewidth=1.5, label="IC 95% bootstrap")
plt.plot(intervalos_boot["superior"], color=COLORES["fac"], linewidth=1.5)
plt.axvline(cafe_serie.index[-1], color=COLORES["banda"], linewidth=1, linestyle=":")
plt.title(
    "Pronóstico del precio del café arábigo a 12 meses\n"
    f"Intervalo normal frente a intervalo por bootstrap de residuales ({B_pron} trayectorias)",
    fontsize=12,
)
plt.xlabel("Fecha")
plt.ylabel("USD por kilo")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "pronostico_bootstrap.png", dpi=300)
plt.show()
