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




#%% Rutas relativas
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
ruta_cafe = DATOS / "cafe_arabica.csv" #ruta al archivo de datos   
RESULTADOS = RAIZ / "resultados"  #carpeta para guardar resultados
RESULTADOS.mkdir(exist_ok=True)  

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

<<<<<<< Updated upstream
# %% METODOLOGÍA BOX-JENKINS
# =========================
# Paso 1: Identificación
# ============================
=======
# %%METODOLOGÍA BOX- JENKINS
#1. Identificación
##############################
>>>>>>> Stashed changes
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
plt.savefig(RESULTADOS / "fac_facp_niveles.png", dpi=300)
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
<<<<<<< Updated upstream
adf_result = adfuller(cafe_serie) # Test de Dickey Fuller con intercepto y sin tendencia determinística
=======
adf_result = adfuller(cafe_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
# En prueba KPSS se interpreta al contrario que una prueba ADF.
#       Si no se rechaza la H0, la serie es estacionaria
#       y si se rechaza la H0, la serie es no estacionaria. 
=======
# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 
>>>>>>> Stashed changes

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

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie.
# %% Diferenciación de la serie (pues resultó NO estacionaria)
cafe_serie_diff = cafe_serie.diff().dropna()

# Gráfica de la diferencia del precio del café
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_diff)
plt.title("Serie diferenciada")
plt.xlabel("Año")
<<<<<<< Updated upstream
plt.ylabel("Cambio en USD/kg")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "serie_diferenciada.png", dpi=300)
=======
plt.ylabel("Precio")
plt.grid(True)
>>>>>>> Stashed changes
plt.show()
# %% Diferencia del logaritmo

cafe_serie_log_diff = np.log(cafe_serie).diff().dropna()

# Gráfica de la diferencia del logaritmo del precio del café
plt.figure(figsize=(10, 5))
plt.plot(cafe_serie_log_diff)
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Año")
<<<<<<< Updated upstream
plt.ylabel("Δ log(precio)")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTADOS / "log_diferencia.png", dpi=300)
=======
plt.ylabel("Precio")
plt.grid(True)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
#Los cuatro modelos pasan la prueba de Ljung-Box a 12 y 24 rezagos (p > 0.05).

# %% =========================
# Paso 1 (continuación): Prueba de estacionalidad
# ============================
=======
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

# %%FAC y FACP de los residuales
>>>>>>> Stashed changes

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
<<<<<<< Updated upstream
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
=======

axes[0].set_title("FAC de los residuales")

plot_pacf(
    residuales,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales")

plt.tight_layout()
>>>>>>> Stashed changes
plt.show()

# %% Prueba Ljung-Box

<<<<<<< Updated upstream
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
=======
ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24],
    return_df=True
>>>>>>> Stashed changes
)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.

# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

<<<<<<< Updated upstream
# No supone normalidad, lo cual es útil porque Jarque-Bera (Paso 3) la rechaza.
# H0: la variación tiene la misma distribución en todos los meses.
kw = kruskal(*grupos)
=======
plot_acf(
    residuales**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
>>>>>>> Stashed changes

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

<<<<<<< Updated upstream
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
=======
axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
plt.show()

# %%Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.

# %% Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
probplot(residuales.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()

# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.

# %%=========================
# PASO 4: Pronóstico
# ============================

# A partir de la estimación del AR(1) realizó el pronóstico

# Protonóstico 10 pasos adelante
pronostico_log = estimacion_ar1.get_forecast(steps=10)

# Pronóstico puntual e intervalos de predicción del logaritmo de las exportaciones tradicionales
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
plt.plot(cafe_serie, label="Datos históricos", color=COLORES["original"])
plt.plot(pronostico_nivel, label="Pronóstico", color=COLORES["diferencia"])
=======
plt.plot(cafe_serie, label="Datos históricos")
plt.plot(pronostico_nivel, label="Pronóstico", color="orange")
>>>>>>> Stashed changes
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
<<<<<<< Updated upstream
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
=======
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza"
)
# %%
>>>>>>> Stashed changes
