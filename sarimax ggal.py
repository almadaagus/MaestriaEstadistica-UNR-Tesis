'''
Bibliografia: https://cienciadedatos.net/documentos/py51-modelos-arima-sarimax-python
https://towardsdatascience.com/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6
'''
# %% [Sección 1: LIBRERIAS]
#### LIBRERIAS ####
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
import numpy as np
import itertools
#graficos
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import KBinsDiscretizer
# statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
#scipy
from scipy.stats import jarque_bera
import scipy.stats as stats
from scipy.stats import shapiro
# %% [Sección 2: DATOS]
empresa = 'GGAL'
archivo = "K:/Rosental/Comun/Bolsa/Mesa Operaciones/Agustin/datos - ggal.xlsx"
df = pd.read_excel(archivo, sheet_name= "BD GGAL")
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
df = df.set_index('fecha')
df = df.sort_values(by='fecha')
fechas_post_eecc = df['fecha_post_eecc']
# Elimino la fecha de las paso de MACRI para probar. Fecha: 12-8-19
#fecha_a_eliminar = '2019-08-12'
#df = df.drop(fecha_a_eliminar)

variables_contables = ["PX_TO_EBITDA","EBITDA_TO_REVENUE", "CF_NET_INC", "IS_COMP_NET_INCOME", "SALES_REV_TURN","IS_OPER_INC" ,"EARN_FOR_COM_TO_TOT_REV"]
otras_variables = ["YIELD GOVT 10 PRE BALANCE", "SPY_pre_eecc" , empresa+"_pre_eecc"]
datos_eecc = df[variables_contables]

# USO DATOS DE YFINANCE Y NO DE BLOOMBERG
start_date = df['fecha_pre_eecc'].min() +  BDay(-1)
end_date = df['fecha_post_eecc'].max() + BDay(1)
datos_spy =     yf.download('SPY',   start= start_date, end=end_date)['Adj Close']
datos_empresa = yf.download(empresa, start= start_date, end=end_date)['Adj Close']

df['SPY_pre_eecc'] = datos_spy.reindex(df['fecha_pre_eecc']).values
df[empresa + '_post_eecc'] = datos_empresa.reindex(df['fecha_post_eecc']).values
df[empresa + '_pre_eecc'] = datos_empresa.reindex(df['fecha_pre_eecc']).values

#ELIMINO MAS COLUMNAS QUE NO SON EXPLICATIVAS
df = df.drop(columns=["hora","fecha_pre_eecc","fecha_post_eecc"])

# creo variables interacciones
#df['YIELD_i_SPY'] = df["YIELD GOVT 10 PRE BALANCE"] * df["SPY_pre_eecc"]
#otras_variables.append('YIELD_i_SPY')
#df['GGAL_i_SPY'] = df[empresa+"_pre_eecc"] * df["SPY_pre_eecc"]
#otras_variables.append('GGAL_i_SPY')
#df['YIELD_i_GGAL'] = df["YIELD GOVT 10 PRE BALANCE"] * df[empresa+"_pre_eecc"]
#otras_variables.append('YIELD_i_GGAL')


# SEPARO EN TRAIN - TEST
datos_train, datos_test = train_test_split(df, test_size=0.2, shuffle=False)

exogenas_train = datos_train.drop(columns=[empresa+"_post_eecc"])
endogena_train = datos_train[empresa+"_post_eecc"]
exogenas_test = datos_test.drop(columns=[empresa+"_post_eecc"])
endogena_test = datos_test[empresa+"_post_eecc"]

#boxplot antes de imputar outliers
flierprops = dict(marker='o', color='red', markersize=1)
exogenas_train[variables_contables].boxplot(flierprops=flierprops)
plt.title('Boxplots de las Variables de los EECC antes de imputar outliers')
plt.xlabel('Variables')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')

# IMPUTO OUTLIERS POR KNN
# Filtrar las columnas de 'variables_contables' tanto en exogenas_train como en exogenas_test
exogenas_train_contables = exogenas_train[variables_contables]
exogenas_test_contables = exogenas_test[variables_contables]

# Paso 1: Calcular Q1, Q3 e IQR solo para las columnas de variables_contables
Q1 = exogenas_train_contables.quantile(0.25)
Q3 = exogenas_train_contables.quantile(0.75)
IQR = Q3 - Q1
# Paso 2: Detectar outliers solo en esas columnas
outlier_threshold = 2
outliers_train = (exogenas_train_contables < (Q1 - outlier_threshold * IQR)) | (exogenas_train_contables > (Q3 + outlier_threshold * IQR))
outliers_test = (exogenas_test_contables < (Q1 - outlier_threshold * IQR)) | (exogenas_test_contables > (Q3 + outlier_threshold * IQR))
# Paso 3: Reemplazar los outliers por NaN solo en esas columnas
exogenas_train_contables[outliers_train] = np.nan
exogenas_test_contables[outliers_test] = np.nan
# Paso 4: Imputar los valores faltantes y outliers con KNNImputer solo en las columnas de variables_contables
imputador_knn = KNNImputer()
# Imputar en exogenas_train_contables y conservar el índice original
exogenas_train_contables_imputadas = pd.DataFrame(imputador_knn.fit_transform(exogenas_train_contables), 
                                                   columns=variables_contables, 
                                                   index=exogenas_train_contables.index)
# Imputar en exogenas_test_contables (usando el modelo ajustado en exogenas_train_contables) y conservar el índice original
exogenas_test_contables_imputadas = pd.DataFrame(imputador_knn.transform(exogenas_test_contables), 
                                                  columns=variables_contables, 
                                                  index=exogenas_test_contables.index)
# Paso 5: Actualizar las columnas imputadas en los DataFrames originales
exogenas_train[variables_contables] = exogenas_train_contables_imputadas
exogenas_test[variables_contables] = exogenas_test_contables_imputadas

# Boxplots de las variables
flierprops = dict(marker='o', color='red', markersize=1)
exogenas_train[variables_contables].boxplot(flierprops=flierprops)
plt.title('Boxplots de las Variables de los EECC luego de imputar outliers')
plt.xlabel('Variables')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')

# INTERACCIÓN DE VARIABLES: Analisis visual
def graficar_con_variable_fija(df, empresa, pares_variables, n_bins=4):
    var_y = empresa + "_post_eecc"  # Variable Y fija
    for i, (var_x, var_color) in enumerate(pares_variables):
        # Crear bins para la variable de color
        df[f'{var_color}_binned'] = pd.cut(df[var_color], bins=n_bins, labels=False)
        
        # Crear el gráfico de dispersión
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=var_x, y=var_y, hue=f'{var_color}_binned', palette="Set1", s=100)
        
        # Personalizar gráfico
        plt.title(f'Gráfico {i+1}: {var_y} vs {var_x} (Color por {var_color})', fontsize=14)
        plt.xlabel(var_x, fontsize=12)
        plt.ylabel(var_y, fontsize=12)
        plt.legend(title=f'{var_color}_binned', loc='best')
        
        # Mostrar gráfico
        plt.show()

# Ejemplo de uso
pares_variables = [('SPY_pre_eecc', 'YIELD GOVT 10 PRE BALANCE'), 
                   (empresa+"_pre_eecc", 'SPY_pre_eecc'), 
                   (empresa+"_pre_eecc", 'YIELD GOVT 10 PRE BALANCE'),]  

# Llamada a la función con el dataframe, la empresa y los pares de variables
graficar_con_variable_fija(datos_train, empresa, pares_variables)


# Prueba de rango con signo de Wilcoxon
variables_eecc_imp = pd.concat([exogenas_train_contables_imputadas,exogenas_test_contables_imputadas], axis = 0 )
from scipy.stats import wilcoxon
wilcoxon_results = {}
for column in variables_eecc_imp.columns:
    stat, p_value = wilcoxon(variables_eecc_imp[column], alternative='two-sided')
    wilcoxon_results[column] = {'stat': stat, 'p_value': p_value}
wilcoxon = pd.DataFrame(wilcoxon_results).T

#%% [Sección 3: ANALISIS DESCRIPTIVO]
# Gráfico variable respuesta EN LA ETAPA DE TRAINING Y TESTING
fig, ax=plt.subplots(figsize=(7, 3))
datos_train[empresa+'_pre_eecc'].plot(ax=ax, marker='o', linestyle='-', label='train')
datos_test[empresa+'_post_eecc'].plot(ax=ax, marker='o', linestyle='-', label='test')
ax.set_title('Precio de la accion luego de presentar EECC')
ax.legend();

# Histograma para cada columna
for i, col in enumerate(datos_eecc.columns, 1):
    plt.subplot(5, 3, i)  # 5 filas, 3 columnas de gráficos
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
plt.tight_layout()

# Pairplot para ver las relaciones entre variables
sns.pairplot(datos_eecc)

# Matriz de correlacion
matriz_correlacion = datos_eecc.corr()
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación de Variables de los EECC')

# Grafico de dispersion para dos variables particulares
plt.scatter(df['IS_COMP_SALES'], df['SALES_REV_TURN'], color='darkblue')
plt.title('Matriz de Correlación de IS_COMP_SALES y SALES_REV_TURN')

#%% [Sección 4: ANALISIS DE COMPONENTES PRINCIPALES]
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(exogenas_train[variables_contables])
df_test_scaled = scaler.transform(exogenas_test[variables_contables])
pca = PCA() 
componentes_principales = pca.fit_transform(df_train_scaled)
# Obtener la proporción de varianza explicada por cada componente
explained_variance_ratio = pca.explained_variance_ratio_
# Calcular la varianza acumulada
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Graficar la varianza explicada y la varianza acumulada
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Varianza explicada individual')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Varianza acumulada')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Proporción de Varianza Explicada')
plt.legend(loc='best')
plt.title('Análisis de Varianza Explicada por Componentes Principales')

# Determinar el número óptimo de componentes para explicar al menos el 90% de la varianza
threshold = 0.9
num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
print(f"El número óptimo de componentes principales para explicar al menos el {threshold*100}% de la varianza es: {num_components}")

# Aplicar PCA con el número óptimo de componentes principales
pca_opt = PCA(n_components=num_components)
componentes_principales_train_opt = pca_opt.fit_transform(df_train_scaled)
componentes_principales_test_opt = pca_opt.transform(df_test_scaled)

# Crear DataFrames con las componentes principales para train y test
column_names = [f'PC{i+1}' for i in range(num_components)]
train_pca = pd.DataFrame(componentes_principales_train_opt, columns=column_names, index=exogenas_train.index)
test_pca = pd.DataFrame(componentes_principales_test_opt, columns=column_names, index=exogenas_test.index)

# Unir las variables PCA con las variables restantes
exogenas_train = pd.concat([train_pca, exogenas_train[otras_variables]], axis=1)
exogenas_test = pd.concat([test_pca, exogenas_test[otras_variables]], axis=1)

# Mostrar la varianza explicada por cada componente principal
print(f"Varianza explicada por cada componente principal: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada: {np.sum(pca.explained_variance_ratio_)}")

# Graficar los componentes principales
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=exogenas_train)
plt.title('Análisis de Componentes Principales')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC3', y='PC4', data=exogenas_train)
plt.title('Análisis de Componentes Principales')
plt.xlabel('Componente Principal 3')
plt.ylabel('Componente Principal 4')

# Composición de cada componente
loadings = pd.DataFrame(pca.components_, columns=variables_contables, index=[f'PC{i+1}' for i in range(pca.n_components_)]).T

# Definimos una función para generar el Loading Plot
def myplot(score, coeff, labels=None):
    xs = score[:, 0]  # Proyecciones en el PC1
    ys = score[:, 1]  # Proyecciones en el PC2
    n = coeff.shape[0]  # Número de variables originales

    # Escalado de los ejes para mejorar la visualización
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    plt.scatter(xs * scalex, ys * scaley, s=5)
    
    # Graficar flechas para cada variable original
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var"+str(i+1), color='green', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
 
    # Añadir etiquetas de los ejes
    plt.xlabel("PC3")
    plt.ylabel("PC4")
    plt.grid()

# Proyectamos los datos en los dos primeros componentes
pca_scores_train = pca_opt.transform(df_train_scaled)[:, 2:4]  # Proyecciones de los datos en los primeros 2 PCs
pca_loadings = pca_opt.components_[0:2, :]  # Coeficientes (cargas) de las variables originales en PC1 y PC2

# Llamamos a la función para generar el gráfico
myplot(pca_scores_train, np.transpose(pca_loadings), labels=variables_contables)

# Definimos la función para generar el Loading Plot (se mantiene igual)
def myplot(score, coeff, labels=None):
    xs = score[:, 0]  # Proyecciones en el PC3 (ahora correctamente asignado)
    ys = score[:, 1]  # Proyecciones en el PC4 (ahora correctamente asignado)
    n = coeff.shape[0]  # Número de variables originales

    # Escalado de los ejes para mejorar la visualización
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    plt.scatter(xs * scalex, ys * scaley, s=5)
    
    # Graficar flechas para cada variable original
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var"+str(i+1), color='green', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
 
    # Añadir etiquetas de los ejes
    plt.xlabel("PC3")
    plt.ylabel("PC4")
    plt.grid()

# Proyectamos los datos en los componentes principales 3 y 4 (Asegurarse que PCA esté calculado previamente)
pca_scores_train_34 = componentes_principales_train_opt[:, 2:4]  # Usamos los PC3 y PC4 de las puntuaciones
pca_loadings_34 = pca_opt.components_[2:4, :]  # Coeficientes (cargas) de las variables originales en PC3 y PC4

# Llamamos a la función para generar el gráfico para los componentes 3 y 4
myplot(pca_scores_train_34, np.transpose(pca_loadings_34), labels=variables_contables)

# Mostrar el gráfico
plt.grid()
plt.show()


# %% [Sección 5: ESTACIONAREIDAD]
#quiero que la serie sea estacionaria => quiero rechazar H0 => p_value < 0.05
adfuller_result = adfuller(df[empresa+"_pre_eecc"])
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')

#los p-values pequeños (por ejemplo, inferiores a 0.05) rechazan la hipótesis nula y sugieren que es necesario diferenciar.
kpss_result = kpss(df[empresa+"_pre_eecc"])
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

datos_diff_1 = df[empresa+"_pre_eecc"].diff().dropna()
adfuller_result_1 = adfuller(datos_diff_1)
print(f'ADF Statistic: {adfuller_result_1[0]}, p-value: {adfuller_result_1[1]}')

kpss_result_1 = kpss(datos_diff_1)
print(f'KPSS Statistic: {kpss_result_1[0]}, p-value: {kpss_result_1[1]}')

# grafico Variable de interes
plt.figure(figsize=(10, 6))
plt.plot(df.index, df[empresa+"_pre_eecc"], marker='o', linestyle='-')
plt.title('Precio de la acción YPF luego de presentar balance')
plt.xlabel('Fecha')
plt.ylabel('USD')
plt.grid(True)

# grafico Variable de interes
plt.figure(figsize=(10, 6))
plt.plot(datos_diff_1, marker='o', linestyle='-')
plt.title('Diferenciación de orden 1')
plt.xlabel('Fecha')
plt.ylabel('USD')
plt.grid(True)

# Analisis de autocorrelacion
# Grafico de autocorrelación para la serie original y la serie diferenciada
# ==============================================================================
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
plot_acf(df[empresa+"_pre_eecc"], ax=axs[0], lags=30, alpha=0.05)
axs[0].set_title('Autocorrelación serie original')
plot_acf(datos_diff_1, ax=axs[1], lags=30, alpha=0.05)
axs[1].set_title('Autocorrelación serie diferenciada (order=1)')

# Autocorrelación parcial para la serie original y la serie diferenciada
# ==============================================================================
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
plot_pacf(df[empresa+"_pre_eecc"], ax=axs[0], lags=20, alpha=0.05)
axs[0].set_title('Autocorrelación parcial serie original')
plot_pacf(datos_diff_1, ax=axs[1], lags=20, alpha=0.05)
axs[1].set_title('Autocorrelación parcial serie diferenciada (order=1)');
plt.tight_layout()

# Diferenciaciación de orden 1 combinada con diferenciación estacional
# ==============================================================================
datos_diff_1_4 = df[empresa+"_pre_eecc"].diff().diff(4).dropna()
adfuller_result_1_4 = adfuller(datos_diff_1_4)
print(f'ADF Statistic: {adfuller_result_1_4[0]}, p-value: {adfuller_result_1_4[1]}')
kpss_result_1_4 = kpss(datos_diff_1_4)
print(f'KPSS Statistic: {kpss_result_1_4[0]}, p-value: {kpss_result_1_4[1]}')

# %% [Sección 6: Grid search basado en backtesting]
param_p = [0, 1,2,3]
param_q = [0, 1,2,3]
param_d = [0, 1,2,3]
param_P = [0, 1,2,3]
param_Q = [0, 1,2,3]
param_D = [0, 1,2,3]
param_m = [12, 4,3] 
# Lista para almacenar resultados
results_list = []
modelos_no_validos = []
# Generar todas las combinaciones posibles de parámetros
for (p, d, q) in itertools.product(param_p, param_d, param_q):
    for (P, D, Q) in itertools.product(param_P, param_D, param_Q):
        for m in param_m:
            try:
                # Ajustar el modelo SARIMAX
                model = SARIMAX(datos_train[empresa+"_post_eecc"], 
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, m),
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                # Almacenar resultados en la lista
                results_list.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, m),
                    'AIC': results.aic})
            except:
                modelos_no_validos.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, m)})
# Convertir la lista de resultados en un DataFrame
results_df = pd.DataFrame(results_list)
'''
# Modelo SARIMAX con statsmodels.Sarimax
# ==============================================================================
modelo_statsmodels = SARIMAX(endog = datos_train['PX POST BALANCE'], order = (1, 1, 1), seasonal_order = (1, 1, 1, 4))
modelo_res_statsmodel = modelo_statsmodels.fit(disp=0)
modelo_res_statsmodel.summary()
predicciones_statsmodels = modelo_res_statsmodel.get_forecast(steps=len(datos_test['PX POST BALANCE'])).predicted_mean
predicciones_statsmodels.name = 'predicciones_statsmodels'
predicciones_statsmodels.index = datos_test['PX POST BALANCE'].index

# Modelo SARIMAX con skforecast.Sarimax
# ==============================================================================
modelo_skforecast = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
modelo_skforecast.fit(y=datos_train['PX POST BALANCE'])
modelo_skforecast.summary()
predicciones_skforecast = modelo_skforecast.predict(steps=len(datos_test['PX POST BALANCE']))
predicciones_skforecast.name = 'predicciones_skforecast'
predicciones_skforecast.index = datos_test['PX POST BALANCE'].index
# Modelo SARIMAX con pdmarima.Sarimax
# ==============================================================================
modelo_pdmarima = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
modelo_pdmarima.fit(y=datos_train['PX POST BALANCE'])
modelo_pdmarima.summary()
predicciones_pdmarima = modelo_pdmarima.predict(len(datos_test['PX POST BALANCE']))
predicciones_pdmarima.name = 'predicciones_pdmarima'
predicciones_pdmarima.index = datos_test['PX POST BALANCE'].index

#### VARIABLES EXOGENAS ####
# Modelo SARIMAX con statsmodels.Sarimax
# ==============================================================================
modelo_exogenas = SARIMAX(endog = datos_train['PX POST BALANCE'], order = (1, 1, 1), seasonal_order = (1, 1, 1, 4), exog = exogenas_train)
modelo_res_exogenas = modelo_exogenas.fit(disp=0)
modelo_res_exogenas.summary()
predicciones_exogenas = modelo_res_exogenas.get_forecast(steps=len(datos_test['PX POST BALANCE']), exog = exogenas_test).predicted_mean
predicciones_exogenas.name = 'predicciones_statsmodels'
predicciones_exogenas.index = datos_test['PX POST BALANCE'].index

# Plot predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
datos_train['PX POST BALANCE'].plot(ax=ax, label='train')
datos_test['PX POST BALANCE'].plot(ax=ax, label='test')
predicciones_statsmodels.plot(ax=ax, label='statsmodels')
predicciones_skforecast.columns = ['skforecast']
predicciones_skforecast.plot(ax=ax, label='skforecast')
predicciones_pdmarima.plot(ax=ax, label='pmdarima')
ax.set_title('Predicciones con modelos ARIMA')
ax.legend();

#### BACKTESTING ####
# Grid search basado en backtesting
# ==============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(
                                order=(1, 1, 1), # Placeholder replaced in the grid search
                                maxiter=500) )
param_grid = {
    'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 4), (1, 1, 1, 4)],
    'trend': [None, 'n', 'c']}
resultados_grid = grid_search_sarimax(
                        forecaster            = forecaster,
                        y                     = datos_train['PX POST BALANCE'],
                        param_grid            = param_grid,
                        steps                 = 4,
                        refit                 = True,
                        metric                = 'mean_absolute_error',
                        initial_train_size    = 1,#len(datos_train),
                        fixed_train_size      = False,
                        return_best           = False,
                        n_jobs                = 'auto',
                        suppress_warnings_fit = True,
                        verbose               = False,
                        show_progress         = True)
resultados_grid.head(5)

# Auto arima: seleccion basada en AIC
# ==============================================================================
modelo = auto_arima(
            y                 = datos_train['PX POST BALANCE'],
            start_p           = 0,
            start_q           = 0,
            max_p             = 3,
            max_q             = 3,
            seasonal          = True,
            test              = 'adf',
            m                 = 4, # periodicidad de la estacionalidad
            d                 = None, # El algoritmo determina 'd'
            D                 = None, # El algoritmo determina 'D'
            trace             = True,
            error_action      = 'ignore',
            suppress_warnings = True,
            stepwise          = True)
'''
# %% [Sección 7: EL MODELO SARIMA]
# Los mejores modelos: (1,2, 1) y  (1, 2, 0, 4)
# Definir los parámetros del modelo SARIMA
order = (3,1,1)
seasonal_order = (3, 1, 2, 4)
model_sarima = sm.tsa.SARIMAX(datos_train[empresa+'_post_eecc'], 
                              order=order, 
                              seasonal_order=seasonal_order)

model_sarima_fit = model_sarima.fit(disp=False)
model_sarima_fit.summary()
# Hacer predicciones
predictions_sarima = model_sarima_fit.forecast(steps=len(datos_test[empresa+'_post_eecc']))
predictions_sarima.index = datos_test[empresa+'_post_eecc'].index
predictions_sarima = pd.DataFrame(predictions_sarima)
mse = mean_squared_error(endogena_test, predictions_sarima)
print(mse)
# %% [Sección 8: EL MODELO SARIMAX]
# Crear y entrenar el modelo SARIMAX
#modelo 2. SARIMAX con tratamiento de outliers
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 2), seasonal_order=(2, 0, 1, 4)) # rsme 1.45
model = SARIMAX(endogena_train, exog=exogenas_train, order=(3, 1, 1), seasonal_order=(3, 1, 2, 4)) # rsme 1.3
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 1, 1, 4)) # rsme 1.28
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 1), seasonal_order=(2, 0, 1, 4)) # rsme 1.1
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 3, 4)) # rsme 1
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 0.86
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 2), seasonal_order=(2, 0, 1, 4)) # rsme 0.85
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 2, 1), seasonal_order=(2, 0, 2, 4)) # rsme 0.84
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 1, 4)) # rsme 0.79

#modelo 3. SARIMAX con outliers y CP
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 2, 2, 4)) # rsme 5
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 1, 2, 4)) # rsme 1.64
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 2, 1), seasonal_order=(2, 0, 2, 4)) # rsme 1.54
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 2), seasonal_order=(2, 0, 2, 4)) # rsme 0.746
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 0.79
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(1, 0, 2, 4)) # rsme 0.77
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 0.76

#modelo 4. SARIMAX con outliers , CP y YIELD_i_SPY
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 2, 1), seasonal_order=(2, 0, 2, 4)) # rsme 12.2
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 2), seasonal_order=(2, 0, 2, 4)) # rsme 4.87
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 4
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 3.7
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 1, 2, 4)) # rsme 1.25
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 1, 4)) # rsme 1.25

#modelo 4. SARIMAX con outliers , CP y GGAL_i_SPY
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 2, 1), seasonal_order=(2, 0, 2, 4)) # rsme 297
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 1, 4))# rsme 64
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 2), seasonal_order=(2, 0, 1, 4)) # rsme 65
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 63
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 3, 4))# rsme 63
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 1), seasonal_order=(2, 0, 1, 4))# rsme 58
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 1, 1, 4))# rsme 101
model = SARIMAX(endogena_train, exog=exogenas_train, order=(3, 1, 1), seasonal_order=(3, 1, 2, 4))  # rsme 102
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 2), seasonal_order=(2, 0, 1, 4)) #60

#modelo 4. SARIMAX con outliers , CP y GGAL_i_SPY
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 2, 1), seasonal_order=(2, 0, 2, 4)) # rsme 2.93
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 1, 4))# rsme 2.18
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 2), seasonal_order=(2, 0, 1, 4)) # rsme 2.18
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 2, 4)) # rsme 2.23
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 1), seasonal_order=(2, 0, 1, 4))# rsme 2.48
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 1, 1, 4))# rsme 4.5
model = SARIMAX(endogena_train, exog=exogenas_train, order=(3, 1, 1), seasonal_order=(3, 1, 2, 4))  # rsme 3.77
model = SARIMAX(endogena_train, exog=exogenas_train, order=(2, 1, 2), seasonal_order=(2, 0, 1, 4)) # 2.8
model = SARIMAX(endogena_train, exog=exogenas_train, order=(1, 1, 1), seasonal_order=(2, 0, 3, 4))# rsme 2


#AJUSTE DEL MODELO 
model_fit = model.fit(disp=False)
model_fit.summary()
# Hacer predicciones
predictions = model_fit.forecast(steps=len(exogenas_test), exog=exogenas_test)
predictions.index = datos_test[empresa+'_post_eecc'].index
predictions = pd.DataFrame(predictions)
# Error del modelo del modelo
mse = mean_squared_error(endogena_test, predictions)
print(f"Mean Squared Error: {mse}")


# ANALISIS DE RESIDUOS
residuos = model_fit.resid
# Gráfico de residuos a lo largo del tiempo
plt.plot(residuos, marker='o', color="darkblue", alpha=0.7, markerfacecolor='darkblue', markeredgecolor='darkblue', markeredgewidth=1, linestyle='-', linewidth=0.5)  # Línea transparente
plt.title('Residuos del modelo')

plt.hist(residuos, bins=15, edgecolor='black')
plt.title('Histograma de los residuos')

stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Gráfico Q-Q de los residuos')

# Gráfico de autocorrelación de los residuos
plot_acf(residuos)
plot_pacf(residuos)

# Training vs estimaciones
plt.figure(figsize=(12, 9))
plt.plot(endogena_test.index, endogena_test,  marker='o', label='Datos reales')
plt.plot(predictions.index, predictions,  marker='o', label='Predicciones', color='red')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.title('Predicciones de SARIMAX')
plt.legend()

# Pruebas de hipotesis
# Un p-valor alto sugiere que no hay autocorrelación significativa en los residuos.
ljung_box_results = acorr_ljungbox(residuos, lags=[1, 5, 10], return_df=True)
print(ljung_box_results)

#Prueba de normalidad
# Un p-valor alto sugiere que los residuos siguen una distribución normal.
jb_test_stat, jb_p_value = jarque_bera(residuos)
print(f"Jarque-Bera test statistic: {jb_test_stat}")
print(f"Jarque-Bera p-value: {jb_p_value}")

# Prueba de normalidad
stat, p = shapiro(residuos)
print(f'Estadístico Shapiro-Wilk: {stat}, p-value: {p}')
# Si el p-valor es menor que 0.05, rechazo la hipótesis de que los residuos siguen una distribución normal.

# RESULTADOS
#con info de yahoo finance
fechas_post_eecc = fechas_post_eecc.tail(len(endogena_test))
start_date = fechas_post_eecc.min()
end_date = fechas_post_eecc.max() + BDay(2)
datos_empresa = yf.download(empresa, start= start_date, end=end_date)[['Open', 'High', 'Low', 'Adj Close']]
df_velas = datos_empresa.loc[fechas_post_eecc]
df_velas = df_velas.rename(columns={'Adj Close': 'Close'})
df_velas = df_velas.sort_values(by='Date')
add_plot = mpf.make_addplot(predictions['predicted_mean'], scatter=True, markersize=30, color='blue')
mpf.plot(df_velas, type='candle', style='charles',addplot=add_plot, title='Gráfico de Velas de GGAL con la prediccion realizada', ylabel='Precio')

'''
# con info de bloomberg
df_velas = pd.read_excel(archivo, sheet_name= "GGAL-VELAS")
df_velas['fecha'] = pd.to_datetime(df_velas['fecha'], format='%Y-%m-%d')
df_velas = df_velas.sort_values(by='fecha')
df_velas = df_velas.set_index('fecha')
df_velas = df_velas.tail(len(datos_test))
add_plot = mpf.make_addplot(predictions['predicted_mean'], scatter=True, markersize=30, color='blue')
mpf.plot(df_velas, type='candle', style='charles',addplot=add_plot, title='Gráfico de Velas con la prediccion', ylabel='Precio')
'''

# %% [Sección 9: IMPORTANCIA DE LAS VARIABLES]
# Con el mejor modelo SARIMAX: : (1, 1, 1) x (2, 0, 1, 4)

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def forward_selection_sarimax(df, target_variable, variables_contables, p, d, q, P, D, Q, s):
    """
    Selección de variables forward para SARIMAX, considerando solo las variables contables.
    
    :param df: DataFrame con las variables.
    :param target_variable: Nombre de la variable target (str).
    :param variables_contables: Lista de nombres de las variables contables a analizar (list).
    :param p, d, q: Parámetros no estacionales del modelo SARIMAX.
    :param P, D, Q, s: Parámetros estacionales del modelo SARIMAX.
    :return: Lista de variables seleccionadas y el modelo final ajustado.
    """
    target = df[target_variable]
    
    # Inicialización de listas
    variables_seleccionadas = []
    mejor_aic = float('inf')
    mejor_modelo = None
    
    # Iteración sobre las variables contables
    for _ in range(len(variables_contables)):
        aic_temp = []
        modelos_temp = []
        
        # Evaluar cada variable no seleccionada
        for variable in variables_contables:
            if variable not in variables_seleccionadas:
                # Crear conjunto de variables + nueva variable
                variables_actuales = variables_seleccionadas + [variable]
                X = df[variables_actuales]
                
                # Ajustar el modelo SARIMAX
                modelo = SARIMAX(target, exog=X, order=(p, d, q), seasonal_order=(P, D, Q, s))
                modelo_ajustado = modelo.fit(disp=False)
                
                # Guardar el AIC y el modelo
                aic_temp.append((modelo_ajustado.aic, variable))
                modelos_temp.append(modelo_ajustado)
        
        # Seleccionar la mejor variable con el menor AIC
        mejor_aic_variable, mejor_variable = min(aic_temp)
        
        # Verificar si mejora el AIC
        if mejor_aic_variable < mejor_aic:
            mejor_aic = mejor_aic_variable
            variables_seleccionadas.append(mejor_variable)
            mejor_modelo = modelos_temp[aic_temp.index((mejor_aic_variable, mejor_variable))]
        else:
            break  # Si no mejora, se detiene la selección

    print(f"Variables seleccionadas: {variables_seleccionadas}")
    print(f"Mejor AIC: {mejor_aic}")
    
    return variables_seleccionadas, mejor_modelo

# Parámetros del modelo SARIMAX
p, d, q = 1, 1, 1
P, D, Q, s = 2, 0, 1, 4  # Parámetros estacionales

# Llamar a la función con el DataFrame 'df', la variable target 'YPF_post_eecc' y las variables contables
variables_seleccionadas, mejor_modelo = forward_selection_sarimax(df, empresa+'_post_eecc', variables_contables, p, d, q, P, D, Q, s)

# Mostrar resumen del mejor modelo
print(mejor_modelo.summary())


import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

def backward_selection_sarimax(df, target_variable, variables_contables, p, d, q, P, D, Q, s):
    """
    Selección de variables backward para SARIMAX, considerando solo las variables contables.

    :param df: DataFrame con las variables.
    :param target_variable: Nombre de la variable target (str).
    :param variables_contables: Lista de nombres de las variables contables a analizar (list).
    :param p, d, q: Parámetros no estacionales del modelo SARIMAX.
    :param P, D, Q, s: Parámetros estacionales del modelo SARIMAX.
    :return: Lista de variables seleccionadas y el modelo final ajustado.
    """
    target = df[target_variable]
    
    # Inicialización de variables seleccionadas: comenzar con todas
    variables_seleccionadas = variables_contables.copy()
    mejor_aic = float('inf')
    mejor_modelo = None

    # Bucle hasta que quede una sola variable o no se mejore el AIC
    while len(variables_seleccionadas) > 0:
        aic_temp = []
        modelos_temp = []
        
        # Ajustar el modelo SARIMAX con las variables seleccionadas actuales
        X = df[variables_seleccionadas]
        modelo = SARIMAX(target, exog=X, order=(p, d, q), seasonal_order=(P, D, Q, s))
        modelo_ajustado = modelo.fit(disp=False)
        
        # Guardar el AIC del modelo actual
        aic_temp.append((modelo_ajustado.aic, None))  # Ninguna eliminación
        
        # Intentar eliminar una variable en cada iteración
        for variable in variables_seleccionadas:
            temp_variables = variables_seleccionadas.copy()
            temp_variables.remove(variable)
            X_temp = df[temp_variables]
            modelo_temp = SARIMAX(target, exog=X_temp, order=(p, d, q), seasonal_order=(P, D, Q, s))
            modelo_ajustado_temp = modelo_temp.fit(disp=False)
            
            # Guardar el AIC y el modelo temporal
            aic_temp.append((modelo_ajustado_temp.aic, variable))
            modelos_temp.append(modelo_ajustado_temp)
        
        # Seleccionar el AIC más bajo
        mejor_aic_temp, peor_variable = min(aic_temp)
        
        if mejor_aic_temp < mejor_aic:
            mejor_aic = mejor_aic_temp
            mejor_modelo = modelo_ajustado if peor_variable is None else modelos_temp[aic_temp.index((mejor_aic_temp, peor_variable))-1]
            
            # Eliminar la peor variable si corresponde
            if peor_variable is not None:
                variables_seleccionadas.remove(peor_variable)
        else:
            break  # Si no mejora, detener el proceso

    print(f"Variables seleccionadas: {variables_seleccionadas}")
    print(f"Mejor AIC: {mejor_aic}")
    
    return variables_seleccionadas, mejor_modelo

# Llamar a la función con el DataFrame 'df', la variable target 'YPF_post_eecc' y las variables contables
variables_seleccionadas, mejor_modelo = backward_selection_sarimax(df, empresa+'_post_eecc', variables_contables, 1, 1, 1, 2, 0, 1, 4)

# Mostrar resumen del mejor modelo
print(mejor_modelo.summary())


# %% [Sección 10: SHAP]
df = pd.read_excel(archivo, sheet_name= "BD GGAL")
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
df = df.set_index('fecha')
df = df.sort_values(by='fecha')
fechas_post_eecc = df['fecha_post_eecc']
# Elimino la fecha de las paso de MACRI para probar. Fecha: 12-8-19
#fecha_a_eliminar = '2019-08-12'
#df = df.drop(fecha_a_eliminar)

variables_contables = ["PX_TO_EBITDA","EBITDA_TO_REVENUE", "CF_NET_INC", "IS_COMP_NET_INCOME", "SALES_REV_TURN","IS_OPER_INC" ,"EARN_FOR_COM_TO_TOT_REV"]
otras_variables = ["YIELD GOVT 10 PRE BALANCE", "SPY_pre_eecc" , empresa+"_pre_eecc"]
datos_eecc = df[variables_contables]

# USO DATOS DE YFINANCE Y NO DE BLOOMBERG
start_date = df['fecha_pre_eecc'].min() +  BDay(-1)
end_date = df['fecha_post_eecc'].max() + BDay(1)
datos_spy =     yf.download('SPY',   start= start_date, end=end_date)['Adj Close']
datos_empresa = yf.download(empresa, start= start_date, end=end_date)['Adj Close']

df['SPY_pre_eecc'] = datos_spy.reindex(df['fecha_pre_eecc']).values
df[empresa + '_post_eecc'] = datos_empresa.reindex(df['fecha_post_eecc']).values
df[empresa + '_pre_eecc'] = datos_empresa.reindex(df['fecha_pre_eecc']).values

#ELIMINO MAS COLUMNAS QUE NO SON EXPLICATIVAS
df = df.drop(columns=["hora","fecha_pre_eecc","fecha_post_eecc"])


# construyo variable de interes. 1: subio. 0: bajó
df["variacion"] = df[empresa+"_post_eecc"] - df[empresa+"_pre_eecc"]
df['variacion'] = df['variacion'].apply(lambda x: 1 if x >= 0 else 0)
#ELIMINO MAS COLUMNAS QUE NO DE LOS EECC
df = df.drop(columns=["YIELD GOVT 10 PRE BALANCE","SPY_pre_eecc",empresa+"_pre_eecc", empresa+"_post_eecc"])

X_train, X_test, y_train, y_test = train_test_split(df.drop('variacion', axis=1),
                                                    df['variacion'].values.reshape(-1,1),
                                                    test_size=0.2,
                                                    random_state=42)

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train.reshape(len(y_train),))

y_pred = model.predict(X_test_scaled)
y_ajuste_train = model.predict(X_train_scaled)

print('Métricas en el conjunto de prueba')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

df_coeficientes=pd.DataFrame(model.coef_.T, index=variables_contables)

# Crea un objeto explainer SHAP
explainer = shap.LinearExplainer(model, X_train_scaled, feature_names=variables_contables)

# Calcula los valores SHAP para un conjunto de ejemplos de prueba
shap_values = explainer.shap_values(X_test_scaled)

explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, feature_names=variables_contables, data=X_test)
shap.plots.bar(explanation)

shap.plots.beeswarm(explanation)