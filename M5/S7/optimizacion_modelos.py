# Métricas de Desempeño de Modelos de Regresión y Clasificación
# Objetivo: Evaluar el rendimiento de modelos de regresión y clasificación usando métricas apropiadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Carga de datos
reg = pd.read_csv('datos_regresion.csv')
clas = pd.read_csv('datos_clasificacion.csv')

# 2. Evaluación de un modelo de regresión
# Preprocesamiento: codificar categoría
reg['Categoria'] = LabelEncoder().fit_transform(reg['Categoria'])
X_reg = reg.drop('Valor_Ventas', axis=1)
y_reg = reg['Valor_Ventas']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Modelo de regresión lineal
reg_model = LinearRegression()
reg_model.fit(Xr_train, yr_train)
reg_pred = reg_model.predict(Xr_test)

# Métricas de regresión
mae = mean_absolute_error(yr_test, reg_pred)
mse = mean_squared_error(yr_test, reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, reg_pred)

print('--- Métricas de Regresión ---')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

# Interpretación
print('\nInterpretación:')
print(f'El MAE indica que el error promedio en las predicciones es de {mae:.2f} unidades de ventas.')
print(f'El RMSE penaliza más los errores grandes y es de {rmse:.2f}.')
print(f'Un R² de {r2:.2f} indica que el modelo explica el {r2*100:.1f}% de la varianza de las ventas.')

# 3. Evaluación de un modelo de clasificación
X_clas = clas.drop('Comprara', axis=1)
y_clas = clas['Comprara']
scaler = StandardScaler()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clas, y_clas, test_size=0.2, random_state=42)
Xc_train = scaler.fit_transform(Xc_train)
Xc_test = scaler.transform(Xc_test)

# Modelo de clasificación
clas_model = LogisticRegression()
clas_model.fit(Xc_train, yc_train)
yc_pred = clas_model.predict(Xc_test)
yc_proba = clas_model.predict_proba(Xc_test)[:, 1]

# Métricas de clasificación
cm = confusion_matrix(yc_test, yc_pred)
precision = precision_score(yc_test, yc_pred)
accuracy = accuracy_score(yc_test, yc_pred)
recall = recall_score(yc_test, yc_pred)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
fpr, tpr, thresholds = roc_curve(yc_test, yc_proba)
auc_score = auc(fpr, tpr)

print('\n--- Métricas de Clasificación ---')
print('Matriz de confusión:')
print(cm)
print(f'Precisión: {precision:.2f}')
print(f'Exactitud: {accuracy:.2f}')
print(f'Sensibilidad (Recall): {recall:.2f}')
print(f'Especificidad: {specificity:.2f}')
print(f'AUC: {auc_score:.2f}')
print(f'F1 Score: {f1_score(yc_test, yc_pred):.2f}')

# Curva ROC
graf = plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.tight_layout()
plt.show()

# Interpretación
print('\nInterpretación:')
print('La matriz de confusión muestra la cantidad de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.')
print(f'La precisión indica que el {precision*100:.1f}% de las predicciones positivas fueron correctas.')
print(f'La exactitud muestra que el modelo acertó en el {accuracy*100:.1f}% de los casos.')
print(f'La sensibilidad indica que el modelo detecta el {recall*100:.1f}% de los casos positivos.')
print(f'La especificidad muestra que el modelo identifica correctamente el {specificity*100:.1f}% de los negativos.')
print(f'El AUC mide la capacidad del modelo para distinguir entre clases; cuanto más cercano a 1, mejor.')

# 4. Análisis de resultados
print('\n--- Análisis de Resultados ---')
print('Las métricas de regresión (MAE, MSE, RMSE, R²) evalúan la calidad de las predicciones numéricas, mientras que las métricas de clasificación (Precisión, Exactitud, Sensibilidad, Especificidad, AUC) evalúan la capacidad del modelo para distinguir entre clases.')
print('MAE es útil cuando se desea interpretar el error promedio en las mismas unidades que la variable objetivo, mientras que RMSE penaliza más los errores grandes.')
print('Para clasificación, la sensibilidad es clave cuando es más importante detectar positivos (por ejemplo, enfermedades), mientras que la especificidad es prioritaria cuando es más importante evitar falsos positivos (por ejemplo, fraudes).')

# =====================
# Optimización del Modelo
# =====================
print('\n--- Optimización del Modelo ---')

# --- Feature Engineering ---
# Crear nuevas características para regresión
reg['Precio_x_Antiguedad'] = reg['Precio'] * reg['Antiguedad']
reg['Precio_log'] = np.log1p(reg['Precio'])
X_reg_fe = reg.drop('Valor_Ventas', axis=1)
y_reg_fe = reg['Valor_Ventas']
Xr_train_fe, Xr_test_fe, yr_train_fe, yr_test_fe = train_test_split(X_reg_fe, y_reg_fe, test_size=0.2, random_state=42)

# Selección de características
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=3)
Xr_train_sel = selector.fit_transform(Xr_train_fe, yr_train_fe)
Xr_test_sel = selector.transform(Xr_test_fe)

# --- Ajuste de Hiperparámetros ---
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {
    'fit_intercept': [True, False], # Si se ajusta la constante/intercepto.
    'copy_X': [True, False], # Si se copia la matriz de entrada.
    'positive': [True, False] # Si se imponen restricciones de positividad.
}
grid = GridSearchCV(LinearRegression(), param_grid, cv=3)
grid.fit(Xr_train_sel, yr_train_fe)
print(f"Mejor combinación GridSearchCV: {grid.best_params_}")

rand = RandomizedSearchCV(LinearRegression(), param_grid, n_iter=2, cv=3, random_state=42)
rand.fit(Xr_train_sel, yr_train_fe)
print(f"Mejor combinación RandomizedSearchCV: {rand.best_params_}")

# --- Regularización ---
'''
alpha: (float) Fuerza de la regularización (más alto = más regularización).
fit_intercept: (bool) Si se ajusta el intercepto.
max_iter: (int) Número máximo de iteraciones.
tol: (float) Tolerancia para la convergencia.
positive: (bool, solo Ridge/Lasso) Restringe los coeficientes a ser positivos.
selection: ('cyclic', 'random', solo Lasso/ElasticNet) Estrategia de actualización de coeficientes.
l1_ratio: (float, solo ElasticNet) Mezcla entre L1 y L2 (0 = Ridge, 1 = Lasso).
'''

from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet

lasso = Lasso(alpha=0.1)
lasso.fit(Xr_train_sel, yr_train_fe)

ridge = Ridge(alpha=1.0)
ridge.fit(Xr_train_sel, yr_train_fe)

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(Xr_train_sel, yr_train_fe)

print(f"Coeficientes Lasso: {lasso.coef_}")
print(f"Coeficientes Ridge: {ridge.coef_}")
print(f"Coeficientes ElasticNet: {elastic.coef_}")

# --- Balanceo de Datos (Clasificación) ---
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score

# Sobremuestreo
ros = RandomOverSampler(random_state=42)
Xc_train_ros, yc_train_ros = ros.fit_resample(Xc_train, yc_train)
clas_model_ros = LogisticRegression()
clas_model_ros.fit(Xc_train_ros, yc_train_ros)
yc_pred_ros = clas_model_ros.predict(Xc_test)
print(f"F1-score con sobremuestreo: {f1_score(yc_test, yc_pred_ros):.2f}")
print(f"Matriz de confusión (ROS):\n{confusion_matrix(yc_test, yc_pred_ros)}")

# Submuestreo
rus = RandomUnderSampler(random_state=42)
Xc_train_rus, yc_train_rus = rus.fit_resample(Xc_train, yc_train)
clas_model_rus = LogisticRegression()
clas_model_rus.fit(Xc_train_rus, yc_train_rus)
yc_pred_rus = clas_model_rus.predict(Xc_test)
print(f"F1-score con submuestreo: {f1_score(yc_test, yc_pred_rus):.2f}")
print(f"Matriz de confusión (RUS):\n{confusion_matrix(yc_test, yc_pred_rus)}")

# curva ROC para modelos balanceados
fpr_ros, tpr_ros, _ = roc_curve(yc_test, clas_model_ros.predict_proba(Xc_test)[:, 1])
fpr_rus, tpr_rus, _ = roc_curve(yc_test, clas_model_rus.predict_proba(Xc_test)[:, 1])
auc_ros = auc(fpr_ros, tpr_ros)
auc_rus = auc(fpr_rus, tpr_rus)

# Gráfico de curvas ROC para modelos balanceados
plt.figure(figsize=(8, 6))
plt.plot(fpr_ros, tpr_ros, label=f'ROC Sobremuestreo (AUC={auc_ros:.2f})')
plt.plot(fpr_rus, tpr_rus, label=f'ROC Submuestreo (AUC={auc_rus:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para Modelos Balanceados')
plt.legend()
plt.show()

# --- Análisis de Resultados de Optimización ---
print('\n--- Análisis de Resultados de Optimización ---')
print('a. ¿Qué técnica de optimización tuvo mayor impacto en la mejora del modelo?')
print('b. ¿Cómo afectó la elección de hiperparámetros al rendimiento?')
print('c. ¿Cómo influyó la regularización en la generalización del modelo?')

'''
Análisis de Resultados de Optimización

a. ¿Qué técnica de optimización tuvo mayor impacto en la mejora del modelo?
El mayor impacto se observa en el balanceo de datos (sobremuestreo y submuestreo). El F1-score pasó de 0.17 (antes de optimizar, calculado a partir de la matriz de confusión original) a 0.60–0.61 tras aplicar técnicas de balanceo, y la cantidad de verdaderos positivos aumentó notablemente. Esto indica que el modelo ahora identifica mucho mejor la clase minoritaria, aunque la matriz de confusión muestra un mayor número de falsos positivos.

b. ¿Cómo afectó la elección de hiperparámetros al rendimiento?
El ajuste de hiperparámetros con GridSearchCV y RandomizedSearchCV seleccionó la mejor configuración (fit_intercept=True), pero el impacto en el rendimiento fue menor comparado con el balanceo. Sin embargo, elegir los hiperparámetros adecuados ayuda a que el modelo sea más estable y evite errores sistemáticos.

c. ¿Cómo influyó la regularización en la generalización del modelo?
La regularización (Lasso y Ridge) ajustó los coeficientes del modelo, reduciendo algunos a cero (Lasso) o disminuyéndolos (Ridge), lo que ayuda a evitar el sobreajuste y mejora la capacidad de generalización. Esto es útil especialmente cuando hay muchas variables o multicolinealidad, aunque en este caso el impacto fue más visible en la interpretación de los coeficientes que en las métricas de desempeño.

Conclusión:
El balanceo de datos fue la técnica que más mejoró la capacidad del modelo para identificar correctamente ambas clases. El ajuste de hiperparámetros y la regularización contribuyeron a la estabilidad y generalización, pero su efecto fue más sutil en comparación con el impacto del balanceo sobre el F1-score y la matriz de confusión.
'''