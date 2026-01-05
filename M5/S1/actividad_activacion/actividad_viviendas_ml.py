"""
FUNDAMENTOS DE APRENDIZAJE DE MÁQUINA

Empresa: XYJ Analytics
Objetivo: Desarrollar un modelo de Machine Learning para predecir el precio de viviendas a partir de datos inmobiliarios.
"""

# 1. Carga y Exploración Inicial del Dataset
import pandas as pd

# Cargar el archivo viviendas.csv
viviendas = pd.read_csv('M5/S1/actividad_activacion/viviendas.csv')

# Mostrar las primeras 5 filas
print("Primeras 5 filas del DataFrame:")
print(viviendas.head())

# Describir la estructura del dataset
print("\nInformación del DataFrame:")
viviendas.info()
print("\nEstadísticas descriptivas:")
print(viviendas.describe())

# 2. Limpieza de Datos
# Detectar valores faltantes
print("\nValores faltantes por columna:")
print(viviendas.isnull().sum())

# Imputar o eliminar valores faltantes (ejemplo: eliminar filas con NA)
viviendas = viviendas.dropna()

# Verificar tipos de datos
print("\nTipos de datos tras limpieza:")
viviendas.dtypes

# 3. Análisis Exploratorio de Datos
import matplotlib.pyplot as plt
import seaborn as sns

# Superficie vs Precio
plt.figure(figsize=(6,4))
sns.scatterplot(x='superficie', y='precio', data=viviendas)
plt.title('Superficie vs Precio')
plt.show()

# Habitaciones vs Precio
plt.figure(figsize=(6,4))
sns.scatterplot(x='habitaciones', y='precio', data=viviendas)
plt.title('Habitaciones vs Precio')
plt.show()

# Superficie vs Precio con línea de regresión
plt.figure(figsize=(6,4))
sns.regplot(x='superficie', y='precio', data=viviendas, scatter_kws={'s':40}, line_kws={'color':'red'})
plt.title('Superficie vs Precio (con línea de regresión)')
plt.show()

# Habitaciones vs Precio con línea de regresión
plt.figure(figsize=(6,4))
sns.regplot(x='habitaciones', y='precio', data=viviendas, scatter_kws={'s':40}, line_kws={'color':'red'})
plt.title('Habitaciones vs Precio (con línea de regresión)')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot: Precio según número de habitaciones
plt.figure(figsize=(7,4))
sns.boxplot(x='habitaciones', y='precio', data=viviendas)
plt.title('Distribución del precio según número de habitaciones')
plt.xlabel('Número de habitaciones')
plt.ylabel('Precio')
plt.show()

# Opcional: Violinplot para ver la densidad
plt.figure(figsize=(7,4))
sns.violinplot(x='habitaciones', y='precio', data=viviendas)
plt.title('Densidad del precio según número de habitaciones')
plt.xlabel('Número de habitaciones')
plt.ylabel('Precio')
plt.show()

# Detectar outliers visualmente
plt.figure(figsize=(6,4))
sns.boxplot(x='precio', data=viviendas)
plt.title('Boxplot de Precio')
plt.show()

# 4. Codificación de variables categóricas
# One-Hot Encoding para Barrio
viviendas = pd.get_dummies(viviendas, columns=['barrio'], drop_first=True)

# encoder = OneHotEncoder(drop='first', sparse=False)
# barrio_encoded = encoder.fit_transform(viviendas[['barrio']])

# 5. División del Dataset
from sklearn.model_selection import train_test_split

X = viviendas.drop('precio', axis=1)
y = viviendas['precio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenamiento de un Modelo de Regresión Lineal
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Mostrar coeficientes
print("\nCoeficientes del modelo:")
for nombre, coef in zip(X_train.columns, modelo.coef_):
    print(f"{nombre}: {coef:.2f}")
print(f"Intercepto: {modelo.intercept_:.2f}")

# 7. Evaluación del Modelo
from sklearn.metrics import mean_squared_error, r2_score

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R² (coeficiente de determinación): {r2:.2f}")

# Comentario sobre el desempeño
if r2 > 0.7:
    print("El modelo tiene un buen desempeño predictivo.")
elif r2 > 0.4:
    print("El modelo tiene un desempeño aceptable, pero puede mejorarse.")
else:
    print("El modelo no predice bien, se recomienda ajustar variables o probar otros modelos.")
