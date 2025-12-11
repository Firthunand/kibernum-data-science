# ===============================
# 1. Introducción al Análisis de Datos
# ===============================
# Introducción al Análisis de Datos
# En el análisis de datos, es fundamental identificar correctamente el tipo de variable con el que trabajamos, ya que esto determina cómo podemos analizar, visualizar y manipular la información.
# Es esencial aprender a clasificar variables, calcular medidas de tendencia central y dispersión, y visualizar datos mediante histogramas y diagramas de caja.

# ===============================
# 2. Tipos de Variables
# ===============================
# Tipos de Variables
# 
# Variables Categóricas
# Representan atributos o características sin valor numérico asociado.
# No tienen un orden cuantificable.
# Ejemplo:
#   Género: Masculino, Femenino, Otro.
#   Color favorito: Rojo, Azul, Verde.
#   Tipo de transporte: Auto, Bicicleta, Autobús.
#
# Variables Cuantitativas
# Representan cantidades numéricas que pueden ser discretas o continuas.
# Ejemplo:
#   Edad en años: 15, 22, 30.
#   Altura en cm: 175, 182, 160.
#   Ingresos mensuales en dólares: 1500, 2800, 3200.

# ===============================
# 3. Subtipos de Variables y Ejemplos en Python
# ===============================
# ===============================
# Subtipos de Variables y Ejemplos en Python
# ===============================

# Variables Categóricas
# - Nominales: Sin orden específico (ejemplo: colores).
# - Ordinales: Con orden implícito, pero diferencias no medibles (ejemplo: nivel de satisfacción: bajo, medio, alto).

print("### Ejemplo en Python para Variables Categóricas (Frecuencias):")
import pandas as pd
df = pd.DataFrame({"Género": ["Masculino", "Femenino", "Masculino", "Otro"]})
print(df["Género"].value_counts())

# 
# Variables Cuantitativas
# - Discretas: Valores enteros contables, sin decimales (ejemplo: número de hijos: 0, 1, 2).
# - Continuas: Cualquier valor dentro de un rango, incluyendo decimales (ejemplo: temperatura: 36.5°C, 22.3°C).
#

print("### Ejemplo en Python para Variables Cuantitativas:")
import numpy as np
# Variable Discreta
discretos = np.array([0, 1, 2, 2, 3])
print(f"Promedio de hijos: {np.mean(discretos)}")
# Variable Continua
pesos = np.array([65.2, 70.1, 68.5, 72.3])
print(f"Peso máximo: {np.max(pesos)}\n")

# ===============================
# 4. Tablas de Frecuencia
# ===============================
# Tablas de Frecuencia
# Una tabla de frecuencia es una herramienta estadística utilizada para organizar y resumir datos,
# mostrando cuántas veces aparece cada valor dentro de una variable.
# Ayuda a visualizar la distribución de los datos y a identificar patrones de frecuencia.
# Tipos:
#   1. Frecuencia absoluta: Número de veces que aparece cada categoría.
#   2. Frecuencia relativa: Proporción de cada categoría en relación con el total.
#   3. Frecuencia acumulada: Suma progresiva de las frecuencias.

print("### Ejemplo en Python de Tabla de Frecuencia:")
datos = [6, 7, 7, 8, 6, 9, 8, 7, 6, 7, 8, 9, 8, 7, 9]
df = pd.DataFrame(datos, columns=["Calificación"])
tabla_frecuencia = df["Calificación"].value_counts().sort_index().to_frame("Frecuencia Absoluta")
tabla_frecuencia["Frecuencia Relativa"] = tabla_frecuencia["Frecuencia Absoluta"] / tabla_frecuencia["Frecuencia Absoluta"].sum()
tabla_frecuencia["Frecuencia Acumulada"] = tabla_frecuencia["Frecuencia Absoluta"].cumsum()
tabla_frecuencia["Frecuencia Relativa Acumulada"] = tabla_frecuencia["Frecuencia Relativa"].cumsum()
print(tabla_frecuencia)

print("Frecuencia relativa: Proporción de cada categoría en relación con el total.")
print("Frecuencia Absoluta: La calificación 7 apareció 5 veces.")
print("Frecuencia Relativa: El 20% de los estudiantes obtuvo una calificación de 6.")
print("Frecuencia Acumulada: Hasta la calificación 8, ya se han registrado 12 estudiantes.")
print("Frecuencia Relativa Acumulada: El 80% de los estudiantes obtuvo una calificación de 8 o menor.\n")

# ===============================
# 5. Medidas de Tendencia Central
# ===============================
# Medidas de Tendencia Central
# - Media (Promedio): Es el valor promedio de un conjunto de datos.
# - Mediana: Es el valor central de un conjunto de datos ordenados.
# - Moda: Es el valor que aparece con mayor frecuencia.

print("### Ejemplo en Python:")
calificaciones = np.array([6, 7, 7, 8, 6, 9, 8, 7, 6, 7, 8, 9, 8, 7, 9])
print(f"Media: {np.mean(calificaciones)}")
print(f"Mediana: {np.median(calificaciones)}")
from scipy import stats
print(f"Moda: {stats.mode(calificaciones, keepdims=True).mode[0]}")

# ===============================
# 6. Medidas de Dispersión
# ===============================
# Medidas de Dispersión
# - Rango: Diferencia entre el valor máximo y mínimo.
# - Varianza: Indica cuánto se alejan los valores respecto a la media.
# - Desviación estándar: Raíz cuadrada de la varianza.

print("### Ejemplo en Python:")
print(f"Rango: {np.max(calificaciones) - np.min(calificaciones)}")
print(f"Varianza: {np.var(calificaciones)}")
print(f"Desviación estándar: {np.std(calificaciones)}\n")

# ===============================
# 7. Población y Muestra
# ===============================
# Población y Muestra
# - Población: Conjunto completo de datos sobre el cual se desea realizar un estudio.
# - Muestra: Subconjunto de la población seleccionado para análisis.

print("### Ejemplo en Python:")
poblacion = np.arange(1, 101)
muestra = np.random.choice(poblacion, size=10, replace=False)
print(f"Población: {poblacion}")
print(f"Muestra aleatoria: {muestra}\n")

# ===============================
# 8. Corrección de Bessel
# ===============================
# Corrección de Bessel
# Se usa al calcular la varianza y desviación estándar de una muestra.
# Se divide entre N-1 en lugar de N para corregir la subestimación de la variabilidad poblacional.

print("### Ejemplo en Python:")
print(f"Varianza poblacional: {np.var(muestra)}")
print(f"Varianza muestral (Bessel): {np.var(muestra, ddof=1)}\n")

# ===============================
# 9. Medidas de Posición
# ===============================
# Medidas de Posición
# - Cuartiles: Dividen los datos en cuatro partes iguales.
# - Quintiles: Cinco partes iguales.
# - Deciles: Diez partes iguales.
# - Percentiles: Cien partes iguales.

print("### Ejemplo en Python (Cuartiles):")
Q1 = np.percentile(calificaciones, 25)
Q2 = np.percentile(calificaciones, 50)
Q3 = np.percentile(calificaciones, 75)
print(f"Q1: {Q1}, Q2 (mediana): {Q2}, Q3: {Q3}\n")

# ===============================
# 10. Outliers (Puntos Atípicos)
# ===============================
# Outliers (Puntos Atípicos)
# Un outlier es un valor que se aleja significativamente del resto de las observaciones.

print("### Ejemplo en Python de detección de outliers con IQR:")
datos_outlier = np.array([10, 12, 12, 13, 12, 11, 14, 102])
Q1 = np.percentile(datos_outlier, 25)
Q3 = np.percentile(datos_outlier, 75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
outliers = datos_outlier[(datos_outlier < limite_inferior) | (datos_outlier > limite_superior)]
print(f"Datos originales: {datos_outlier}")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Límite inferior: {limite_inferior}, Límite superior: {limite_superior}")
print(f"Outliers detectados: {outliers}\n")

# ===============================
# 11. Visualización de Datos
# ===============================
# Visualización de Datos
# Histograma
# El histograma muestra la distribución de frecuencias de una variable numérica.

import matplotlib.pyplot as plt
plt.hist(calificaciones, bins=5, edgecolor="black")
plt.xlabel("Calificación")
plt.ylabel("Frecuencia")
plt.title("Histograma de Calificaciones")
plt.show()

print("\n## Boxplot (Diagrama de Caja)\nEl boxplot resume la distribución de datos mostrando la mediana, cuartiles y valores atípicos.\n")
import seaborn as sns
sns.boxplot(x=calificaciones)
plt.title("Boxplot de Calificaciones")
plt.show()

# ===============================
# 12. Actividad Práctica Guiada
# ===============================
print("# Actividad Práctica Guiada\n")
print("1. Importar librerías.\n2. Crear o cargar un conjunto de datos.\n3. Visualizar los datos.\n4. Identificar outliers usando el rango intercuartílico (IQR).\n5. Analizar los outliers.\n6. Manejar los outliers: eliminar si son errores, transformar datos, usar métodos robustos.\n7. Visualizar los datos sin outliers.\n")