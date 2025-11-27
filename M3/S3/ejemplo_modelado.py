# Paso 1: Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
 
# Paso 2: Generar datos sintéticos
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
plt.scatter(X, y, color="blue", label="Datos reales")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
 
# Paso 3: Modelos sin validación cruzada
results_simple = []
 
# Regresión lineal
model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_pred_lin = model_lin.predict(X_test)
results_simple.append({
    'Modelo': 'Regresión Lineal',
    'R²': r2_score(y_test, y_pred_lin),
    'MSE': mean_squared_error(y_test, y_pred_lin),
    'MAE': mean_absolute_error(y_test, y_pred_lin)
})
 
# Regresión Polinómica
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
results_simple.append({
    'Modelo': 'Regresión Polinómica',
    'R²': r2_score(y_test, y_pred_poly),
    'MSE': mean_squared_error(y_test, y_pred_poly),
    'MAE': mean_absolute_error(y_test, y_pred_poly)
})
 
# Árbol de decisión
model_tree = DecisionTreeRegressor(random_state=42)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
results_simple.append({
    'Modelo': 'Árbol de Decisión',
    'R²': r2_score(y_test, y_pred_tree),
    'MSE': mean_squared_error(y_test, y_pred_tree),
    'MAE': mean_absolute_error(y_test, y_pred_tree)
})
 
df_simple = pd.DataFrame(results_simple).set_index("Modelo")
 
# Paso 4: Gráfico scatter de comparaciones de predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Datos Reales", alpha=0.6)
plt.scatter(X_test, y_pred_lin, color="red", label="Regresión Lineal", marker='x')
plt.scatter(X_test, y_pred_poly, color="green", label="Regresión Polinómica", marker='^')
plt.scatter(X_test, y_pred_tree, color="purple", label="Árbol de Decisión", marker='s')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparación de Predicciones")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Datos Reales", alpha=0.6)
plt.plot(X_test, y_pred_lin, color="red", label="Regresión Lineal", linewidth=2)  # Línea de regresión
plt.scatter(X_test, y_pred_poly, color="green", label="Regresión Polinómica", marker='^')
plt.scatter(X_test, y_pred_tree, color="purple", label="Árbol de Decisión", marker='s')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparación de Predicciones")
plt.grid(True)
plt.show()
 
# Paso 5: Validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results_cv = []
 
# Linear Regression CV
scores_r2 = cross_val_score(model_lin, X, y, cv=cv, scoring='r2')
scores_mse = cross_val_score(model_lin, X, y, cv=cv, scoring='neg_mean_squared_error')
scores_mae = cross_val_score(model_lin, X, y, cv=cv, scoring='neg_mean_absolute_error')
results_cv.append({
    'Modelo': 'Regresión Lineal',
    'R²': scores_r2.mean(),
    'MSE': -scores_mse.mean(),
    'MAE': -scores_mae.mean()
})
 
# Polynomial Regression CV
X_poly = poly.fit_transform(X)
scores_r2 = cross_val_score(model_poly, X_poly, y, cv=cv, scoring='r2')
scores_mse = cross_val_score(model_poly, X_poly, y, cv=cv, scoring='neg_mean_squared_error')
scores_mae = cross_val_score(model_poly, X_poly, y, cv=cv, scoring='neg_mean_absolute_error')
results_cv.append({
    'Modelo': 'Regresión Polinómica',
    'R²': scores_r2.mean(),
    'MSE': -scores_mse.mean(),
    'MAE': -scores_mae.mean()
})
 
# Decision Tree CV
scores_r2 = cross_val_score(model_tree, X, y, cv=cv, scoring='r2')
scores_mse = cross_val_score(model_tree, X, y, cv=cv, scoring='neg_mean_squared_error')
scores_mae = cross_val_score(model_tree, X, y, cv=cv, scoring='neg_mean_absolute_error')
results_cv.append({
    'Modelo': 'Árbol de Decisión',
    'R²': scores_r2.mean(),
    'MSE': -scores_mse.mean(),
    'MAE': -scores_mae.mean()
})
 
df_cv = pd.DataFrame(results_cv).set_index("Modelo")
 
# Paso 6: Mostrar comparación por consola
print("\n=== Resultados en conjunto de Test ===\n")
print(df_simple.round(4))
 
print("\n=== Resultados con Validación Cruzada (CV) ===\n")
print(df_cv.round(4))
 
# Paso 7: Comparar lado a lado
df_comparado = pd.concat([df_simple.add_suffix(" (Test)"), df_cv.add_suffix(" (CV)")], axis=1)
print("\n=== Comparación lado a lado ===\n")
print(df_comparado.round(4))
 
# Paso 8: Visualización con valores en las barras
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
metrics = ["R²", "MSE", "MAE"]
titles = ["Coeficiente de Determinación (R²)", "Error Cuadrático Medio (MSE)", "Error Medio Absoluto (MAE)"]
 
for i, metric in enumerate(metrics):
    ax = axs[i]
    
    # Dibujar barras de test y CV
    bars_test = ax.bar(df_simple.index, df_simple[metric], width=0.4, label="Test", color="skyblue", align='center')
    bars_cv = ax.bar(df_cv.index, df_cv[metric], width=0.4, label="Cross-Validation", color="orange", align='edge')
    
    # Título, etiquetas y leyenda
    ax.set_title(titles[i])
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)
 
    # Agregar etiquetas de valores
    for bar in bars_test:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.05, height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars_cv:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2 + 0.05, height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
 
plt.suptitle("Comparativa de Modelos de Regresión", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
 
# Paso 9: Determinar mejor modelo según R² en test y en CV
mejor_modelo_test = df_simple["R²"].idxmax()
mejor_modelo_cv = df_cv["R²"].idxmax()
 
print(f"\n✅ El mejor modelo según R² en conjunto de test es: **{mejor_modelo_test}**")
print(f"✅ El mejor modelo según R² en validación cruzada es: **{mejor_modelo_cv}**")
 
 
# Paso 10: Conclusiones
print("""
\nConclusiones:
 
- La regresión polinómica es claramente el mejor modelo en ambos escenarios, con el mayor R² y los menores errores (MSE y MAE).
 
- El desempeño empeora ligeramente con validación cruzada, como es esperable por la evaluación más robusta.
 
- La regresión lineal es la menos precisa, especialmente en un problema con curvatura como este.
""")
 