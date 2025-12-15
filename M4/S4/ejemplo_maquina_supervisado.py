# Importaciones generales necesarias para los ejercicios
import numpy as np # Para operaciones numéricas, especialmente con datos sintéticos.
import pandas as pd # Para manejo y manipulación de datos en DataFrames.
from sklearn.model_selection import train_test_split, GridSearchCV # Para dividir datos y optimizar hiperparámetros.
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet # Modelos de regresión, incluyendo regularización.
from sklearn.tree import DecisionTreeClassifier # Modelo de árbol de decisión.
from sklearn.neighbors import KNeighborsClassifier # Modelo K-NN.
from sklearn.svm import SVC # Máquinas de Soporte Vectorial.
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Para transformación de datos (escalado, codificación).
from sklearn.compose import ColumnTransformer # Para aplicar transformaciones a columnas específicas.
from sklearn.pipeline import Pipeline # Para encadenar pasos de procesamiento y modelado.
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score # Métricas de evaluación de modelos.
# Importaciones para Balanceo de Datos (de la librería imbalanced-learn, que complementa scikit-learn)
from imblearn.over_sampling import RandomOverSampler, SMOTE # Técnicas de sobremuestreo.
from imblearn.under_sampling import RandomUnderSampler # Técnicas de submuestreo.
from imblearn.combine import SMOTEENN # Técnica híbrida de balanceo.
import warnings # Para manejar advertencias (se usa para suprimir algunas en los ejemplos).

# Suprimir advertencias para una salida más limpia en los ejemplos demostrativos.
warnings.filterwarnings('ignore')

# --- Página 1: Máquina Supervisado - Aprendizaje de - Mecanismos para mejorar el desempeño de un algoritmo ---

# Título del tema: Máquina Supervisado - Aprendizaje de
# Mecanismos para mejorar el desempeño de un algoritmo
# Explicación del contenido:
# Esta sección inicial introduce el concepto fundamental de la optimización de modelos en Machine Learning.
# Se destaca que la **optimización de modelos es un proceso esencial** para mejorar significativamente el
# rendimiento de los algoritmos. El propósito de la sesión es explorar diversas técnicas diseñadas
# para potenciar el desempeño de nuestros modelos, abarcando desde la **ingeniería de características**
# (Feature Engineering) hasta el **balanceo de datos**. Estas técnicas buscan hacer que los modelos
# sean más robustos, precisos y generalizables a datos no vistos.

# Ejercicio Demostrativo 1: Introducción a un flujo de trabajo básico de Machine Learning.
# Este ejercicio ilustra la idea de preparar datos y entrenar un modelo simple como punto de partida,
# antes de aplicar cualquier técnica de optimización.

print("\n" + "="*80)
print("--- Contenido de la Página 1: Introducción a la Optimización de Modelos ---")
print("="*80 + "\n")

try:
    # 1. Generación de datos sintéticos: Creamos un conjunto de datos simple para clasificación.
    #    Esto simula un escenario donde tenemos características (X) y una variable objetivo (y).
    np.random.seed(42) # Para asegurar que los resultados sean reproducibles.
    X_intro = np.random.rand(100, 5) * 10 # 100 muestras, 5 características aleatorias entre 0 y 10.
    y_intro = (X_intro[:, 0] + X_intro[:, 1] > 10).astype(int) # La clase se define por una regla simple.

    print("[Ejercicio Pág 1] Datos Sintéticos Generados:")
    print(f"  Forma de las características (X): {X_intro.shape}") # Muestra la cantidad de filas y columnas.
    print(f"  Forma de las etiquetas (y): {y_intro.shape}") # Muestra la cantidad de etiquetas.
    print("  Primeras 5 filas de X:\n", X_intro[:5])
    print("  Primeras 5 etiquetas de y:", y_intro[:5])

    # En un escenario real, los datos se cargarían desde un archivo:
    # df = pd.read_csv('ruta/a/tu_dataset.csv')
    # X = df.drop('columna_objetivo', axis=1) # Características.
    # y = df['columna_objetivo'] # Variable objetivo.

    # 2. División de datos: Separamos el conjunto de datos en entrenamiento y prueba.
    #    El conjunto de entrenamiento se usa para que el modelo aprenda.
    #    El conjunto de prueba se usa para evaluar el rendimiento del modelo en datos nuevos.
    X_train_intro, X_test_intro, y_train_intro, y_test_intro = train_test_split(
        X_intro, y_intro, test_size=0.3, random_state=42 # 30% para prueba, 70% para entrenamiento.
    )

    print("\n  Datos divididos en conjuntos de entrenamiento y prueba.")
    print(f"  Tamaño del conjunto de entrenamiento: {X_train_intro.shape} muestras")
    print(f"  Tamaño del conjunto de prueba: {X_test_intro.shape} muestras")

    # 3. Entrenamiento de un modelo básico: Entrenamos un modelo simple sin optimización.
    #    Usamos Regresión Logística, un clasificador común.
    model_basic = LogisticRegression(random_state=42)
    model_basic.fit(X_train_intro, y_train_intro) # El modelo "aprende" de los datos de entrenamiento.

    # 4. Evaluación inicial del modelo: Medimos el rendimiento del modelo en el conjunto de prueba.
    initial_accuracy = model_basic.score(X_test_intro, y_test_intro) # La precisión es una métrica común.
    print(f"\n  Precisión inicial del modelo básico (sin optimización): {initial_accuracy:.4f}")
    print("  Este es nuestro punto de partida. Las siguientes secciones mostrarán cómo mejorar este desempeño.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 1: {e}")

# --- Página 2: 2 Optimización de Hiperparámetros, 3 Regularización, 4 Balanceo de Datos ---

# Título del tema: Mecanismos para Mejorar el Desempeño
# Explicación del contenido:
# Esta página funciona como una **tabla de contenidos o resumen** de las principales técnicas
# para mejorar el desempeño de los modelos de Machine Learning. Se enumeran y definen
# brevemente tres categorías clave:
# 1. **Optimización de Hiperparámetros**: Proceso de ajustar las configuraciones (hiperparámetros)
#    que controlan el comportamiento del algoritmo antes de su entrenamiento. Estas configuraciones
#    impactan directamente el rendimiento y la capacidad del modelo para generalizar.
# 2. **Regularización**: Una técnica utilizada para **prevenir el sobreajuste** del modelo. Lo logra
#    penalizando la complejidad del modelo, lo que ayuda a que este generalice mejor a datos nuevos
#    y no vistos durante el entrenamiento.
# 3. **Balanceo de Datos**: Estrategias para abordar el problema de **clases desequilibradas**
#    en problemas de clasificación. El objetivo es evitar que el modelo se sesgue hacia la clase
#    mayoritaria y, en su lugar, mejorar la predicción de las clases minoritarias, que a menudo
#    son las más importantes.

# Ejercicio Demostrativo 2: Presentación conceptual de las técnicas.
# Este ejercicio simplemente refuerza los nombres y conceptos de las tres áreas clave que se detallarán.

print("\n" + "="*80)
print("--- Contenido de la Página 2: Resumen de Mecanismos de Optimización ---")
print("="*80 + "\n")

print("[Ejercicio Pág 2] Las técnicas principales que exploraremos para mejorar el desempeño de modelos son:")
print(f"  - **Optimización de Hiperparámetros**: Ajustar configuraciones del algoritmo para mejor rendimiento.")
print(f"  - **Regularización**: Prevenir el sobreajuste penalizando la complejidad del modelo.")
print(f"  - **Balanceo de Datos**: Manejar clases desequilibradas para evitar sesgos.")
print("\n  Cada una de estas técnicas se desarrollará en profundidad en las siguientes secciones.\n")


# --- Página 3: Feature Engineering ---

# Título del tema: Feature Engineering
# ¿Qué es?
# Explicación del contenido:
# La **Ingeniería de Características (Feature Engineering)** es un proceso crucial que implica
# **transformar, crear y seleccionar variables de entrada** para mejorar el rendimiento de un modelo.
# Su principal beneficio es que ayuda al modelo a **capturar mejor las relaciones** entre las variables
# de entrada y la variable objetivo, lo que puede conducir a un aprendizaje más efectivo.
# Los **pasos principales** en la Ingeniería de Características incluyen:
# - **Creación de nuevas características** a partir de las existentes (ej. combinaciones, transformaciones).
# - **Selección de características relevantes**, eliminando aquellas que son redundantes o no informativas.
# - **Transformación de datos** mediante técnicas como la normalización, el escalado (ej. StandardScaler)
#   o la codificación de variables categóricas (ej. One-Hot Encoding).

# Ejercicio Demostrativo 3: Ejemplo básico de Creación de Características y Escalado.
# Este ejercicio muestra cómo se pueden crear nuevas características a partir de datos existentes
# y cómo aplicar una transformación común como el escalado.

print("\n" + "="*80)
print("--- Contenido de la Página 3: Feature Engineering ---")
print("="*80 + "\n")

try:
    # 1. Generación de datos sintéticos para ilustrar la Ingeniería de Características.
    #    Simulamos datos de propiedades inmobiliarias.
    data_fe = pd.DataFrame({
        'num_habitaciones': np.random.randint(1, 5, 100), # Número de habitaciones.
        'superficie_m2': np.random.randint(50, 200, 100), # Superficie en metros cuadrados.
        'antiguedad_anos': np.random.randint(0, 30, 100), # Antigüedad de la propiedad.
        'tipo_propiedad': np.random.choice(['Casa', 'Apartamento'], 100) # Tipo de propiedad (categórica).
    })
    # Añadimos una columna 'precio' para que el DataFrame sea más realista, aunque no la usaremos en FE.
    data_fe['precio'] = (data_fe['superficie_m2'] * 500 + data_fe['num_habitaciones'] * 10000 - data_fe['antiguedad_anos'] * 1000) + np.random.randn(100) * 5000

    print("[Ejercicio Pág 3] Datos iniciales para Feature Engineering (primeras 5 filas):\n", data_fe.head())

    # 2. Paso 1: Creación de una nueva característica.
    #    Creamos 'superficie_por_habitacion' dividiendo la superficie por el número de habitaciones.
    #    Esta nueva característica puede ser más informativa que las originales por separado.
    data_fe['superficie_por_habitacion'] = data_fe['superficie_m2'] / data_fe['num_habitaciones']
    print("\n  Característica 'superficie_por_habitacion' creada.")

    # 3. Paso 2: Transformación de datos - Escalado de características numéricas.
    #    Muchos algoritmos de ML funcionan mejor cuando las características numéricas están en una escala similar.
    #    StandardScaler transforma los datos para que tengan media 0 y desviación estándar 1.
    numeric_features = ['num_habitaciones', 'superficie_m2', 'antiguedad_anos', 'superficie_por_habitacion']
    scaler = StandardScaler()
    # Aplicamos el escalado solo a las columnas numéricas.
    data_fe[numeric_features] = scaler.fit_transform(data_fe[numeric_features])
    print("\n  Características numéricas escaladas con StandardScaler.")

    # 4. Paso 3: Codificación de variables categóricas.
    #    Los modelos de ML generalmente requieren entradas numéricas. One-Hot Encoding convierte
    #    variables categóricas en columnas binarias (0 o 1).
    data_fe = pd.get_dummies(data_fe, columns=['tipo_propiedad'], drop_first=True) # drop_first evita la multicolinealidad.
    print("\n  Variable categórica 'tipo_propiedad' codificada (One-Hot Encoding).")

    print("\n  Datos después del Feature Engineering (primeras 5 filas con nuevas/transformadas características):\n", data_fe.head())
    print("  Nuevas/transformadas columnas de características:", data_fe.columns.tolist())
    print("\n  Este proceso ayuda a que el modelo interprete mejor los datos y aprenda relaciones más complejas,")
    print("  mejorando así su rendimiento.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 3: {e}")


# --- Página 4: Ejemplos Prácticos (Feature Engineering) y Hiperparámetros (Definición) ---

# Título del tema: Ejemplos Prácticos (Feature Engineering) y Hiperparámetros
# Explicación del contenido:
# Esta página proporciona **ejemplos prácticos de Ingeniería de Características**, como la creación de la
# característica "tamaño por habitación" en la predicción de precios de casas, o el uso de TF-IDF para
# representar numéricamente texto en problemas de clasificación de texto.
# A continuación, introduce el concepto de **Hiperparámetros**:
# Los hiperparámetros son **configuraciones que se establecen antes de entrenar un modelo**. A diferencia
# de los parámetros del modelo (que el algoritmo aprende durante el entrenamiento, como los coeficientes
# en regresión lineal), los hiperparámetros **controlan cómo el algoritmo lleva a cabo su aprendizaje**.
# Son, en esencia, "configuraciones de ajuste fino" para el proceso de aprendizaje del modelo.

# Ejercicio Demostrativo 4: Replicación de un ejemplo de FE y demostración de cómo se definen hiperparámetros.
# Este ejercicio ilustra un ejemplo concreto de Feature Engineering mencionado en la fuente y muestra
# cómo se asignan los hiperparámetros al instanciar un modelo.

print("\n" + "="*80)
print("--- Contenido de la Página 4: Ejemplos de Feature Engineering y Definición de Hiperparámetros ---")
print("="*80 + "\n")

try:
    # 1. Ejemplo práctico de Feature Engineering: "tamaño por habitación"
    print("[Ejercicio Pág 4] Ejemplo de Feature Engineering: 'tamaño por habitación' para casas.")
    df_casas = pd.DataFrame({
        'Tamano_Total_m2': [100, 150, 200, 120, 180], # Superficie total en m².
        'Num_Habitaciones': [2, 3, 4, 2, 5] # Número de habitaciones.
    })
    # Se crea la nueva característica dividiendo el tamaño total por el número de habitaciones.
    df_casas['Tamano_por_Habitacion'] = df_casas['Tamano_Total_m2'] / df_casas['Num_Habitaciones']
    print("  Datos de casas con nueva característica 'Tamano_por_Habitacion':\n", df_casas)

    # 2. Introducción a los Hiperparámetros
    print("\n[Ejercicio Pág 4] Concepto de Hiperparámetros.")
    print("  Los hiperparámetros son configuraciones que se ajustan **ANTES** de que el modelo se entrene.")
    print("  Son diferentes de los parámetros del modelo que se aprenden durante el entrenamiento.")

    # Definición de un modelo con hiperparámetros iniciales.
    # Para un Decision Tree, `max_depth` (profundidad máxima) es un hiperparámetro común.
    # Lo asignamos directamente al crear la instancia del modelo.
    dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
    print(f"\n  Modelo DecisionTreeClassifier inicializado con max_depth={dt_classifier.max_depth}.")
    print("  Aquí, 'max_depth' es un hiperparámetro; nosotros lo configuramos manualmente,")
    print("  no es algo que el modelo aprenda de los datos durante el entrenamiento.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 4: {e}")


# --- Página 5: Hiperparámetros (Importancia y Ejemplos por Algoritmo) ---

# Título del tema: Hiperparámetros
# Importancia y Ejemplos por Algoritmo
# Explicación del contenido:
# Esta página enfatiza la **importancia crítica de los hiperparámetros**. Una elección adecuada
# afecta directamente el **rendimiento del modelo y su capacidad de generalización** a datos nuevos.
# Los hiperparámetros también **controlan la complejidad del modelo**, siendo fundamentales para
# **evitar el sobreajuste** (cuando el modelo memoriza el ruido de los datos de entrenamiento
# en lugar de aprender patrones generales). Una mala selección puede llevar a modelos que
# no se comportan bien con datos no vistos.
# Se proporcionan **ejemplos de hiperparámetros específicos** para diferentes algoritmos:
# - **Árboles de decisión**: `max_depth` (profundidad máxima), `min_samples_split` (mínimo de muestras para dividir).
# - **Regresión regularizada**: `alpha` (fuerza de regularización).
# - **K-NN (K-Nearest Neighbors)**: `n_neighbors` (número de vecinos).
# - **SVM (Support Vector Machine)**: `C` (parámetro de penalización), `kernel` (tipo de kernel).

# Ejercicio Demostrativo 5: Ilustración de diversos hiperparámetros en diferentes algoritmos.
# Este ejercicio muestra cómo se accede y se configuran algunos de los hiperparámetros
# mencionados para distintos tipos de modelos en scikit-learn.

print("\n" + "="*80)
print("--- Contenido de la Página 5: Importancia y Ejemplos de Hiperparámetros ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 5] Importancia de los Hiperparámetros y Ejemplos por Algoritmo.")
    print("  La configuración adecuada de hiperparámetros es vital para el rendimiento y la generalización.")

    # 1. Hiperparámetros para un Árbol de Decisión (Decision Tree)
    dt_model_params = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    print(f"\n  Decision Tree Classifier:")
    print(f"    - `max_depth` (profundidad máxima del árbol): {dt_model_params.max_depth}")
    print(f"    - `min_samples_split` (mínimo de muestras para dividir un nodo): {dt_model_params.min_samples_split}")

    # 2. Hiperparámetros para Regresión Logística (análogo a regresión regularizada)
    #    El parámetro `C` en LogisticRegression es el inverso de `alpha` en otras regresiones regularizadas.
    #    Un `C` más pequeño implica mayor regularización (análogo a un `alpha` más grande).
    lr_model_params = LogisticRegression(C=0.1, penalty='l2', random_state=42)
    print(f"\n  Logistic Regression (con regularización L2):")
    print(f"    - `C` (inverso de la fuerza de regularización): {lr_model_params.C}")
    print(f"    - `penalty` (tipo de regularización): '{lr_model_params.penalty}'")

    # 3. Hiperparámetros para K-Nearest Neighbors (K-NN)
    knn_model_params = KNeighborsClassifier(n_neighbors=7)
    print(f"\n  K-Nearest Neighbors Classifier:")
    print(f"    - `n_neighbors` (número de vecinos a considerar): {knn_model_params.n_neighbors}")

    # 4. Hiperparámetros para Support Vector Machine (SVC)
    svm_model_params = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    print(f"\n  Support Vector Machine (SVC):")
    print(f"    - `C` (parámetro de penalización de error): {svm_model_params.C}")
    print(f"    - `kernel` (tipo de función kernel): '{svm_model_params.kernel}'")

    print("\n  Cada algoritmo tiene un conjunto único de hiperparámetros que controlan su comportamiento y capacidad de aprendizaje.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 5: {e}")


# --- Página 6: Regularización (Concepto y L1/Lasso) ---

# Título del tema: Regularización
# Concepto y Regularización L1 (Lasso)
# Explicación del contenido:
# La **regularización** es una técnica fundamental utilizada para **prevenir el sobreajuste**
# en los modelos de Machine Learning. Funciona penalizando la complejidad del modelo.
# Esto se logra añadiendo un término de penalización a la función de costo del modelo, lo que
# tiene el efecto de **reducir la magnitud de los coeficientes** de las características.
# Al reducir estos coeficientes, el modelo se vuelve más simple y, por lo tanto, tiene una
# mejor capacidad para **generalizar** a datos nuevos y no vistos.
# La **Regularización L1**, también conocida como **Lasso (Least Absolute Shrinkage and Selection Operator)**,
# añade una penalización proporcional al **valor absoluto** de los coeficientes del modelo.
# Una característica distintiva de Lasso es que puede reducir algunos coeficientes **exactamente a cero**,
# lo que lo convierte en un **selector de características automático**. Esto es particularmente útil
# cuando se sospecha que muchas características en el conjunto de datos son irrelevantes.

# Ejercicio Demostrativo 6: Demostración de Regularización L1 (Lasso).
# Creamos un conjunto de datos con algunas características realmente irrelevantes y mostramos cómo
# Lasso puede reducir sus coeficientes a cero.

print("\n" + "="*80)
print("--- Contenido de la Página 6: Regularización (Concepto y L1/Lasso) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 6] Regularización L1 (Lasso).")
    # 1. Generar datos sintéticos con características relevantes e irrelevantes.
    np.random.seed(42)
    n_samples = 100
    n_features = 10 # Creamos 10 características.
    X_reg = np.random.rand(n_samples, n_features) * 10 # Características aleatorias.

    # La variable objetivo 'y' dependerá solo de las primeras 3 características (relevantes).
    # Las características restantes (X_reg[:, 3:]) son irrelevantes para la predicción de 'y'.
    y_reg = (3 * X_reg[:, 0] + 2 * X_reg[:, 1] - 1 * X_reg[:, 2] + np.random.randn(n_samples) * 0.5).astype(int)
    # Convertimos 'y' a entero para usar LogisticRegression de manera más sencilla en el ejemplo conceptual.

    # 2. División de los datos en conjuntos de entrenamiento y prueba.
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    # 3. Entrenamiento de un modelo de Regresión Logística sin regularización (para comparación).
    #    Aquí `penalty='none'` indica que no se aplicará regularización.
    #    Usamos `solver='lbfgs'` y `max_iter` para evitar advertencias.
    print("\n  Coeficientes del modelo de Regresión Logística sin regularización:")
    model_no_reg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000, random_state=42)
    model_no_reg.fit(X_train_reg, y_train_reg)
    print(np.round(model_no_reg.coef_, 4)) # Mostramos los coeficientes del modelo.
    print("  Nota: Observe que todos los coeficientes tienen valores distintos de cero.")

    # 4. Aplicación de Regularización L1 (Lasso).
    #    `LogisticRegression` permite usar `penalty='l1'`. El parámetro `C` es el inverso de `alpha`.
    #    Un `C` más pequeño aumenta la fuerza de la regularización L1.
    print("\n  Coeficientes del modelo con Regularización L1 (Lasso):")
    lasso_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42) # `liblinear` soporta L1.
    lasso_model.fit(X_train_reg, y_train_reg)

    # Observar cómo algunos coeficientes se acercan o se vuelven exactamente cero.
    print(np.round(lasso_model.coef_, 4))
    print("\n  Nota: Observe que varios coeficientes de las características irrelevantes (ej. las últimas) ")
    print("  se han reducido a cero o valores muy cercanos a cero.")
    print("  Esto demuestra la capacidad de Lasso para realizar una **selección automática de características**.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 6: {e}")


# --- Página 7: Regularización (L2/Ridge y Elastic Net) ---

# Título del tema: Regularización L2 (Ridge) y Elastic Net
# Explicación del contenido:
# La **Regularización L2**, conocida como **Ridge Regression**, añade un término de penalización
# proporcional al **cuadrado de la magnitud de los coeficientes** del modelo.
# A diferencia de Lasso, Ridge **reduce la magnitud de todos los coeficientes** pero **no los elimina
# completamente** (es decir, no los hace exactamente cero). Esta técnica es particularmente útil
# cuando se considera que **todas las características son potencialmente relevantes** y se desea
# simplemente reducir su impacto general en el modelo, distribuyendo el efecto entre ellas.
# **Elastic Net** es una técnica de regularización que **combina las penalizaciones L1 y L2**.
# Esto le permite ofrecer un **equilibrio** entre la selección de características (propiedad de L1)
# y la reducción de la magnitud de los coeficientes (propiedad de L2). Elastic Net es
# especialmente valiosa en escenarios donde existen **muchas características correlacionadas**,
# ya que puede manejar mejor estas situaciones que L1 o L2 por separado.

# Ejercicio Demostrativo 7: Demostración de Regularización L2 (Ridge) y Elastic Net.
# Continuamos con el conjunto de datos sintéticos de la página anterior para comparar
# los efectos de Ridge y Elastic Net en los coeficientes del modelo.

print("\n" + "="*80)
print("--- Contenido de la Página 7: Regularización (L2/Ridge y Elastic Net) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 7] Regularización L2 (Ridge) y Elastic Net.")
    # Reutilizamos los datos X_train_reg y y_train_reg generados en el ejercicio anterior (Pág 6).

    # 1. Aplicar Regularización L2 (Ridge) usando LogisticRegression.
    #    `penalty='l2'` indica regularización L2. `C` es el inverso de `alpha` (fuerza de regularización).
    print("\n  Coeficientes del modelo con Regularización L2 (Ridge):")
    ridge_model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=42)
    ridge_model.fit(X_train_reg, y_train_reg)

    print(np.round(ridge_model.coef_, 4))
    print("  Nota: Los coeficientes se han reducido en magnitud, pero **ninguno se ha vuelto exactamente cero**.")
    print("  Esto contrasta con Lasso, donde algunos coeficientes sí se anulaban.")

    # 2. Aplicar Elastic Net usando LogisticRegression.
    #    `penalty='elasticnet'` combina L1 y L2. `l1_ratio` controla la mezcla (0=L2 pura, 1=L1 pura).
    #    `solver='saga'` es necesario para `elasticnet`.
    print("\n  Coeficientes del modelo Elastic Net (combinación L1 y L2):")
    elastic_net_model = LogisticRegression(penalty='elasticnet', C=0.1, l1_ratio=0.5, solver='saga', max_iter=1000, random_state=42)
    elastic_net_model.fit(X_train_reg, y_train_reg)

    print(np.round(elastic_net_model.coef_, 4))
    print("\n  Nota: Elastic Net ofrece un equilibrio entre la selección de características (como L1) ")
    print("  y la reducción de la magnitud de coeficientes (como L2).")
    print("  Es especialmente útil cuando hay muchas características correlacionadas.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 7: {e}")


# --- Página 8: Optimización de Parámetros (Espacio de Búsqueda y Estrategia de Búsqueda) ---

# Título del tema: Optimización de Parámetros
# Definir el Espacio de Búsqueda y Seleccionar la Estrategia de Búsqueda
# Explicación del contenido:
# La optimización de hiperparámetros es un proceso sistemático que involucra varios pasos.
# El primero es **Definir el Espacio de Búsqueda**: Esto implica identificar cuáles
# hiperparámetros se van a optimizar y establecer los rangos o conjuntos de valores
# posibles que se probarán para cada uno. Por ejemplo, para un modelo como
# Random Forest, se podrían optimizar hiperparámetros como `n_estimators` (número de árboles),
# `max_depth` (profundidad máxima de cada árbol) y `min_samples_split` (mínimo de muestras
# para dividir un nodo).
# El segundo paso es **Seleccionar la Estrategia de Búsqueda**: Se debe elegir el método
# que se utilizará para explorar este espacio de hiperparámetros. Las opciones comunes
# incluyen **Grid Search** (que es exhaustivo pero computacionalmente costoso),
# **Random Search** (más eficiente cuando hay muchos hiperparámetros ya que muestrea
# aleatoriamente el espacio), y la **Optimización Bayesiana** (una estrategia más avanzada
# y generalmente más eficiente, que utiliza modelos probabilísticos para guiar la búsqueda).
# La elección de la estrategia dependerá del contexto, el tamaño del espacio de búsqueda y
# los recursos computacionales disponibles.

# Ejercicio Demostrativo 8: Definición de un espacio de búsqueda para un modelo.
# Este ejercicio muestra cómo se estructura el diccionario de parámetros que se utiliza
# como "espacio de búsqueda" para técnicas como `GridSearchCV` o `RandomizedSearchCV` en scikit-learn.

print("\n" + "="*80)
print("--- Contenido de la Página 8: Optimización de Parámetros (Espacio y Estrategia de Búsqueda) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 8] Definición del Espacio de Búsqueda de Hiperparámetros.")

    # 1. Definición del espacio de búsqueda para un RandomForestClassifier.
    #    Se crea un diccionario donde las claves son los nombres de los hiperparámetros
    #    y los valores son listas de los posibles valores que se desean probar.
    param_grid_rf = {
        'n_estimators': [50, 100, 200],         # Número de árboles en el bosque (ej. 50, 100 o 200 árboles).
        'max_depth': [None, 10, 20],            # Profundidad máxima de cada árbol (None significa sin límite).
        'min_samples_split': [2, 5, 10]         # Mínimo de muestras requeridas para dividir un nodo interno.
    }

    print("\n  Espacio de búsqueda de hiperparámetros definido para un RandomForestClassifier:")
    for param, values in param_grid_rf.items():
        print(f"    - {param}: {values}")

    print("\n  2. Selección de la Estrategia de Búsqueda:")
    print("    - **Grid Search**: Explora exhaustivamente cada combinación. Ideal para espacios pequeños.")
    print("    - **Random Search**: Muestrea combinaciones aleatoriamente. Más eficiente para espacios grandes.")
    print("    - **Optimización Bayesiana**: Utiliza un modelo probabilístico para guiar una búsqueda más inteligente.")
    print("  La elección de la estrategia depende del tamaño del problema y los recursos computacionales.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 8: {e}")


# --- Página 9: Optimización de Parámetros (Validación Cruzada y Selección del Mejor Modelo) ---

# Título del tema: Optimización de Parámetros
# Implementar la Validación Cruzada y Seleccionar el Mejor Modelo
# Explicación del contenido:
# Continuando con el proceso de optimización de parámetros:
# 3. **Implementar la Validación Cruzada**: Es crucial usar técnicas como **k-fold cross-validation**
#    para evaluar cada combinación de hiperparámetros de una manera robusta y fiable. La validación
#    cruzada ayuda a evitar el **sobreajuste** a un conjunto específico de datos de validación, proporcionando
#    una estimación más fiable del rendimiento del modelo en datos no vistos.
# 4. **Seleccionar el Mejor Modelo**: Después de evaluar todas las combinaciones de hiperparámetros
#    mediante validación cruzada, se compara el rendimiento de cada configuración utilizando métricas
#    relevantes para el problema (como **precisión, recall o F1-score**). La combinación que
#    ofrezca el mejor resultado según estas métricas se selecciona como la óptima para el modelo final.

# Ejercicio Demostrativo 9: Uso de GridSearchCV con Validación Cruzada.
# Este ejercicio muestra cómo se integra la definición del espacio de búsqueda (de la Pág 8)
# con la validación cruzada para encontrar los mejores hiperparámetros y seleccionar el modelo óptimo.

print("\n" + "="*80)
print("--- Contenido de la Página 9: Optimización de Parámetros (Validación Cruzada y Selección) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 9] Implementación de Validación Cruzada y Selección del Mejor Modelo.")
    # 1. Generar datos sintéticos para el ejemplo de Grid Search.
    np.random.seed(42)
    X_gs = np.random.rand(150, 4) # 150 muestras, 4 características.
    y_gs = (X_gs[:, 0] + X_gs[:, 1] > 1.0).astype(int) # Clase binaria.

    # Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
        X_gs, y_gs, test_size=0.3, random_state=42
    )

    # 2. Definir el modelo base (en este caso, un Decision Tree Classifier).
    dt_classifier_gs = DecisionTreeClassifier(random_state=42)

    # 3. Definir el espacio de búsqueda de hiperparámetros.
    param_grid_dt = {
        'max_depth': [None, 3, 5, 10],        # Profundidad máxima del árbol.
        'min_samples_leaf': [1, 2, 4]      # Mínimo de muestras por hoja.
    }

    # 4. Crear el objeto GridSearchCV.
    #    `cv=5` indica que se usará 5-fold cross-validation para evaluar cada combinación.
    #    `scoring='accuracy'` define la métrica para seleccionar el mejor modelo.
    #    `n_jobs=-1` permite usar todos los núcleos del procesador para acelerar la búsqueda.
    grid_search = GridSearchCV(
        estimator=dt_classifier_gs,
        param_grid=param_grid_dt,
        cv=5,                 # Número de folds para la validación cruzada.
        scoring='accuracy',   # Métrica para evaluar y seleccionar el mejor modelo.
        n_jobs=-1             # Utiliza todos los procesadores disponibles.
    )

    print("\n  Iniciando Grid Search con validación cruzada (esto puede tomar unos segundos)...")
    grid_search.fit(X_train_gs, y_train_gs) # Ejecuta la búsqueda en el conjunto de entrenamiento.

    # 5. Obtener los mejores hiperparámetros y el mejor score de validación cruzada.
    print(f"\n  Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    print(f"  Mejor score de precisión (validación cruzada): {grid_search.best_score_:.4f}")

    # 6. Evaluar el mejor modelo en el conjunto de prueba (datos no vistos durante la optimización).
    best_dt_model = grid_search.best_estimator_ # Obtiene el modelo con los mejores hiperparámetros.
    test_accuracy = best_dt_model.score(X_test_gs, y_test_gs)
    print(f"  Precisión del mejor modelo en el conjunto de prueba: {test_accuracy:.4f}")

    print("\n  Este proceso de validación cruzada garantiza que el modelo elegido")
    print("  genere resultados robustos y generalice bien a nuevos datos.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 9: {e}")


# --- Página 10: Grilla de Parámetros (Definición, Especificación y Creación) ---

# Título del tema: Grilla de Parámetros
# 1. Definición de Hiperparámetros, 2. Especificación de Valores, 3. Creación de la Grilla
# Explicación del contenido:
# Esta sección detalla los primeros pasos para construir una "grilla de parámetros" para la
# optimización de hiperparámetros, que es esencialmente el diccionario que define el espacio
# de búsqueda.
# 1. **Definición de Hiperparámetros**: El primer paso es identificar los hiperparámetros
#    clave que se desean optimizar para un modelo específico. Por ejemplo, para un
#    Random Forest, se pueden elegir `n_estimators`, `max_depth` y `min_samples_split`.
# 2. **Especificación de Valores**: Para cada hiperparámetro identificado, se debe definir
#    un conjunto de valores posibles que el algoritmo de búsqueda probará. Por ejemplo,
#    para `n_estimators` podríamos probar ; para `max_depth`, [None, 10, 20];
#    y para `min_samples_split`,.
# 3. **Creación de la Grilla**: Con los hiperparámetros y sus valores definidos, se genera
#    la "grilla" propiamente dicha. Esta grilla consiste en **todas las combinaciones posibles**
#    de los valores de los hiperparámetros. Por ejemplo, si tenemos 3 valores para cada uno
#    de 3 hiperparámetros, la grilla tendrá 3 × 3 × 3 = 27 combinaciones distintas para evaluar.

# Ejercicio Demostrativo 10: Construcción explícita de una grilla de parámetros.
# Este ejercicio muestra cómo se define el diccionario de Python que representa la grilla
# de parámetros, tal como se usaría en `GridSearchCV`.

print("\n" + "="*80)
print("--- Contenido de la Página 10: Grilla de Parámetros (Definición y Creación) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 10] Definición y Creación de la Grilla de Parámetros.")

    # 1. Identificación de Hiperparámetros Clave.
    #    Aquí especificamos los hiperparámetros de un RandomForestClassifier que queremos ajustar.
    print("\n  1. Identificación de Hiperparámetros Clave (ej. para un Random Forest):")
    print("    - `n_estimators` (número de árboles en el bosque)")
    print("    - `max_depth` (profundidad máxima de cada árbol)")
    print("    - `min_samples_split` (número mínimo de muestras para dividir un nodo)")

    # 2. Especificación de Valores Posibles para cada Hiperparámetro.
    #    Definimos listas de valores que el Grid Search probará para cada hiperparámetro.
    print("\n  2. Especificación de Valores Posibles:")
    n_estimators_values = [50, 100, 200]
    max_depth_values = [None, 10, 20]
    min_samples_split_values = [2, 5, 10]
    print(f"    `n_estimators`: {n_estimators_values}")
    print(f"    `max_depth`: {max_depth_values}")
    print(f"    `min_samples_split`: {min_samples_split_values}")

    # 3. Creación de la Grilla (Diccionario de Parámetros).
    #    Este diccionario será la `param_grid` que se pasa a GridSearchCV.
    param_grid_example = {
        'n_estimators': n_estimators_values,
        'max_depth': max_depth_values,
        'min_samples_split': min_samples_split_values
    }

    print("\n  3. Grilla de Parámetros Resultante (como diccionario Python para GridSearchCV):")
    print(param_grid_example)

    # Calcular el número total de combinaciones que se evaluarán.
    num_combinations = (len(n_estimators_values) *
                        len(max_depth_values) *
                        len(min_samples_split_values))
    print(f"\n  Número total de combinaciones a evaluar en esta grilla: {num_combinations}.")
    print("  Cada una de estas combinaciones será entrenada y evaluada mediante validación cruzada.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 10: {e}")


# --- Página 11: Grilla de Parámetros (Evaluación y Selección) y Balanceo de Datos (Problema del Desbalanceo) ---

# Título del tema: Grilla de Parámetros y Balanceo de Datos
# Explicación del contenido:
# Esta página finaliza la sección de "Grilla de Parámetros" y comienza con "Balanceo de Datos".
# **Grilla de Parámetros - 4. Evaluación y Selección**: Después de crear la grilla, el siguiente
# paso es entrenar y evaluar el modelo con cada una de las combinaciones de hiperparámetros
# definidas en la grilla. Esto se realiza utilizando validación cruzada para obtener estimaciones
# robustas del rendimiento. Una vez evaluadas todas las combinaciones, se selecciona aquella
# que proporcione el mejor resultado según las métricas de rendimiento elegidas (ej. precisión,
# recall, F1-score).
# **Problema del Desbalanceo**: Introduce un problema común en clasificación: el **desbalanceo de clases**.
# Ocurre cuando una clase en el conjunto de datos tiene muchísimas más muestras que otra (clase minoritaria).
# Esto puede llevar a que el modelo se sesgue hacia la clase mayoritaria, lo que resulta en un **rendimiento
# deficiente para la clase minoritaria**, que a menudo es la de mayor interés (ej. detección de fraude,
# enfermedades raras).

# Ejercicio Demostrativo 11: Finalización de Grid Search y creación de un dataset desbalanceado.
# Este ejercicio ilustra la etapa final de la optimización de hiperparámetros y luego crea un
# conjunto de datos sintético para demostrar el problema del desbalanceo de clases.

print("\n" + "="*80)
print("--- Contenido de la Página 11: Grilla de Parámetros (Evaluación) y Problema de Desbalanceo ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 11] Evaluación y Selección de la Grilla de Parámetros.")
    # Retomamos los resultados de GridSearchCV del Ejercicio Pág 9.
    # Aquí simulamos que ya se ha ejecutado el Grid Search.
    print(f"  Los mejores hiperparámetros encontrados en el Grid Search fueron: {grid_search.best_params_}")
    print(f"  Y el mejor score de precisión (validación cruzada) fue: {grid_search.best_score_:.4f}")
    print("  Este es el paso final en la optimización de hiperparámetros: seleccionar la mejor configuración encontrada.\n")

    print("\n[Ejercicio Pág 11] Problema del Desbalanceo de Datos.")
    # 1. Generar un conjunto de datos sintético para ilustrar el desbalanceo.
    #    Crearemos un dataset donde la clase 0 (mayoritaria) tiene ~90% de las muestras y la clase 1 (minoritaria) ~10%.
    np.random.seed(42)
    n_samples = 1000 # Total de muestras.
    n_features = 2 # Número de características.
    X_imbalanced = np.random.randn(n_samples, n_features) # Características aleatorias.

    y_imbalanced = np.zeros(n_samples, dtype=int) # Inicializamos todas las etiquetas a 0 (clase mayoritaria).
    # Seleccionamos aleatoriamente un pequeño porcentaje de muestras para que sean de la clase minoritaria (1).
    minority_indices = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
    y_imbalanced[minority_indices] = 1 # Asignamos la etiqueta 1 a las muestras minoritarias.

    print(f"  Número total de muestras en el dataset: {len(y_imbalanced)}")
    print(f"  Conteo de clases en el dataset desbalanceado: {pd.Series(y_imbalanced).value_counts()}")
    print(f"  Proporción de la clase 0 (mayoritaria): {np.sum(y_imbalanced == 0) / len(y_imbalanced):.2f}")
    print(f"  Proporción de la clase 1 (minoritaria): {np.sum(y_imbalanced == 1) / len(y_imbalanced):.2f}")
    print("\n  Un dataset tan desbalanceado puede hacer que un modelo ignore la clase minoritaria,")
    print("  llevando a un bajo rendimiento en la detección de eventos importantes de esa clase.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 11: {e}")


# --- Página 12: Balanceo de Datos (Submuestreo y Sobremuestreo) ---

# Título del tema: Balanceo de Datos
# Submuestreo (Undersampling) y Sobremuestreo (Oversampling)
# Explicación del contenido:
# Esta página presenta dos estrategias principales para manejar el desbalanceo de clases:
# 1. **Submuestreo (Undersampling)**: Esta técnica busca equilibrar las clases **reduciendo el número
#    de muestras de la clase mayoritaria**. Es un método simple y rápido. Sin embargo, su principal
#    desventaja es que puede **perder información valiosa** al descartar muestras de la clase mayoritaria.
#    Es más útil cuando se dispone de una gran cantidad de datos en la clase mayoritaria.
# 2. **Sobremuestreo (Oversampling)**: Esta estrategia busca equilibrar las clases **aumentando el
#    número de muestras de la clase minoritaria**. Esto se puede lograr duplicando muestras existentes
#    o generando nuevas muestras. A diferencia del submuestreo, el sobremuestreo **no pierde información**.
#    Sin embargo, si se duplican exactamente las mismas muestras, puede causar **sobreajuste** (el modelo
#    memoriza esas muestras duplicadas). `RandomOverSampler` es una implementación común de esta técnica.

# Ejercicio Demostrativo 12: Aplicación de Submuestreo y Sobremuestreo.
# Usaremos el dataset desbalanceado generado en el ejercicio de la Página 11 para demostrar
# cómo estas técnicas modifican la distribución de las clases.

print("\n" + "="*80)
print("--- Contenido de la Página 12: Balanceo de Datos (Submuestreo y Sobremuestreo) ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 12] Demostración de Submuestreo (Undersampling) y Sobremuestreo (Oversampling).")
    # Reutilizamos las variables `X_imbalanced` y `y_imbalanced` del ejercicio anterior (Pág 11).
    print(f"  Distribución original de clases: {pd.Series(y_imbalanced).value_counts()}")

    # 1. **Submuestreo (Undersampling) con RandomUnderSampler.**
    #    Reduce el número de muestras de la clase mayoritaria (0) para que coincida con la minoritaria (1).
    rus = RandomUnderSampler(random_state=42)
    X_resampled_rus, y_resampled_rus = rus.fit_resample(X_imbalanced, y_imbalanced)

    print("\n  Después de aplicar Submuestreo (RandomUnderSampler):")
    print(f"    Conteo de clases balanceadas: {pd.Series(y_resampled_rus).value_counts()}")
    print(f"    Tamaño del dataset original: {len(y_imbalanced)} vs. tamaño resampleado: {len(y_resampled_rus)}")
    print("  Nota: El submuestreo reduce el tamaño total del dataset, lo que puede implicar pérdida de información valiosa.")

    # 2. **Sobremuestreo (Oversampling) con RandomOverSampler.**
    #    Aumenta el número de muestras de la clase minoritaria (1) para que coincida con la mayoritaria (0).
    ros = RandomOverSampler(random_state=42)
    X_resampled_ros, y_resampled_ros = ros.fit_resample(X_imbalanced, y_imbalanced)

    print("\n  Después de aplicar Sobremuestreo (RandomOverSampler):")
    print(f"    Conteo de clases balanceadas: {pd.Series(y_resampled_ros).value_counts()}")
    print(f"    Tamaño del dataset original: {len(y_imbalanced)} vs. tamaño resampleado: {len(y_resampled_ros)}")
    print("  Nota: El sobremuestreo aumenta el tamaño total del dataset y evita la pérdida de información.")
    print("  Sin embargo, la simple duplicación puede llevar a sobreajuste si no se usa con cuidado.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 12: {e}")


# --- Página 13: Balanceo de Datos (SMOTE y Técnicas Híbridas) e Implementación ---

# Título del tema: SMOTE y Técnicas Híbridas
# Implementación en Scikit-Learn
# Explicación del contenido:
# Esta página introduce técnicas más avanzadas para el balanceo de datos:
# 1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Esta técnica genera **muestras sintéticas**
#    para la clase minoritaria. Lo hace interpolando entre las muestras existentes de la clase minoritaria,
#    creando nuevos puntos de datos que no son copias exactas. El beneficio clave de SMOTE es que
#    **reduce el riesgo de sobreajuste** que podría ocurrir si simplemente se duplicaran las muestras existentes.
# 2. **Técnicas Híbridas**: Estas combinan elementos de sobremuestreo y submuestreo. Un ejemplo es **SMOTEENN**,
#    que combina SMOTE (sobremuestreo sintético) con Edited Nearest Neighbours (ENN, una forma de submuestreo
#    para limpiar el ruido en el conjunto de datos) para lograr un equilibrio aún mejor entre las clases.
#    Estas técnicas se implementan comúnmente utilizando las bibliotecas **`scikit-learn`** e **`imbalanced-learn`**
#    en Python. La diapositiva también menciona una gráfica (no incluida en el texto fuente) que muestra
#    el rendimiento de diferentes técnicas de balanceo en métricas como precisión y recall.

# Ejercicio Demostrativo 13: Aplicación de SMOTE y SMOTEENN.
# Continuamos con el dataset desbalanceado para demostrar cómo estas técnicas avanzadas generan o eliminan
# muestras para lograr el balance.

print("\n" + "="*80)
print("--- Contenido de la Página 13: SMOTE y Técnicas Híbridas de Balanceo ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 13] Demostración de SMOTE y Técnicas Híbridas (SMOTEENN).")
    # Reutilizamos las variables `X_imbalanced` y `y_imbalanced` del ejercicio anterior (Pág 11).
    print(f"  Distribución original de clases: {pd.Series(y_imbalanced).value_counts()}")

    # 1. **SMOTE (Synthetic Minority Over-sampling Technique).**
    #    Crea nuevas muestras sintéticas para la clase minoritaria, interpolando entre las existentes.
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_imbalanced, y_imbalanced)

    print("\n  Después de aplicar SMOTE:")
    print(f"    Conteo de clases balanceadas: {pd.Series(y_resampled_smote).value_counts()}")
    print(f"    Tamaño del dataset original: {len(y_imbalanced)} vs. tamaño resampleado: {len(y_resampled_smote)}")
    print("  Nota: SMOTE aumenta el número de muestras de la clase minoritaria creando ejemplos *sintéticos*,")
    print("  lo que reduce el riesgo de sobreajuste en comparación con la simple duplicación.")

    # 2. **SMOTEENN (Técnica Híbrida).**
    #    Combina SMOTE (sobremuestreo sintético) con Edited Nearest Neighbours (submuestreo para limpiar ruido).
    smoteenn = SMOTEENN(random_state=42)
    X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X_imbalanced, y_imbalanced)

    print("\n  Después de aplicar SMOTEENN (Híbrido):")
    print(f"    Conteo de clases balanceadas: {pd.Series(y_resampled_smoteenn).value_counts()}")
    print(f"    Tamaño del dataset original: {len(y_imbalanced)} vs. tamaño resampleado: {len(y_resampled_smoteenn)}")
    print("  Nota: Las técnicas híbridas como SMOTEENN buscan un equilibrio óptimo entre las clases,")
    print("  a menudo eliminando ejemplos ruidosos después del sobremuestreo.")

    print("\n  Estas técnicas están disponibles en bibliotecas de Python como **scikit-learn** y **imbalanced-learn**.\n")

except Exception as e:
    print(f"  Error en Ejercicio Pág 13: {e}")


# --- Página 14: Ejercicio Guiado - Requisitos e Implementación en Scikit-Learn ---

# Título del tema: Ejercicio Guiado
# Requisitos e Implementación en Scikit-Learn
# Explicación del contenido:
# Esta sección detalla los **requisitos y los pasos para un ejercicio guiado completo**
# centrado en la aplicación y evaluación de las diversas técnicas de balanceo de datos.
# Los pasos principales del ejercicio son:
# 1. **Importar librerías** necesarias (como `numpy`, `pandas`, `sklearn`, `imblearn`).
# 2. **Cargar los Datos** (o en este caso, generarlos para simulación).
# 3. **Generación de datos desbalanceados**.
# 4. **División de datos** en conjuntos de entrenamiento y prueba.
# 5. **Aplicación de las técnicas de balanceo**: Esto incluye Submuestreo (Undersampling),
#    Sobremuestreo (Oversampling), Combinación de Submuestreo y Sobremuestreo (SMOTEENN),
#    y Generación de Muestras Sintéticas (SMOTE).
# 6. **Entrenamiento y evaluación del modelo** para cada técnica de balanceo aplicada.
# Se menciona que el detalle completo de esta actividad se encuentra en la guía de estudio de la sesión.
# La implementación se realiza utilizando las bibliotecas `scikit-learn` e `imbalanced-learn`.

# Ejercicio Demostrativo 14: Estructura de un ejercicio completo de balanceo de datos.
# Este ejercicio sigue los pasos indicados en la diapositiva, aplicando y evaluando las técnicas
# de balanceo en un contexto más completo para ver su impacto en el rendimiento del modelo.

print("\n" + "="*80)
print("--- Contenido de la Página 14: Ejercicio Guiado de Balanceo de Datos ---")
print("="*80 + "\n")

try:
    print("[Ejercicio Pág 14] Estructura de un Ejercicio Guiado de Balanceo de Datos.")
    print("  Este ejercicio integra los pasos para aplicar y evaluar técnicas de balanceo de principio a fin.\n")

    # 1. Importar librerías (ya realizado al inicio de este script).
    print("  1. Librerías necesarias (numpy, pandas, sklearn, imblearn) ya importadas al inicio del script.")

    # 2. Cargar los Datos / 3. Generación de datos desbalanceados.
    #    Reutilizamos el dataset desbalanceado generado previamente en la Pág 11.
    print("\n  2. y 3. Datos desbalanceados generados para demostración (como en Pág 11):")
    print(f"    Clase 0 (mayoritaria): {np.sum(y_imbalanced == 0)} muestras")
    print(f"    Clase 1 (minoritaria): {np.sum(y_imbalanced == 1)} muestras")

    # 4. División de datos (Train-Test Split).
    #    `stratify=y_imbalanced` asegura que la proporción de clases se mantenga en los conjuntos de entrenamiento y prueba.
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_imbalanced, y_imbalanced, test_size=0.3, random_state=42, stratify=y_imbalanced
    )
    print("\n  4. Datos divididos en conjuntos de entrenamiento y prueba (estratificados).")
    print(f"    Clases en el conjunto de entrenamiento: {pd.Series(y_train_bal).value_counts()}")
    print(f"    Clases en el conjunto de prueba: {pd.Series(y_test_bal).value_counts()}")

    # Inicializar un modelo base para todas las evaluaciones (ej. Regresión Logística).
    model_instance = LogisticRegression(solver='liblinear', random_state=42)

    # Diccionario para almacenar los resultados de las métricas para cada técnica de balanceo.
    results = {}

    # Función auxiliar para entrenar, predecir y evaluar un modelo.
    # Se enfoca en el `recall` y `precision` de la clase minoritaria, que son críticas en datasets desbalanceados.
    def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, model_obj):
        model_obj.fit(X_train, y_train) # Entrena el modelo con los datos balanceados (o no balanceados).
        y_pred = model_obj.predict(X_test) # Realiza predicciones en el conjunto de prueba.
        # Calcula métricas de evaluación. `pos_label=1` especifica que la clase minoritaria es 1.
        acc = accuracy_score(y_test, y_pred)
        rec_minority = recall_score(y_test, y_pred, pos_label=1)
        prec_minority = precision_score(y_test, y_pred, pos_label=1, zero_division=0) # zero_division=0 evita warnings si no hay predicciones positivas.
        f1_minority = f1_score(y_test, y_pred, pos_label=1)
        return {"accuracy": acc, "recall_minority": rec_minority, "precision_minority": prec_minority, "f1_minority": f1_minority}

    # Evaluación del modelo sin aplicar ninguna técnica de balanceo (línea base).
    print("\n  6. Evaluación del modelo sin balanceo (línea base):")
    results['No Balanceo'] = train_and_evaluate(X_train_bal, y_train_bal, X_test_bal, y_test_bal, 'No Balanceo', model_instance)
    print(f"    Sin Balanceo - Precisión (General): {results['No Balanceo']['accuracy']:.4f}")
    print(f"    Sin Balanceo - Recall (Clase Minoritaria): {results['No Balanceo']['recall_minority']:.4f}")
    print(f"    Sin Balanceo - Precision (Clase Minoritaria): {results['No Balanceo']['precision_minority']:.4f}")
    print(f"    Sin Balanceo - F1-Score (Clase Minoritaria): {results['No Balanceo']['f1_minority']:.4f}")

    # 5. Aplicación de las técnicas de balanceo y 6. Entrenamiento y evaluación para cada una.
    print("\n  5. y 6. Aplicación de técnicas de balanceo y evaluación comparativa:")

    # a. Submuestreo (RandomUnderSampler).
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_bal, y_train_bal)
    results['Submuestreo (RUS)'] = train_and_evaluate(X_train_rus, y_train_rus, X_test_bal, y_test_bal, 'Submuestreo (RUS)', model_instance)
    print(f"\n    Submuestreo (RUS) - Recall (Min): {results['Submuestreo (RUS)']['recall_minority']:.4f}, Precision (Min): {results['Submuestreo (RUS)']['precision_minority']:.4f}")

    # b. Sobremuestreo (RandomOverSampler).
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_bal, y_train_bal)
    results['Sobremuestreo (ROS)'] = train_and_evaluate(X_train_ros, y_train_ros, X_test_bal, y_test_bal, 'Sobremuestreo (ROS)', model_instance)
    print(f"    Sobremuestreo (ROS) - Recall (Min): {results['Sobremuestreo (ROS)']['recall_minority']:.4f}, Precision (Min): {results['Sobremuestreo (ROS)']['precision_minority']:.4f}")

    # c. SMOTE (Generación de Muestras Sintéticas).
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_bal, y_train_bal)
    results['SMOTE'] = train_and_evaluate(X_train_smote, y_train_smote, X_test_bal, y_test_bal, 'SMOTE', model_instance)
    print(f"    SMOTE - Recall (Min): {results['SMOTE']['recall_minority']:.4f}, Precision (Min): {results['SMOTE']['precision_minority']:.4f}")

    # d. SMOTEENN (Combinación de Submuestreo y Sobremuestreo).
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_bal, y_train_bal)
    results['SMOTEENN'] = train_and_evaluate(X_train_smoteenn, y_train_smoteenn, X_test_bal, y_test_bal, 'SMOTEENN', model_instance)
    print(f"    SMOTEENN - Recall (Min): {results['SMOTEENN']['recall_minority']:.4f}, Precision (Min): {results['SMOTEENN']['precision_minority']:.4f}")

    print("\n  **Conclusión del Ejercicio Guiado:**")
    print("  Este ejercicio demuestra cómo las diferentes técnicas de balanceo impactan en las métricas")
    print("  del modelo, especialmente el **Recall de la Clase Minoritaria**, que es crucial en datasets desbalanceados.")
    print("  A menudo, el balanceo de datos mejora la capacidad del modelo para identificar correctamente")
    print("  las instancias de la clase minoritaria.")

except Exception as e:
    print(f"  Error en Ejercicio Pág 14: {e}")


# --- Página 15: Preguntas / Máquina Supervisado Aprendizaje de ---

# Título del tema: Preguntas
# Explicación del contenido:
# Esta página parece ser una sección de cierre o una invitación a una sesión de preguntas y respuestas
# para la temática de "Máquina Supervisado - Aprendizaje de" y los mecanismos para mejorar el desempeño
# de los algoritmos. Es una transición para continuar con actividades o para la discusión final.

# Ejercicio Demostrativo 15: Conclusión y resumen de los temas cubiertos.

print("\n" + "="*80)
print("--- Contenido de la Página 15: Preguntas y Conclusión ---")
print("="*80 + "\n")

print("[Ejercicio Pág 15] Fin de la Sesión: Preguntas y Reflexión.")
print("  Hemos recorrido los mecanismos fundamentales para mejorar el desempeño de los algoritmos de Machine Learning:")
print("  - **Ingeniería de Características (Feature Engineering)**: Transformar y crear variables para mejorar el modelo.")
print("  - **Optimización de Hiperparámetros**: Ajustar configuraciones del algoritmo para un rendimiento óptimo.")
print("  - **Regularización**: Prevenir el sobreajuste penalizando la complejidad del modelo (L1, L2, Elastic Net).")
print("  - **Balanceo de Datos**: Manejar el desequilibrio de clases para mejorar la predicción de minorías (Undersampling, Oversampling, SMOTE, Híbridos).")
print("\n  Se invita a continuar con las actividades propuestas para profundizar su comprensión")
print("  y aplicar estos conceptos en proyectos prácticos.\n")