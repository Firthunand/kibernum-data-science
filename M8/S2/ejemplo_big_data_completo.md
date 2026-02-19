# Ejemplo completo de Big Data: Comercio Electrónico

## Contexto
Una empresa de comercio electrónico quiere analizar grandes volúmenes de datos de clientes, productos y transacciones para detectar patrones de compra y recomendar productos en tiempo real.

---
### ¿Qué son las 5V’s de Big Data?.

**1. Volumen:** Se refiere a la enorme cantidad de datos generados y almacenados. Ejemplo: Facebook almacena petabytes de fotos y publicaciones de usuarios.

**2. Velocidad:** Es la rapidez con la que se generan y procesan los datos. Ejemplo: Twitter procesa millones de tuits por minuto en tiempo real.

**3. Variedad:** Diversidad de formatos y fuentes de datos (texto, imágenes, videos, sensores, etc.). Ejemplo: Un hospital almacena datos de pacientes en imágenes médicas, registros escritos y señales de monitores.

**4. Veracidad:** Calidad y confiabilidad de los datos. Ejemplo: Datos de sensores defectuosos pueden generar información errónea en una fábrica automatizada.

**5. Valor:** Utilidad que se puede extraer de los datos. Ejemplo: Un supermercado analiza los tickets de compra para identificar productos más vendidos y optimizar inventario.

---

## 1. Generación y almacenamiento de datos (simulación)

Supongamos que los datos de transacciones se generan continuamente y se almacenan en un sistema distribuido como Hadoop HDFS o una base NoSQL.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simular datos de transacciones
np.random.seed(42)
clientes = [f"C{str(i).zfill(4)}" for i in range(1, 1001)]
productos = [f"P{str(i).zfill(3)}" for i in range(1, 51)]
fechas = [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(10000)]

datos = {
    'cliente_id': np.random.choice(clientes, 10000),
    'producto_id': np.random.choice(productos, 10000),
    'fecha': fechas,
    'cantidad': np.random.randint(1, 5, 10000),
    'precio_unitario': np.random.uniform(10, 100, 10000)
}
df = pd.DataFrame(datos)
df['total'] = df['cantidad'] * df['precio_unitario']

# Guardar como CSV (simulando almacenamiento en HDFS o NoSQL)
df.to_csv('transacciones.csv', index=False)
```

---

## 2. Ingesta y procesamiento distribuido (PySpark)

Supongamos que los datos se procesan en un clúster Spark para obtener insights y recomendaciones.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

spark = SparkSession.builder.appName("EcommerceBigData").getOrCreate()
# Leer datos simulados
transacciones = spark.read.csv('transacciones.csv', header=True, inferSchema=True)

# 1. Calcular el total de ventas por producto
ventas_producto = transacciones.groupBy('producto_id').agg(spark_sum('total').alias('ventas_totales'))
ventas_producto.orderBy(col('ventas_totales').desc()).show(5)

# 2. Recomendar productos populares a un cliente
cliente = 'C0001'
productos_populares = ventas_producto.orderBy(col('ventas_totales').desc()).limit(3)
print(f"Recomendaciones para {cliente}:")
productos_populares.show()
```

---

## 3. Visualización y análisis final (Python local)

Después del procesamiento distribuido, los resultados pueden exportarse y analizarse localmente.

```python
# Leer resultados procesados (simulación)
import matplotlib.pyplot as plt
ventas = pd.read_csv('ventas_producto.csv')  # Supón que exportaste desde Spark

plt.bar(ventas['producto_id'], ventas['ventas_totales'])
plt.xlabel('Producto')
plt.ylabel('Ventas Totales')
plt.title('Ventas por Producto')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

---

## 4. Conclusión
Este ejemplo muestra el ciclo completo de Big Data:
- Generación y almacenamiento masivo de datos
- Procesamiento distribuido con Spark
- Recomendaciones en tiempo real
- Visualización y análisis final
