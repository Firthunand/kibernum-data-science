"""
Actividad: Probabilidad y Experimentos Aleatorios - Ejemplo

REQUERIMIENTOS:

1. Distinguir entre experimento aleatorio y determinístico
   a) Medir la temperatura de una ciudad a las 12:00 pm.
   b) Encender una lámpara con un interruptor.

2. Construcción del espacio muestral
   - Escribe el espacio muestral para el experimento de lanzar dos dados de seis caras.

3. Identificación de eventos aleatorios
   a) Evento de obtener una suma igual a 7 al lanzar dos dados.
   b) Evento de obtener al menos un 6 al lanzar dos dados.

4. Cálculo de probabilidad de un evento
   - Calcula la probabilidad de obtener una suma igual a 7 al lanzar dos dados.

5. Árbol de probabilidades
   - Dibuja un árbol de probabilidades para el experimento de lanzar una moneda tres veces.
"""

# 1. Experimento aleatorio vs determinístico
# a) Medir la temperatura de una ciudad a las 12:00 pm
# Aleatorio: La temperatura puede variar cada día, no se puede predecir con certeza.
# b) Encender una lámpara con un interruptor
# Determinístico: Si el interruptor funciona, siempre que lo enciendas la lámpara se prenderá.

# 2. Espacio muestral de lanzar dos dados de seis caras
# El espacio muestral (S) es el conjunto de todos los pares posibles (dado1, dado2)
# S = {(1,1), (1,2), ..., (1,6), (2,1), ..., (6,6)}
# Hay 36 posibles resultados.

# 3. Eventos aleatorios con dos dados
# a) Evento de obtener una suma igual a 7
# Los pares que suman 7 son: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
# Evento A = {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}

# b) Evento de obtener al menos un 6
# Todos los pares donde al menos uno de los dados muestra 6:
# (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (1,6), (2,6), (3,6), (4,6), (5,6)
# Evento B = {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (1,6), (2,6), (3,6), (4,6), (5,6)}
# Total: 11 resultados

# 4. Cálculo de probabilidad de un evento
# Probabilidad de obtener una suma igual a 7 al lanzar dos dados
# Casos favorables: 6 (ver arriba)
# Casos posibles: 36
# P(A) = 6 / 36 = 1/6 ≈ 0.1667
print("Probabilidad de obtener una suma igual a 7 al lanzar dos dados: P(A) = 6/36 = 1/6 ≈ 0.1667")

# 5. Árbol de probabilidades para lanzar una moneda tres veces (forma textual y visual)

# Forma textual del árbol de probabilidades:
print("\nÁrbol de probabilidades para lanzar una moneda tres veces:")
print("Inicio")
print("├── C (0.5)")
print("│   ├── C (0.5)")
print("│   │   ├── C (0.5) → (C, C, C)  P=0.125")
print("│   │   └── X (0.5) → (C, C, X)  P=0.125")
print("│   └── X (0.5)")
print("│       ├── C (0.5) → (C, X, C)  P=0.125")
print("│       └── X (0.5) → (C, X, X)  P=0.125")
print("└── X (0.5)")
print("    ├── C (0.5)")
print("    │   ├── C (0.5) → (X, C, C)  P=0.125")
print("    │   └── X (0.5) → (X, C, X)  P=0.125")
print("    └── X (0.5)")
print("        ├── C (0.5) → (X, X, C)  P=0.125")
print("        └── X (0.5) → (X, X, X)  P=0.125")

# Explicación:
# Cada lanzamiento tiene dos opciones (Cara o Cruz) con probabilidad 0.5.
# Hay 2^3 = 8 posibles resultados, cada uno con probabilidad 0.125.

# Visualización con Matplotlib (sin networkx, solo usando anotaciones)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.axis('off')

# Coordenadas de los nodos
coords = {
    'Inicio': (0, 0),
    'C1': (-3, -1.5), 'X1': (3, -1.5),
    'CC': (-4, -3), 'CX': (-2, -3), 'XC': (2, -3), 'XX': (4, -3),
    'CCC': (-4.5, -4.5), 'CCX': (-3.5, -4.5), 'CXC': (-2.5, -4.5), 'CXX': (-1.5, -4.5),
    'XCC': (1.5, -4.5), 'XCX': (2.5, -4.5), 'XXC': (3.5, -4.5), 'XXX': (4.5, -4.5)
}

# Dibujar líneas (ramas)
plt.plot([coords['Inicio'][0], coords['C1'][0]], [coords['Inicio'][1], coords['C1'][1]], 'k-')
plt.plot([coords['Inicio'][0], coords['X1'][0]], [coords['Inicio'][1], coords['X1'][1]], 'k-')
plt.plot([coords['C1'][0], coords['CC'][0]], [coords['C1'][1], coords['CC'][1]], 'k-')
plt.plot([coords['C1'][0], coords['CX'][0]], [coords['C1'][1], coords['CX'][1]], 'k-')
plt.plot([coords['X1'][0], coords['XC'][0]], [coords['X1'][1], coords['XC'][1]], 'k-')
plt.plot([coords['X1'][0], coords['XX'][0]], [coords['X1'][1], coords['XX'][1]], 'k-')
plt.plot([coords['CC'][0], coords['CCC'][0]], [coords['CC'][1], coords['CCC'][1]], 'k-')
plt.plot([coords['CC'][0], coords['CCX'][0]], [coords['CC'][1], coords['CCX'][1]], 'k-')
plt.plot([coords['CX'][0], coords['CXC'][0]], [coords['CX'][1], coords['CXC'][1]], 'k-')
plt.plot([coords['CX'][0], coords['CXX'][0]], [coords['CX'][1], coords['CXX'][1]], 'k-')
plt.plot([coords['XC'][0], coords['XCC'][0]], [coords['XC'][1], coords['XCC'][1]], 'k-')
plt.plot([coords['XC'][0], coords['XCX'][0]], [coords['XC'][1], coords['XCX'][1]], 'k-')
plt.plot([coords['XX'][0], coords['XXC'][0]], [coords['XX'][1], coords['XXC'][1]], 'k-')
plt.plot([coords['XX'][0], coords['XXX'][0]], [coords['XX'][1], coords['XXX'][1]], 'k-')

# Etiquetas de los nodos
plt.text(*coords['Inicio'], 'Inicio', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="lightblue"))
plt.text(*coords['C1'], 'C', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['X1'], 'X', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['CC'], 'C', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['CX'], 'X', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['XC'], 'C', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['XX'], 'X', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round", fc="w"))
plt.text(*coords['CCC'], '(C,C,C)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['CCX'], '(C,C,X)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['CXC'], '(C,X,C)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['CXX'], '(C,X,X)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['XCC'], '(X,C,C)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['XCX'], '(X,C,X)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['XXC'], '(X,X,C)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
plt.text(*coords['XXX'], '(X,X,X)\n0.125', ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))

# Etiquetas de probabilidad en las ramas (solo algunas para claridad)
plt.text(-1.5, -0.7, '0.5', fontsize=10)
plt.text(1.5, -0.7, '0.5', fontsize=10)
plt.text(-3.5, -2.2, '0.5', fontsize=10)
plt.text(-2.5, -2.2, '0.5', fontsize=10)
plt.text(2.5, -2.2, '0.5', fontsize=10)
plt.text(3.5, -2.2, '0.5', fontsize=10)

plt.title('Árbol de Probabilidades: Lanzar una moneda tres veces', fontsize=14)
plt.ylim(-5, 1)
plt.xlim(-5, 5)
plt.show()