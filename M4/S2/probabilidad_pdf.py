# Fórmula general de probabilidad clásica para un evento:
# P(A) = Número de casos favorables / Número total de casos posibles
#
# Donde:
# - P(A) es la probabilidad de que ocurra el evento A.
# - "Casos favorables" es la cantidad de resultados que cumplen la condición del evento.
# - "Casos posibles" es el total de resultados posibles en el experimento (para un dado de 6 caras, es 6).
#
# Ejemplo aplicado:
# - Probabilidad de sacar un número mayor a 4:
#   P(mayor a 4) = 2/6 (los números 5 y 6 son favorables)
# - Probabilidad de sacar un número impar:
#   P(impar) = 3/6 (los números 1, 3, 5 son favorables)

# El espacio muestral (S) es el conjunto de todos los posibles resultados de un experimento aleatorio.
# Para un dado de seis caras:
espacio_muestral = [1, 2, 3, 4, 5, 6]
print(f"Espacio muestral (S): {espacio_muestral}")
print(f"Número total de casos posibles: {len(espacio_muestral)}")

#-------------------------------------
#------------- Ejemplo ---------------
#-------------------------------------
# Probabilidad en el lanzamiento de un dado de seis caras

# Espacio muestral: {1, 2, 3, 4, 5, 6}
total_casos = 6

# 1. Probabilidad de que salga un número mayor a 4 (es decir, 5 o 6)
casos_mayor_4 = 2  # {5, 6}
p_mayor_4 = casos_mayor_4 / total_casos
print(f"Probabilidad de número mayor a 4: P = {casos_mayor_4}/{total_casos} = {p_mayor_4:.2f}")

# 2. Probabilidad de que salga un número mayor a 5 (solo 6)
casos_mayor_5 = 1  # {6}
p_mayor_5 = casos_mayor_5 / total_casos
print(f"Probabilidad de número mayor a 5: P = {casos_mayor_5}/{total_casos} = {p_mayor_5:.2f}")

# 3. Probabilidad de que salga número impar (1, 3, 5)
casos_impar = 3  # {1, 3, 5}
p_impar = casos_impar / total_casos
print(f"Probabilidad de número impar: P = {casos_impar}/{total_casos} = {p_impar:.2f}")

# 4. Probabilidad de que salga número par (2, 4, 6)
casos_par = 3  # {2, 4, 6}
p_par = casos_par / total_casos
print(f"Probabilidad de número par: P = {casos_par}/{total_casos} = {p_par:.2f}")

# 5. Probabilidad de que salga 1
casos_1 = 1  # {1}
p_1 = casos_1 / total_casos
print(f"Probabilidad de que salga 1: P = {casos_1}/{total_casos} = {p_1:.2f}")

# 6. Probabilidad de que salga 6
casos_6 = 1  # {6}
p_6 = casos_6 / total_casos
print(f"Probabilidad de que salga 6: P = {casos_6}/{total_casos} = {p_6:.2f}")