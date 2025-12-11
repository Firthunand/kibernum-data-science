#-------------------------------------
#------------- Actividad 1 -----------
#-------------------------------------
# Enunciado:
# Supón que lanzas una moneda al aire. La moneda puede caer mostrando "cara" o "sello".
# Calcula las siguientes probabilidades:
# 1. ¿Cuál es la probabilidad de que salga "cara"?
# 2. ¿Cuál es la probabilidad de que salga "sello"?
# 3. ¿Cuál es la probabilidad de que salga cualquier resultado posible (evento seguro)?
# 4. ¿Cuál es la probabilidad de que salga un resultado imposible (por ejemplo, que la moneda caiga de canto)?

# Probabilidad en el lanzamiento de una moneda

# Espacio muestral: {"cara", "sello"}
espacio_muestral_moneda = ["cara", "sello"]
total_casos_moneda = len(espacio_muestral_moneda)

# 1. Probabilidad de que salga "cara"
casos_cara = 1

# 2. Probabilidad de que salga "sello"
caso_sello = 1

# 3. Probabilidad de que salga cualquier resultado (evento seguro)
caso_evento_seguro = total_casos_moneda

# 4. Probabilidad de que salga un resultado imposible (por ejemplo, "canto")
caso_imposible = 0