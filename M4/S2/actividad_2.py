#-------------------------------------
#------------- Actividad 2 -----------
#-------------------------------------
# Enunciado:
# Una urna contiene 5 bolas rojas, 3 bolas azules y 2 bolas verdes.
# Se extrae una bola al azar de la urna.
# Calcula las siguientes probabilidades:
# 1. ¿Cuál es la probabilidad de que la bola extraída sea roja?
# 2. ¿Cuál es la probabilidad de que la bola extraída sea azul?
# 3. ¿Cuál es la probabilidad de que la bola extraída sea verde?
# 4. ¿Cuál es la probabilidad de que la bola extraída NO sea roja?
# 5. ¿Cuál es la probabilidad de que la bola extraída sea roja o azul?

# Probabilidad en la extracción de una bola de una urna

# Espacio muestral: 5 rojas, 3 azules, 2 verdes
total_bolas = 5 + 3 + 2

# 1. Probabilidad de que sea roja
casos_roja = 5

# 2. Probabilidad de que sea azul
casos_azul = 3

# 3. Probabilidad de que sea verde
casos_verde = 2

# 4. Probabilidad de que NO sea roja
casos_no_roja = total_bolas - casos_roja


# 5. Probabilidad de que sea roja o azul
casos_roja_azul = casos_roja + casos_azul