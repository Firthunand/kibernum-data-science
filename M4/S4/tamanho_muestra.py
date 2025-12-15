from statsmodels.stats.power import TTestIndPower

effect_size = 0.5      # diferencia estandarizada
alpha = 0.05           # nivel de significancia
power = 0.8            # potencia deseada

analysis = TTestIndPower()
n_muestra = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
print(f"Tamaño de muestra por grupo: {n_muestra:.2f}")