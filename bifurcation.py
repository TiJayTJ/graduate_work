import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Определение системы уравнений
def system(t, state, a_param):
    x, y, z = state
    dxdt = y - a_param * x + y * z
    dydt = 30 * x - 3 * y - x * z
    dzdt = -3.5 * z + x * y + x**2
    return [dxdt, dydt, dzdt]

# Диапазон параметров a
b_values = np.linspace(0, 50, 500)
initial_state = [1.0, 1.0, 1.0]  # Начальные условия

# Построение графика для параметра a
bifurcation_data = []
for b_param in b_values:
    print(b_param)
    # Решение системы
    solution = solve_ivp(
        system, [0, 200], initial_state, args=(b_param,), t_eval=np.linspace(100, 200, 1000)
    )
    x_values, y_values, z_values = solution.y

    # Учитываем только последние значения для устойчивого состояния
    bifurcation_data.extend((b_param, abs(y)) for y in y_values[-100:])

# Построение графика
param_data, y_data = zip(*bifurcation_data)
plt.figure(figsize=(10, 6))
plt.scatter(param_data, y_data, s=0.1, color='blue')
plt.title('Bifurcation Diagram for Parameter b')
plt.xlabel('b')
plt.ylabel('|y| where x=y')
plt.grid()
plt.show()
