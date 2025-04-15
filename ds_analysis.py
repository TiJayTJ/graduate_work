import numpy as np

from system import calc_equilibrium_points, main_system_without_t, main_jacobian

equilibrium_point = calc_equilibrium_points(main_system_without_t)
for state in equilibrium_point:
    J = main_jacobian(state)
    for i in range(3):
        print(*J[i])
    values = np.linalg.eigvals(main_jacobian(state))
    print('Состояние равновесия: ', state)
    for i, val in enumerate(values):
        print(f"λ_{i} = {val}")