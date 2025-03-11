import numpy as np

from matrix import mix_matrix, unmix_matrix

matrix = np.empty((8, 8, 3), dtype=object)
n = len(matrix)     # кол-во столбцов матрицы
m = len(matrix[0])      # кол-во строк матрицы
h = len(matrix[0][0])      # глубина матрицы

count = 1
for i in range(n):
    for j in range(m):
        matrix[i][j][0] = str(count) + 'green'
        matrix[i][j][1] = str(count) + 'red'
        matrix[i][j][2] = str(count) + 'yellow'
        count += 1

mixed_matrix = mix_matrix(matrix)

for k in range(h):
    for i in range(n):
        for j in range(m):
            print(mixed_matrix[i][j][k], end=' ')
        print()
    print('\n\n')


unmixed_matrix = unmix_matrix(mixed_matrix)

for k in range(h):
    for i in range(n):
        for j in range(m):
            print(unmixed_matrix[i][j][k], end=' ')
        print()
    print('\n\n')