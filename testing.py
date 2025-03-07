import numpy as np


def mix_matrix(matrix):
    n = len(matrix)     # кол-во столбцов матрицы
    m = len(matrix[0])      # кол-во строк матрицы
    h = len(matrix[0][0])  # глубина матрицы
    row_half = n // 2
    mixed_matrix = np.empty((n, m, h), dtype=np.uint8)

    for k in range(h):
        counter = 0
        for i in range(row_half):
            for j in range(i, m - i - 1):
                mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n][k] = matrix[i][j][k]
                mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n + 1][k] = matrix[j][n - i - 1][(k + 1) % 3]
                mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n + 1][k] = matrix[n - i - 1][n - j - 1][(k + 2) % 3]
                mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n][k] = matrix[n - j - 1][i][k]
                counter += 1
    return mixed_matrix


def unmix_matrix(mixed_matrix):
    n = len(mixed_matrix)     # кол-во столбцов матрицы
    m = len(mixed_matrix[0])      # кол-во строк матрицы
    h = len(mixed_matrix[0][0])      # глубина матрицы
    row_half = n // 2
    unmixed_matrix = np.empty((n, m, h), dtype=np.uint8)

    for k in range(h):
        counter = 0
        for i in range(row_half):
            for j in range(i, m - i - 1):
                unmixed_matrix[i][j][k] = mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n][k]
                unmixed_matrix[j][n - i - 1][(k + 1) % 3] = mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n + 1][k]
                unmixed_matrix[n - i - 1][n - j - 1][(k + 2) % 3] = mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n + 1][k]
                unmixed_matrix[n - j - 1][i][k] = mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n][k]
                counter += 1
    return unmixed_matrix