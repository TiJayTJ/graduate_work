import numpy as np

from matrix import mix_block_matrix, make_block_matrix, remake_block_matrix


def mix_matrix(matrix):
    n = len(matrix)     # кол-во столбцов матрицы
    m = len(matrix[0])      # кол-во строк матрицы
    h = len(matrix[0][0])  # глубина матрицы
    row_half = n // 2
    mixed_matrix = np.empty((n, m, h), dtype=object)

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
    unmixed_matrix = np.empty((n, m, h), dtype=object)

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

block_matrix = make_block_matrix(matrix, 2)
remaked_block_matrix = remake_block_matrix(block_matrix)

for i in range(len(remaked_block_matrix)):
    for j in range(len(remaked_block_matrix[0])):
        print(remaked_block_matrix[i][j][0], end=' ')
    print()

# mixed_block_matrix = mix_block_matrix(block_matrix)

# for i in range(len(mixed_block_matrix)):
#     for j in range(len(mixed_block_matrix[0])):
#         mixed_block_matrix[i][j] = mix_matrix(mixed_block_matrix[i][j])
#
# for i in range(len(mixed_block_matrix)):
#     for j in range(len(mixed_block_matrix[0])):
#         for k in range(len(mixed_block_matrix[0][0])):
#             for f in range(len(mixed_block_matrix[0][0][0])):
#                 print(mixed_block_matrix[i][j][k][f][0], end=' ')
#             print()
#         print('\n\n')

# mixed_matrix = mix_block_matrix(matrix)
#
# for k in range(h):
#     for i in range(n):
#         for j in range(m):
#             print(mixed_matrix[i][j][k], end=' ')
#         print()
#     print('\n\n')
#
#
# unmixed_matrix = unmix_matrix(mixed_matrix)
#
# for k in range(h):
#     for i in range(n):
#         for j in range(m):
#             print(unmixed_matrix[i][j][k], end=' ')
#         print()
#     print('\n\n')