import numpy as np


def array_to_matrix(array, n, m):
    matrix = np.empty((n, m), dtype=np.uint8)
    for i in range(n*m):
        matrix[i // m][i % m] = array[i]
    return matrix


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
                counter_2_div_n = ((counter * 2) // n) * 2
                mixed_matrix[counter_2_div_n][(counter * 2) % n][k] = matrix[i][j][k]
                mixed_matrix[counter_2_div_n][(counter * 2) % n + 1][k] = matrix[j][n - i - 1][(k + 1) % 3]
                mixed_matrix[counter_2_div_n + 1][(counter * 2) % n + 1][k] = matrix[n - i - 1][n - j - 1][(k + 2) % 3]
                mixed_matrix[counter_2_div_n + 1][(counter * 2) % n][k] = matrix[n - j - 1][i][k]
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


def make_block_matrix(matrix, block_size):
    b_s = block_size
    n = len(matrix) // b_s
    m = len(matrix[0]) // b_s
    h = len(matrix[0][0])
    block_matrix = np.empty((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            block = np.empty((b_s, b_s, h), dtype=np.uint8)
            for k in range(b_s):
                for f in range(b_s):
                    block[k][f] = matrix[i * b_s + k][j * b_s + f]
            block_matrix[i][j] = block
    return block_matrix


def remake_block_matrix(block_matrix):
    block_size = len(block_matrix[0][0])
    n = len(block_matrix)
    m = len(block_matrix[0])
    h = len(block_matrix[0][0][0][0])
    matrix = np.empty((n * block_size, m * block_size, h), dtype=np.uint8)
    for i in range(n):
        for k in range(block_size):
            for j in range(m):
                for f in range(block_size):
                    matrix[i * block_size + k][j * block_size + f] = block_matrix[i][j][k][f]
    return matrix


def mix_block_matrix(matrix):
    n = len(matrix)     # кол-во столбцов матрицы
    m = len(matrix[0])      # кол-во строк матрицы
    row_half = n // 2
    mixed_matrix = np.empty((n, m), dtype=object)

    counter = 0
    for i in range(row_half):
        for j in range(i, m - i - 1):
            mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n] = matrix[i][j]
            mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n + 1] = matrix[j][n - i - 1]
            mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n + 1] = matrix[n - i - 1][n - j - 1]
            mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n] = matrix[n - j - 1][i]
            counter += 1
    return mixed_matrix


def unmix_block_matrix(mixed_matrix):
    n = len(mixed_matrix)     # кол-во столбцов матрицы
    m = len(mixed_matrix[0])      # кол-во строк матрицы
    row_half = n // 2
    unmixed_matrix = np.empty((n, m), dtype=object)

    counter = 0
    for i in range(row_half):
        for j in range(i, m - i - 1):
            unmixed_matrix[i][j] = mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n]
            unmixed_matrix[j][n - i - 1] = mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n + 1]
            unmixed_matrix[n - i - 1][n - j - 1] = mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n + 1]
            unmixed_matrix[n - j - 1][i] = mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n]
            counter += 1
    return unmixed_matrix


def mix_image(image_array):
    mixed_image = mix_matrix(image_array)
    block_matrix = make_block_matrix(mixed_image, 8)
    mixed_block_matrix = mix_block_matrix(block_matrix)
    remaked_mixed_matrix = remake_block_matrix(mixed_block_matrix)

    return remaked_mixed_matrix


def unmix_image(mixed_image):
    block_matrix = make_block_matrix(mixed_image, 8)
    unmixed_block_matrix = unmix_block_matrix(block_matrix)
    remaked_mixed_matrix = remake_block_matrix(unmixed_block_matrix)
    unmixed_image = unmix_matrix(remaked_mixed_matrix)

    return unmixed_image