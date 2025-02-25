# Функция преобразования 2-битных пар в ДНК основания
import numpy as np


def bits_to_dna(bits):
    mapping = {"00": "A", "01": "G", "10": "C", "11": "T"}
    return mapping[bits]


# Функция преобразования ДНК основания в 2-битных пар
def dna_to_bits(dna):
    mapping = {"A": "00", "G": "01", "C": "10", "T": "11"}
    return mapping[dna]


# Функция разбиения байта (8 бит) на 4 пары битов и их преобразование в ДНК
def byte_to_dna(byte):
    binary_str = format(byte, "08b")  # Преобразуем байт в 8-битную строку
    return [bits_to_dna(binary_str[i:i+2]) for i in range(0, 8, 2)]


def dna_to_byte(dna):
    binary_str = "".join([dna_to_bits(base) for base in dna])  # Соединяем 4 пары в 8-битное число
    return int(binary_str, 2)  # Переводим в десятичное значение


# Функция сложения оснований ДНК
def dna_add(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
        ("G", "G"): "C", ("G", "C"): "T", ("G", "T"): "A",
        ("C", "C"): "A", ("C", "T"): "G",
        ("T", "T"): "C"
    }
    return [addition_rules.get((base1[i], base2[i])) or addition_rules.get((base2[i], base1[i])) for i in range(4)]


# Функция xor оснований ДНК
def dna_xor(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
        ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
        ("C", "C"): "A", ("C", "T"): "G",
        ("T", "T"): "A"
    }
    return [addition_rules.get((base1[i], base2[i])) or addition_rules.get((base2[i], base1[i])) for i in range(4)]


# Функция вычитания оснований ДНК
def dna_sub(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "T", ("A", "C"): "C", ("A", "T"): "G",
        ("G", "A"): "G", ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
        ("C", "A"): "C", ("C", "G"): "G", ("C", "C"): "A", ("C", "T"): "T",
        ("T", "A"): "T", ("T", "G"): "C", ("T", "C"): "G", ("T", "T"): "A"
    }
    return [addition_rules.get((base1[i], base2[i])) for i in range(4)]
