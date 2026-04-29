#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def determinant(matrix):
    """
    Helper function to calculate the determinant of a matrix
    """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def adjugate(matrix):
    """
    Helper function to calculate the adjugate matrix of a matrix
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:] for row in
                          (matrix[:i] + matrix[i + 1:])]
            cofactor_val = ((-1) ** (i + j)) * determinant(sub_matrix)
            adj_matrix[j][i] = cofactor_val
    return adj_matrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    adj = adjugate(matrix)
    inv_matrix = [[element / det for element in row] for row in adj]

    return inv_matrix
