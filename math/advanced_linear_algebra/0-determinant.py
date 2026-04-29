#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns:
        The determinant of the matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        if matrix == []:
            return 1
        raise TypeError("matrix must be a list of lists")
    
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][2 - 1])

    det = 0
    for j in range(n):
        # Create the sub-matrix (minor) by removing row 0 and column j
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # Recursively add the cofactor
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det
