"""
HW-1 list functions. 

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

import random


def list_stats(values:list):
    """Returns the min, max, average, and sum of the values in the given
    list as a tuple (min, max, avg, sum).
      
    Args:
        values: The list of values to compute statistics over.

    Notes:
        Tuple (None, None, None, None) returned if values is empty.
        Assumes a list of numerical values. 

    Example: 
        >>> list_stats([1, 2, 3])
        (1, 3, 2.0, 6)

    """
    # Type checking
    if not isinstance(values, list):
        raise TypeError('values must be a list')
    
    if len(values) == 0:
        return (None, None, None, None)
    else:
        return (min(values), max(values), sum(values)/len(values), sum(values))


def convert_numeric(value:str):
    """Returns corresponding numeric value for given string value.

    Args:
        value: The string value to convert.

    Notes:
        Given value returned if cannot be converted to int or float.

    Examples:
        >>> convert_numeric('abc')
        'abc'
        >>> convert_numeric('42')
        42
        >>> convert_numeric('3.14')
        3.14

    """
    # Type checking
    if value.isdigit():
        return int(value)
    
    try:
        return float(value)
    except: # throw
        return value


def random_matrix_for(m, n):
    """Return an m x n matrix as a list of lists containing randomly
    generated integer values.

    Args:
        m: The number of rows. 
        n: The number of columns.

    Notes:
        Values are from 0 up to but not including m*n.
    
    Example:
        >>> random_matrix_for(2, 3)
        [[2, 1, 0], [3, 6, 4]]

    """
    # Type checking
    if m < 1 or n < 1:
        raise ValueError('m and n must be greater than 0')
    
    matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(random.randint(0, m*n-1))
        matrix.append(row)
    
    return matrix


def random_matrix_comp(m, n):
    """Return an m x n matrix as a list of lists containing randomly
    generated integer values.

    Args:
        m: The number of rows. 
        n: The number of columns.

    Notes:
        Values are from 0 up to but not including m*n.
    
    Example:
        >>> random_matrix_for(2, 3)
        [[2, 1, 0], [3, 6, 4]]

    """
    # Type checking
    if m < 1 or n < 1:
        raise ValueError('m and n must be greater than 0')
    
    return [[random.randint(0, m*n-1) for j in range(n)] for i in range(m)]


def transpose_matrix(list_matrix): 
    """Return the transpose of the given matrix represented as a list of
    lists.

    Args:
        list_matrix: The list version of the matrix to transpose.

    Example: 
        >>> transpose_matrix([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]

    """
    # Type checking
    if not isinstance(list_matrix, list) or not isinstance(list_matrix[0], list):
        raise TypeError('list_matrix must be a nested list')
    if len(list_matrix) < 1 or len(list_matrix[0]) < 1:
        raise ValueError('list_matrix must be non-empty')
    
    original_row = len(list_matrix)
    original_col = len(list_matrix[0])
    new_matrix = []

    for i in range(original_col):
        row = []
        for j in range(original_row):
            row.append(list_matrix[j][i])
        new_matrix.append(row)
    
    return new_matrix


def reshape_matrix(list_matrix, m, n):
    """Return a new matrix based on the given matrix but scaled to m rows
    and n columns.

    Args:
        list_matrix: The matrix to reshape.
        m: The new number of rows.
        n: The new number of columns.

    Notes:
        New rows or columns are filled with 0 values.

    Example: 
        >>> reshape_matrix([[1, 2, 3], [4, 5, 6]], 3, 2)
        [[1, 2], [4, 5], [0, 0]]

    """
    # Type checking
    if not isinstance(list_matrix, list) or not isinstance(list_matrix[0], list):
        raise TypeError('list_matrix must be a nested list')
    if len(list_matrix) < 1 or len(list_matrix[0]) < 1:
        raise ValueError('list_matrix must be non-empty')
    if m < 1 or n < 1:
        raise ValueError('m and n must be greater than 0')
    
    original_row = len(list_matrix)
    original_col = len(list_matrix[0])
    new_matrix = []

    for i in range(m):
        row = []
        for j in range(n):
            if i < original_row and j < original_col:
                row.append(list_matrix[i][j])
            else:
                row.append(0)
        new_matrix.append(row)
    
    return new_matrix

