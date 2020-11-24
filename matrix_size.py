import enum


class MatrixSize(enum.Enum):
    small_small = 1
    small_large = 2
    large_small = 3
    large_large = 4


def get_matrix_weight(x: MatrixSize):
    if x == MatrixSize.small_small:
        return 0.4
    elif x == MatrixSize.small_large or x == MatrixSize.large_small:
        return 0.7
    else:
        return 1.0
