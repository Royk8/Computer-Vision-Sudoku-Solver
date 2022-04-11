#import numpy as np
from numpy import zeros
from datetime import datetime


def solve_sudoku(original, sudoku, end=False, pos=(0, 0)):
    """
        Algoritmo recursivo que recibe el sudoku original, una copia de este que se modificará y una posicion
        desde donde continuar el proceso de evaluacion del sudoku
    :param original: Numpy Array con la version original del sudoku sin resolver
    :param sudoku: Numpy Array con el sudoku en su estado actual de solución
    :param pos: Tupla con la posicion actual que esta siendo evaluada en el algoritmo
    :return: Sudoku resuelto.
    """
    if pos == (9, 9):
        if valid_pos(sudoku, (8, 8)):
            end = True
            print(sudoku)
        return sudoku
    if original[pos] == 0:
        for i in range(10):
            sudoku[pos] = i
            if valid_pos(sudoku, pos):
                solve_sudoku(original, sudoku, end, next_pos(pos))
                if end:
                    return sudoku
        else:
            sudoku[pos] = 0
    else:
        solve_sudoku(original, sudoku, end, next_pos(pos))
    return sudoku


def next_pos(pos):
    """
        Encuentra la siguiente posicion despues de la enviada como parametro
    :param pos: Tupla con la posicion actual a evaluar
    :return: Tupla con la siguiente posicion a evaluar
    """
    if pos[1] == 8:
        if pos[0] == 8:
            return 9, 9
        return pos[0] + 1, 0
    return pos[0], pos[1] + 1


def valid_pos(sudoku, pos):
    """
        Recibe el sudoku y las coordenadas de la posicion con el fin de evaluar si es el dato es valido
    :param sudoku: Numpy Array con el sudoku
    :param pos: Tupla con las dos coordenadas i y j.
    :return: Verdadero si el dato no no incumple ninguna de las normas de juego del sudoku
    """
    if sudoku[pos] < 1 or sudoku[pos] > 9:
        return False
    for i in range(9):
        if sudoku[i,pos[1]] == sudoku[pos]:
            if (i,pos[1]) == pos:
                pass
            else:
                return False
        if sudoku[pos[0], i] == sudoku[pos]:
            if (pos[0], i) == pos:
                continue
            else:
                return False
    c0 = (pos[0] // 3) *3
    c1 = (pos[1] // 3) *3
    for i in range(3):
        for j in range(3):
            if sudoku[pos] == sudoku[c0 + i, c1 + j]:
                if (c0 + i, c1 + j) == pos:
                    continue
                else:
                    return False
    return True


def read_sudoku():
    """ Lee el sudoku de un documento .txt almacenado en la raiz del proyecto
        :return: Devuelve el sudoku como una matriz de 9x9 en formato array de numpy
    """
    file = open('../SudokuSolver/sudoku.txt', 'r')
    sudoku = zeros((9,9))
    for i in range(9):
        s = file.readline()
        for j in range(9):
            sudoku[i, j] = int(s[j])
    return sudoku


def resolve_sudoku(sudoku):
    solved = solve_sudoku(sudoku, sudoku.copy())

if __name__ == '__main__':
    sudoku = read_sudoku()
    print(sudoku)
    start_time = datetime.now()
    print(start_time)
    end = False
    solved = solve_sudoku(sudoku, sudoku.copy())
    end_time = datetime.now()
    duration = end_time - start_time
    print(solved)
    print(end_time)
    print(duration)
