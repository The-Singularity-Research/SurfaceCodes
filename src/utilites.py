# Computing Homology functions
# Code from J. Kun: https://jeremykun.com/2013/04/10/computing-homology/
# Modified from SymPy rref: https://docs.sympy.org/latest/tutorial/matrices.html#rref
# for simultaneous column reduction of matrix A and row reduction of matrix B
import numpy as np

# def x_measurement(circuit: qiskit.circuit.QuantumCircuit, qubit: int, cbit: int):
#     """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
#     circuit.measure = measure  # fix a bug in qiskit.circuit.measure
#     circuit.h(qubit)
#     circuit.measure(qubit, cbit)
#     circuit.h(qubit)
#     return circuit
#
#
# def compute_phi(sigma: Tuple[Tuple[int]], alpha: Tuple[Tuple[int]]) -> List[List[int]]:
#     """compute the list of lists full cyclic form of phi (faces of dessin [sigma, alpha, phi])"""
#     s = Permutation(sigma)
#     a = Permutation(alpha)
#     f = ~(a * s)
#     f = f.full_cyclic_form
#     return f
#

def permlist_to_tuple(perms):
    """
    convert list of lists to tuple of tuples in order to have two level iterables
    that are hashable for the dictionaries used later
    """
    return tuple(tuple(perm) for perm in perms)


def rowSwap(A, i, j):
    temp = np.copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp


def colSwap(A, i, j):
    temp = np.copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp


def scaleCol(A, i, c):
    A[:, i] *= int(c) * np.ones(A.shape[0], dtype=np.int64)


def scaleRow(A, i, c):
    A[i, :] = np.array(A[i, :], dtype=np.float64) * c * np.ones(A.shape[1], dtype=np.float64)


def colCombine(A, addTo, scaleCol, scaleAmt):
    A[:, addTo] += scaleAmt * A[:, scaleCol]


def rowCombine(A, addTo, scaleRow, scaleAmt):
    A[addTo, :] += scaleAmt * A[scaleRow, :]


def simultaneousReduce(A, B):
    if A.shape[1] != B.shape[0]:
        raise Exception("Matrices have the wrong shape.")

    numRows, numCols = A.shape

    i, j = 0, 0
    while True:
        if i >= numRows or j >= numCols:
            break

        if A[i, j] == 0:
            nonzeroCol = j
            while nonzeroCol < numCols and A[i, nonzeroCol] == 0:
                nonzeroCol += 1

            if nonzeroCol == numCols:
                i += 1
                continue

            colSwap(A, j, nonzeroCol)
            rowSwap(B, j, nonzeroCol)

        pivot = A[i, j]
        scaleCol(A, j, 1.0 / pivot)
        scaleRow(B, j, 1.0 / pivot)

        for otherCol in range(0, numCols):
            if otherCol == j:
                continue
            if A[i, otherCol] != 0:
                scaleAmt = -A[i, otherCol]
                colCombine(A, otherCol, j, scaleAmt)
                rowCombine(B, j, otherCol, -scaleAmt)

        i += 1;
        j += 1

    return A%2, B%2


def finishRowReducing(B):
    numRows, numCols = B.shape

    i, j = 0, 0
    while True:
        if i >= numRows or j >= numCols:
            break

        if B[i, j] == 0:
            nonzeroRow = i
            while nonzeroRow < numRows and B[nonzeroRow, j] == 0:
                nonzeroRow += 1

            if nonzeroRow == numRows:
                j += 1
                continue

            rowSwap(B, i, nonzeroRow)

        pivot = B[i, j]
        scaleRow(B, i, 1.0 / pivot)

        for otherRow in range(0, numRows):
            if otherRow == i:
                continue
            if B[otherRow, j] != 0:
                scaleAmt = -B[otherRow, j]
                rowCombine(B, otherRow, i, scaleAmt)

        i += 1;
        j += 1

    return B%2


def numPivotCols(A):
    z = np.zeros(A.shape[0])
    return [np.all(A[:, j] == z) for j in range(A.shape[1])].count(False)


def numPivotRows(A):
    z = np.zeros(A.shape[1])
    return [np.all(A[i, :] == z) for i in range(A.shape[0])].count(False)


def bettiNumber(d_k, d_kplus1):
    A, B = np.copy(d_k), np.copy(d_kplus1)
    simultaneousReduce(A, B)
    finishRowReducing(B)

    dimKChains = A.shape[1]
    print("dim 1-chains:",dimKChains)
    kernelDim = dimKChains - numPivotCols(A)
    print("dim ker d_1:",kernelDim)
    imageDim = numPivotRows(B)
    print("dim im d_2:",imageDim)

    return "dim homology:",kernelDim - imageDim

