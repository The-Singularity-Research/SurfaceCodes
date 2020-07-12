#

def permlist_to_tuple(perms):
    """
    convert list of lists to tuple of tuples in order to have two level iterables
    that are hashable for the dictionaries used later
    """
    return tuple(tuple(perm) for perm in perms)



def rowSwap(A, i, j):
    temp = numpy.copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp


def colSwap(A, i, j):
    temp = numpy.copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp


def scaleCol(A, i, c):
    A[:, i] *= int(c) * numpy.ones(A.shape[0], dtype=np.int64)


def scaleRow(A, i, c):
    A[i, :] = np.array(A[i, :], dtype=numpy.float64) * c * numpy.ones(A.shape[1], dtype=numpy.float64)


def colCombine(A, addTo, scaleCol, scaleAmt):
    A[:, addTo] += scaleAmt * A[:, scaleCol]


def rowCombine(A, addTo, scaleRow, scaleAmt):
    A[addTo, :] += scaleAmt * A[scaleRow, :]