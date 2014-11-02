import cdtw

def window(x,y):
    num_windows = len(y) - len(x) + 1
    cdef np.ndarray[DTYPE_t, ndim = 1] D = np.zeros(num_windows, dtype=DTYPE)
    for i in range(num_windows):
        yi = y[i:i+len(x)]
        d = dtw(x,yi)
        D[i] = d
    return D
