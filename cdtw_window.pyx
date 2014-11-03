import numpy as np
cimport numpy as np
import cython
import cdtw

DTYPE = np.double
ctypedef np.double_t DTYPE_t

def window(x,y,num_windows):
    cdef np.ndarray[DTYPE_t, ndim = 1] D = np.zeros(num_windows, dtype=DTYPE)
    for i in range(num_windows):
        yi = y[i:i+len(x)]
        d = cdtw.dtw(x,yi)
        D[i] = d
    return D

def compute_similarity(np.ndarray[DTYPE_t,ndim=2] test_set,\
        np.ndarray[DTYPE_t,ndim=2] train_set):
    cdef unsigned int test_len    = test_set.shape[0]
    cdef unsigned int train_len   = train_set.shape[0]
    cdef unsigned int window_len  = train_set.shape[1]
    cdef unsigned int num_windows = test_set.shape[1] - train_set.shape[1] + 1
    cdef np.ndarray[DTYPE_t, ndim = 3] ref_tr_dtw = \
            np.zeros([num_windows, test_len, train_len], dtype=DTYPE)

    for i3 in range(test_len):
        for i2 in range(train_len):
            x = test_set[i3,:] 
            y = train_set[i2,:]
            for i1 in range(num_windows):
                xi = x[i1:i1+window_len]
                d = cdtw.dtw(xi,y)
                ref_tr_dtw[i1,i3,i2] = d
    return ref_tr_dtw

    

    
