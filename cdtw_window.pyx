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

#def compute_similarity(np.ndarray[DTYPE_t,ndim=2] test_set,\
#        np.ndarray[DTYPE_t,ndim=2] train_set):
#    cdef unsigned int test_len    = test_set.shape[0]
#    cdef unsigned int train_len   = train_set.shape[0]
#    cdef unsigned int window_len  = train_set.shape[1]
#    cdef unsigned int num_windows = test_set.shape[1] - train_set.shape[1] + 1
#    cdef np.ndarray[DTYPE_t, ndim = 3] ref_tr_dtw = \
#            np.zeros([num_windows, test_len, train_len], dtype=DTYPE)
#
#    for i3 in range(test_len):
#        for i2 in range(train_len):
#            x = test_set[i3,:] 
#            y = train_set[i2,:]
#            for i1 in range(num_windows):
#                xi = x[i1:i1+window_len]
#                d = cdtw.dtw(xi,y)
#                ref_tr_dtw[i1,i3,i2] = d
#    return ref_tr_dtw

    
def dtw_dists(np.ndarray[DTYPE_t,ndim=3] test_set,\
        np.ndarray[DTYPE_t,ndim=2] train_set):


    cdef unsigned int num_windows  = test_set.shape[0]
    cdef unsigned int num_test     = test_set.shape[1]
    cdef unsigned int num_samples  = test_set.shape[2]
    cdef unsigned int num_train    = train_set.shape[0]

    cdef np.ndarray[DTYPE_t, ndim = 3] D = np.zeros([num_windows,\
            num_test,num_train], dtype=DTYPE)

    cdef unsigned int _i
    for _i in range(num_train):
        train_set[<unsigned int>_i,:] = _mpt_scale(train_set[<unsigned int>_i,])

    cdef unsigned int i3,i2,j1
    for i3 in range(num_windows):
        for i2 in range(num_test):
            for j1 in range(num_train): 
                D[<unsigned int>i3,<unsigned int>i2,<unsigned int>j1] =\
                        cdtw.dtw_unbound(test_set[<unsigned int>i3,<unsigned int>i2,]\
                        ,train_set[<unsigned int>j1,])[0]
#                D[<unsigned int>i3,<unsigned int>i2,<unsigned int>j1] =\
#                        cdtw.dtw_bound(test_set[<unsigned int>i3,<unsigned int>i2,]\
#                        ,train_set[<unsigned int>j1,],mp=.1)[0]

    return D




#def compute_similarity(np.ndarray[DTYPE_t,ndim=3] test_set,\
#        np.ndarray[DTYPE_t,ndim=2] train_set):
#
#    cdef unsigned int test_len    = test_set.shape[0]
#    cdef unsigned int train_len   = train_set.shape[0]
#    cdef unsigned int window_len  = train_set.shape[1]
#    cdef unsigned int num_windows = test_set.shape[1] - train_set.shape[1] + 1
#    cdef np.ndarray[DTYPE_t, ndim = 3] ref_tr_dtw = \
#            np.zeros([num_windows, test_len, train_len], dtype=DTYPE)
#
#    for i2 in range(train_len):
#        train_set[i2,:] = _mpt_scale(train_set[i2,:])
#        
#    
#    for i3 in range(test_len):
#        for i2 in range(train_len):
#            x = test_set[i3,:] 
#            y = _mpt_scale(train_set[i2,:])
#            for i1 in range(num_windows):
#                xi = x[i1:i1+window_len]
#                d = cdtw.dtw(xi,y)
#                ref_tr_dtw[i1,i3,i2] = d
#    return ref_tr_dtw
    

def _mpt_scale(x):
    x = abs(x)
    cdef int i
    cdef int xlen = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] x2 = np.zeros(xlen/2)
    for i in range(0,xlen/2):
        x2[<unsigned int> i] = x[<unsigned int> i]
    cdef float scalar = np.mean(x2)
    return (x/scalar)


