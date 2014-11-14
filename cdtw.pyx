""" A cython implementation of dynamic time warping (DTW).

Code is inspired from github.com/pierre-rouanet/dtw. Preliminary tests
show over 100x speed up from pure python.

author: Chris Rawles
git:    https://github.com/crawles

"""
import numpy as np
cimport numpy as np
import cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
def dtw_unbound(np.ndarray[DTYPE_t,ndim=1] x,np.ndarray[DTYPE_t, ndim=1] y):
    """ Performs an un-bounded DTW. Returns 1) cumulative distance, 2) cumulative
    distance matrix 3) x,y mapping. 
    
    Keyword arguments:
    x  -- the real part
    y  -- the imaginary part
    """

    cdef int r = x.shape[0]
    cdef int c = y.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] D = np.zeros([r + 1, c + 1], dtype=DTYPE)
    D[0, 1:] = float('inf')
    D[1:, 0] = float('inf')

    ### cost matrix ###
    cdef unsigned int inc = 0
    cdef unsigned int i,j

    for i in range(r):
        for j in range(c):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] = abs(x[i] - y[j]) 
            #D[<unsigned int>(i+1),<unsigned int>(j+1)] = square(x[i] - y[j])

    for i in range(r):
        for j in range(c):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] += \
                    min(D[i, j], D[i, j+1], D[i+1, j])

    D = D[1:, 1:]
    
    cdef float dist = D[-1, -1]

#    return dist,D
    return (dist,D)#, _trackeback(D)

@cython.boundscheck(False)
def dtw_bound(np.ndarray[DTYPE_t,ndim=1] x,np.ndarray[DTYPE_t, ndim=1] y,float mp = .1):
    """ Performs an un-bounded DTW. Returns 1) cumulative distance, 2) cumulative
    distance matrix 3) x,y mapping. 
    
    Keyword arguments:
    x  -- the real part
    y  -- the imaginary part
    mp -- percent of window for lower bound
    """
    #TODO testing
    #TODO/NOTE len(x) must == len(y)
    cdef int r = x.shape[0]
    cdef int c = y.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] D = np.zeros([r + 1, c + 1], dtype=DTYPE)
    D[0, 1:] = float('inf')
    D[1:, 0] = float('inf')

    #cost matrix
    cdef unsigned int inc = 0
    cdef unsigned int i,j
    cdef int m = int(mp*r) #TODO r == c 
    cdef unsigned int pad = 10 
    mpad = m + pad #TODO does this work?

    for i in range(mpad):
        for j in range(mpad+inc):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] = abs(x[i] - y[j])
        inc += 1
    inc = 1
    for i in range(mpad,(r-mpad)):
        for j in range(inc, (inc + (2*mpad))):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] = abs(x[i] - y[j])
        inc += 1
    inc = 0
    for i in range(r-mpad,r):
        for j in range((c - (2*mpad) + inc),c):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] = abs(x[i] - y[j])
        inc += 1

    for i in range(r):
        for j in range(c):
            D[<unsigned int>(i+1),<unsigned int>(j+1)] += \
                    min(D[i, j], D[i, j+1], D[i+1, j])


#    ### TESTING ###
#    for i in range(r):
#        for j in range(c):
#            D[<unsigned int>(i+1),<unsigned int>(j+1)] = abs(x[i] - y[j]) 
#            #D[<unsigned int>(i+1),<unsigned int>(j+1)] = square(x[i] - y[j])
#    ### TESTING ###


#   #cummulative cost matrix
#    inc = 0
#    for i in range(m):
#        for j in range(m+inc):
#            D[<unsigned int>(i+1),<unsigned int>(j+1)] += \
#                    min(D[i, j], D[i, j+1], D[i+1, j])
#        inc += 1
#    inc = 1
#    for i in range(m,(r-m)):
#        for j in range(inc, (inc + (2*m))):
#            D[<unsigned int>(i+1),<unsigned int>(j+1)] += \
#                    min(D[i, j], D[i, j+1], D[i+1, j])
#        inc += 1
#    inc = 0
#    for i in range(r-m,r):
#        for j in range((c - (2*m) + inc),c):
#            D[<unsigned int>(i+1),<unsigned int>(j+1)] += \
#                    min(D[i, j], D[i, j+1], D[i+1, j])
#        inc += 1

    D = D[1:, 1:]
    
    cdef float dist = D[-1, -1]

    return dist,D#, _trackeback(D)

cdef inline square(x): return x * x

def _trackeback(D):
    #source: github.com/pierre-rouanet/dtw
    #NOTE un-cythonized
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))

        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1

        p.insert(0, i)
        q.insert(0, j)

    p.insert(0, 0)
    q.insert(0, 0)
    return (np.array(p), np.array(q))

