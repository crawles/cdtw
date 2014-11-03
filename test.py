#import pyximport
#pyximport.install(reload_support=True)

from IPython import get_ipython
ipython = get_ipython()

import numpy as np
import trace
import dtw
import cdtw
import cdtw_window

x = np.array(trace.data[0:200])
xs = np.array([x,x,x,x])
y = np.array(trace.data[50:100])
ys = np.array([y,y])

#d = cdtw.dtw(x,y)
#d = dtw.dtw(x,y)

#%timeit -n2 -r3 cdtw.dtw(x,y)
#%timeit -n2 -r3 dtw.dtw(x,y)
#D = cdtw.window(x[80:150],y)

dists = cdtw_window.compute_similarity(xs,ys)
