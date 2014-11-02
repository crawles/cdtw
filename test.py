#import pyximport
#pyximport.install(reload_support=True)

from IPython import get_ipython
ipython = get_ipython()

import numpy as np
import trace
import dist
import dtw
import cdtw

x = np.array(trace.data[0:200])
y = np.array(trace.data[0:200])

#d = cdtw.dtw(x,y)
#d = dtw.dtw(x,y)

%timeit -n2 -r3 cdtw.dtw(x,y)
%timeit -n2 -r3 dtw.dtw(x,y)
#D = cdtw.window(x[80:150],y)

