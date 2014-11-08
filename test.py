#import pyximport
#pyximport.install(reload_support=True)

from IPython import get_ipython
ipython = get_ipython()

import numpy as np
import trace
import dtw
import cdtw
import cdtw_window

x = np.array(trace.data[0:700])
xs = np.array([[x,x]])
y = np.array(trace.data[0:700])
ys = np.array([y,y])
#
#import time
#a = time.time()
#D,d = cdtw.dtw(x,y)
#b = time.time()
#b - a
#d = cdtw.dtw_unbound(x,y)
a,b = cdtw.dtw_unbound(x,y)


#%timeit -n2 -r3 cdtw.dtw(x,y)
#%timeit -n2 -r3 dtw.dtw(x,y)
#D = cdtw.window(x[80:150],y)

#dists = cdtw_window.compute_similarity(xs,ys)
#import cdtw_window
#train = np.array([ -2.08008619e+02,  -7.64264322e+01,   2.45767851e+01,
#         3.35988164e+00,  -5.49476614e+01,   8.18398420e+00,
#         7.57698986e+01,   4.24065775e+01,   9.81242163e+01,
#         2.51901173e+02,   3.08802372e+02,   1.77179987e+02,
#        -8.90297528e+01,  -2.27990490e+02,  -1.21244452e+02,
#        -1.49563695e+02,  -4.79656729e+02,  -5.27068735e+02,
#        -1.35897822e+02,   4.96109489e+01,  -5.53851366e+01,
#        -7.14041520e+01,   1.35905956e+02,   3.72311467e+02,
#         2.34914110e+02,  -3.43249302e+02,  -9.93804995e+02,
#        -7.77931385e+02,   5.65242438e+02,   1.61736242e+03,
#         1.43388426e+03,  -1.01664939e+02,  -2.12509712e+03,
#        -1.26440480e+03,   3.48366053e+03,   6.36029185e+03,
#         2.45869846e+03,  -4.53305125e+03,  -7.00091401e+03,
#        -2.20036013e+03,   3.71089363e+03,   2.27376246e+03,
#        -3.70913053e+03,  -4.48942678e+03,  -1.07862257e+02,
#         2.48741221e+03,   1.77511761e+03,   1.32398734e+03,
#         1.52806605e+03,  -1.30408987e+03,  -3.98881999e+03,
#         3.65601424e+02,   6.32894883e+03,   4.53283688e+03,
#        -1.74388275e+03,  -4.09108639e+03,  -2.46073449e+03,
#        -1.42756386e+03,  -5.02210413e+02,   1.40319710e+03,
#         1.34927618e+03,  -1.01319110e+02,   2.65992283e+02,
#         1.27073746e+03,   8.79275837e+02,  -3.26566623e+02,
#        -1.33707077e+03])
#x1 = cdtw_window._mpt_scale(train)
#
#pos_set2=data.Dataset.normalize_instance_array(pos_set.instances_array, 'mpt')
