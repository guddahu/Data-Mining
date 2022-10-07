import sympy  as np
#import numpy as np
import statistics

x1 = np.Matrix([[0.1,0.1,0.2]])
x2 = np.Matrix([[0.5,0.6,0.4]])
y = np.Matrix([[0.1, 0.4, 0.8]])


x1 = np.Matrix([[0.1,0.5], [0.1,0.6], [0.2, 0.4]])
w = (x1.T * x1).inv() * x1.T* y.T 
print(w)
print(x1 * w)
