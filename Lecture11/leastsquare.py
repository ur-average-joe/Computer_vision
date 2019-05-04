import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T

ls = np.linalg.lstsq(A, y)
m, c = ls[0]
print(m, c)
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])


plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()

# exercises: curve fitting of the following data. 
# Try to use both numpy.linalg.lstsq and scipy.optimize.leastsq

xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])



