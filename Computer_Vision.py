import numpy as np
import cv2
import matplotlib.pyplot as plt

#       numpy exersizes 

space = ""

print("1.	Create a 3x3 identity matrix")
a = np.eye(3)
print(a)
print(space)
print(space)

print("2.	Create a 3x3x array with random values") 
b = np.arange(9).reshape(3,3)
print(b)
print(space)
print(space)

print("3.	Create a 10x10 array with random values and find the min and max values")
c = np.random.random((10, 10))
cmin, cmax = c.min(), c.max()
print(c) 
print("Min: " + str(cmin))
print("Max: " + str(cmax))
print(space)
print(space)

print("4.	How to add a border (filled with 0s) around an existing array")
import numpy as np
d = np.ones((5,5))
d = np.pad(d, pad_width=1, mode='constant', constant_values=0)
print(d)
print(space)
print(space)

print("5.	Create a random vector of size 40 and find the mean value")
e = np.random.random(40)
e = e.mean()
print("Mean: "+ str(e))
print(space)
print(space)

print("6.	Create a checkerboard 8x8 matrix using the tile function")
f = np.zeros((8,8),dtype=int)
f[1::2,::2] = 1
f[::2,1::2] = 1
print(f)
print(space)
print(space)

print("7.	Create a vector of 100 uniform distributed values between 0 and 1.") 
g = np.random.uniform(low=0.0, high=1.0, size=100)
print(g)
print(space)
print(space)

#       matplotlib exercizes

print("Using Numpy, Create a vector of 1000 random values drawn from a normal distribution\nwith a mean of 0 and a standard deviation of 0.5")
e = np.random.normal(0, 0.5, 1000)
print(e)
print(space)
print(space)

x = []
y = []
x1 = 0
y1 = 0

for i in range(1001): 
    x.append(x1)
    y.append(y1)
    x1 += 1
    y1 += 1
    if i == 500:
        x1 += 1
        y1 -= 1
        if x1 == 1000:
            break 
        elif y1 == 0:
            break
plt.title("Normal Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.bar(x, y, label="Normal Distribution")
plt.show()

# Load an color image in grayscale
img = cv2.imread('images/duck.png',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('duck_grey.png',img)
    cv2.destroyAllWindows()