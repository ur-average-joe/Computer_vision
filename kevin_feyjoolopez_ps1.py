import numpy as np
import cv2
from scipy import ndimage
import skimage
import PIL
from scipy import misc
import matplotlib.pyplot as plt
from pylab import *

#problem 1
print("Problem 1: Basic Matrix/Vector Manipulation\n")
print(" A: Define Matrix M and Vectors a, b, c in Python.")
a = [1,1,1]
b = [1,1,1]
c = [1,1,1]
m = np.array([[a],[b],[c]])
print(a,"\n",b,"\n",c,"\n","\n",m,"\n")

print(" B: Find the dot product of vectors a and b.\n Save this value to the variable aDotb and write its value in your report.")
aDotb = np.dot(a, b, out=None)
print(aDotb,"\n","\n")

print(" C: Find the element-wise product of a and b  and write it to your report.")
ewp = np.multiply(a, b)
print(ewp,"\n","\n")

print(" D: Using a for loop, create a new image such that every odd numbered row is the corresponding row from image1 and the \n every even row is the corresponding row from image2 (Hint: Remember that indices start at 0 and not 1 in Python). Display this image in your report.\n")

img = cv2.imread("images/oreo_resized.jpg")
image1 = np.float64(misc.imread('images/oreo_filtered.jpg', flatten = 1, mode='F'))
image2 = np.float64(misc.imread('images/oreo_resized.jpg', flatten = 1, mode='F'))
normalizedImage1 = np.zeros((850, 1000))
normalizedImage1 = cv2.normalize(image1, normalizedImage1, 0, 1, cv2.NORM_MINMAX)
normalizedImage2 = np.zeros((850, 1000))
normalizedImage2 = cv2.normalize(image2, normalizedImage2, 0, 1, cv2.NORM_MINMAX)

img = np.empty((850, 1000))
for row in range(len(normalizedImage1)):
    if row % 2 == 1: #odd
        img[row] = normalizedImage1[row]

    else: #even
        img[row] = normalizedImage2[row]

plt.imshow(img)
plt.show()

""" 
img1 = np.zeros((256,256))
img2 = np.random.rand(256,256)
img3 = np.array([[]])

for row in img1:
    i = 0
    i = img1[i]
    np.append(img3, i)
    i += 2
 if row == 255:
        break
for row in img2:
    b = 0
    b = img1[b]
    np.append(img3, b)
    b += 2
     if row == 255:
        break

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(img3)
plt.show() """



print(" E: Accomplish the same task as part e) without using a for-loop (the functions reshape and repmat may be helpful here).\n")



print(" F: Convert the result from part f) to a grayscale image. Display the grayscale image with a title in your report.\n")
plt.imshow(img3, 0)#convert to grey scale