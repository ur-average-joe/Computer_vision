import numpy as np
from numpy import matlib
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import scipy
import skimage as io
from PIL import Image


print("Homework One Report:\n====1 Matrix Manipulation====")
"""1.	Basic Matrix/Vector Manipulation (20 points)
In Python, please calculate the following. Given matrix M and vectors a,b,c, such that:
"""
#a)	Define Matrix M and Vectors a, b, c in Python.  You should use Numpy.
M = np.array(
            [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 2, 2]])
a = np.array([1, 1, 0])
b = np.array([-1, 2, 5])
c = np.array([0, 2, 3, 2])

#b)	Find the dot product of vectors a and b.  (If you don’t know what dot product means,  read about dot product in any linear algebra tutorial. )  Save this value to the variable aDotb and write its value in your report. 
print("The dot product is:")
aDotb = np.dot(a, b)
print(aDotb)

#c)	Find the element-wise product of a and b   and  
#write it to your report. 
print("Element-wise product:")
print((np.multiply(a, b)).T)

#d)	Find a^tb and write it to your report. 
d = np.dot(a.T, b) * np.dot(M, a)
print("(a^Tb)Ma =")
print(d)

#e)	Without using a loop, multiply each row and of M element-wise by a. (Hint: the function repmat() may come in handy). Write the results in your report. 
newA = np.matlib.repmat(a.T, 4, 1)
e = np.multiply(newA, M)
print(e)

#f)	Without using a loop, sort all of the values of the new M from (e) in increasing order and plot them in your report. 
sortedMatrix = e.flatten()
sortedMatrix.sort()
sortedMatrix.shape = (e.shape[0], e.shape[1])
print(sortedMatrix)

print(" ====2 Basic Image Manipulation====")
#2.	Basic Image Manipulations (20 points) 

#a)	Read in the images, image1.jpg and image2.jpg
#There are many different ways to read in images. Matplotlib.image is a good one. cv2.imread() is another good one.  You can also use scipy.misc.imread().
image1 = Image.open('images/puppy.jpg')
image2 = Image.open('images/puppy.jpg')


#b)	Convert the images to double precision and rescale them to stretch from minimum value 0 to maximum value 1.
image1 = np.float64(misc.imread('images/puppy.jpg', flatten = 1, mode='F'))
image2 = np.float64(misc.imread('images/puppy.jpg', flatten = 1, mode='F'))

#c)	Add the images together and re-normalize them to have minimum, value 0 and maximum value 1. Display this image in your report. 
#normalization
normalizedImage1 = np.zeros((720, 652))
normalizedImage1 = cv2.normalize(image1, normalizedImage1, 0, 1, cv2.NORM_MINMAX)
normalizedImage2 = np.zeros((720, 652))
normalizedImage2 = cv2.normalize(image2, normalizedImage2, 0, 1, cv2.NORM_MINMAX)
Normalized = np.zeros((720, 652))
Normalized = cv2.normalize((normalizedImage1 + normalizedImage2), Normalized, 0, 1, cv2.NORM_MINMAX)
plt.imshow(Normalized, cmap=plt.cm.gray)
plt.show()

#d)	Create a new image such that the left half of the image is the left half of image1 and the right half of the image is the right half of image 2. 
croppedImage1 = normalizedImage1[0:720, 0:652/2]
croppedImage2 = normalizedImage2[:, 652/2:]
newImage = np.concatenate((croppedImage1, croppedImage2), axis=1)

#e)	Using a for loop, create a new image such that every odd numbered row is the corresponding row from image1 and the every even row is the corresponding row from image2 (Hint: Remember that indices start at 0 and not 1 in Python). Display this image in your report.
frankensteinImage = np.empty((720, 652))
for row in range(len(normalizedImage1)):
    if row % 2 == 1: #odd
        frankensteinImage[row] = normalizedImage1[row]

    else: #even
        frankensteinImage[row] = normalizedImage2[row]

plt.imshow(frankensteinImage)
plt.show()


#f)	Accomplish the same task as part e) without using a for-loop (the functions reshape and repmat may be helpful here).
oddMatrix = normalizedImage1[1::2]
evenMatrix = normalizedImage2[::2]
frankensteinImage2 = np.empty((720, 652))
frankensteinImage2[1::2] = oddMatrix
frankensteinImage2[::2] = evenMatrix

#g)	Convert the result from part f) to a grayscale image. Display the grayscale image with a title in your report.
plt.imshow(frankensteinImage2, cmap=plt.cm.gray)
plt.title("Two Birds in One")
plt.axis('off')
plt.show()

print(" ====3 Compute the average faces====")

#a)	Download labeled faces in the Wild dataset (google: LFW face dataset or click the link). Pick a face with at least 100 images. 

#b)	Call numpy.zeros to create a 250 x 250 x 3 float64 tensor to hold the results. 
results = np.float64(np.zeros((255,255,0)))

#c)	Read each image with skimage.io.imread, convert to float and accumulate. 
results = skimmage.io.imread_collection("images/lfw/George_W_Bush")

#d)	Write the averaged result with skimage.io.imsave.  Post your resulted image in the report. 
