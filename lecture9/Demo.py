# you can also look at the code here:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Messi15.png')

# scaling is just resizing the image.
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)


# translation is the shifting of object's location.
# Translation matrix
# M = [ 1 ,0 ; 0 1; tx,ty]
rows,cols,t = img.shape
M = np.float32([[1,0,100],[0,1,50]])
print ("1st print:")
print (M)
dst = cv2.warpAffine(img,M,(cols,rows))

#cv2.imshow('img',dst)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# rotation.
#
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
print ("2nd print:")
print (M)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(1)
cv2.destroyAllWindows()

## Exercise 1
# How do you rotate the image by 45 degree and then scale it to half?
# Can you write out the matrix M?

## Exercise 2

# Can you rotate the image back? By constructing an inverse Rotation


## Exercise 2
## Write your own matrix for a shearing matrix and apply to the Messi image

# affine transformation
# In affine transformation, all parallel lines in the original image will still be parallel in the output image.
# To find the transformation matrix, we need three points from input image and their corresponding locations in output image.
# Then cv2.getAffineTransform will create a 2x3 matrix
# which is to be passed to cv2.warpAffine.

img = cv2.imread('Messi15.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)
print ("3rd print:")
print (M)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
