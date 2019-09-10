import cv2
import numpy as np
import sys

A = cv2.imread('data/apple.jpg')
B = cv2.imread('data/orange.jpg')

# generate Gaussian pyramid for A
G = A
gaussianPyramidA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gaussianPyramidA.append(G)

# generate Gaussian pyramid for B
G = B
gaussianPyramidB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gaussianPyramidB.append(G)

# generate Laplacian Pyramid for A
LaplacianPyramidA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
LaplacianPyramidB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(LaplacianPyramidA,LaplacianPyramidB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)