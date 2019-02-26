import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')

print("im: ", "\n", im)
plt.imshow(im, cmap=plt.cm.gray)

# 2. Blur the image using a Gaussian filter
im = ndimage.gaussian_filter(im, 8)
plt.imshow(im, cmap=plt.cm.gray)

# 3. Apply Sobel filter to both x and y direction.
sx = ndimage.sobel(im, axis=0, mode='constant')
sy = ndimage.sobel(im, axis=1, mode='constant')

plt.figure(figsize=(20, 4))
plt.subplot(141)
# original
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('square', fontsize=20)
plt.subplot(142)
# sobel x
plt.imshow(sx)
plt.axis('off')
plt.title('Sobel (x direction)', fontsize=20)
plt.subplot(143)
plt.imshow(sy)
plt.axis('off')
plt.title('Sobel (y direction)', fontsize=20)
plt.subplot(144)
# plt.imshow(sob)
plt.axis('off')
plt.title('magnitude', fontsize=20)

plt.show()