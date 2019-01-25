# Basic image tutorials
"""
Created on Tue Jan 13 20:47:26 2015
@author: bxiao
"""
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from mpl_toolkits.mplot3d import Axes3D


def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

##  load and display an image
lambo = misc.imread('images/lambo.jpg',flatten=1)
plt.imshow(lambo,cmap=plt.cm.gray)
#plt.show()

# here is the code to resize a big image, it is not effective in the lena case
#lambo_resized = misc.imresize(lambo, (512,512), interp='bilinear', mode=None)
#plt.imshow(lambo_resized,cmap=plt.cm.gray)
#plt.show()

# print out some information
print("shape:")
print(lambo.shape)
print("type:")
print(lambo.dtype)
print("max:")
print(lambo.max())
print("min:")
print(lambo.min())

# change brightness
# darker

lambo_dark = lambo-125
plt.imshow(lambo_dark, vmin = 0, vmax = 128,cmap=plt.cm.gray)
plt.show()
#misc.imwrite('lambo_dark.png', lambo_dark)

#
#
## create a surface plot of the image
x, y = np.ogrid[0:lambo.shape[0], 0:lambo.shape[1]]
fig=plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,lambo,rstride=4, cstride=4, cmap=plt.cm.jet, linewidth=0.2)
plt.show()


# Image convolution
# mean filter
# there are many ways to do convolution in Python.
# I use scipy.ndimage.filters.convolve
# Other choices are: scipy.signal.convolve2d
k = np.ones((5,5))/25
# convolve with the image
b= ndimage.filters.convolve(lambo,k)
b = misc.toimage(b)
b.save('lambo_blur.png')
plt.imshow(b, cmap=plt.cm.gray)
plt.show()


# or you can use uniform_filter, same as filters.convolve with a uniform kernel
# local_mean = ndimage.uniform_filter(, size=5)
# plt.imshow(local_mean, cmap=plt.cm.gray)
# plt.show()

#sharpening a blurred image
blurred = ndimage.gaussian_filter(lambo, 5)
print(blurred)
plt.imshow(blurred, cmap=plt.cm.gray)
plt.show()

filter_blurred= ndimage.gaussian_filter(blurred,1)
alpha = 30
sharpened = blurred + alpha * (blurred - filter_blurred)
# #
# # plotting3 figures in one subplot
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(lambo, cmap=plt.cm.gray, vmin = 0, vmax = 255)
plt.subplot(132)
plt.imshow(blurred,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(sharpened, cmap=plt.cm.gray)
plt.axis('off')

plt.show()
#
#
# #==============================================================================
# # ### cropping
# # ##lx,ly = lena.shape
# # ##crop_lena = lena[lx / 4: - lx / 4, ly / 4: - ly / 4]
# # ##flip_ud_lena = np.flipud(lena)
# # ##rotate_lena = ndimage.rotate(lena, 45)
# #==============================================================================

# #==============================================================================
#  ## generate image from noise
# #plt.figure
# #im = np.zeros((128, 128))
# #im[32:-32, 32:-32] = 1
# #im = ndimage.rotate(im, 15, mode='constant')
# #im = ndimage.gaussian_filter(im, 4)
# #im += 0.2 * np.random.random(im.shape)
# #plt.imshow(im, cmap=plt.cm.jet)
# #plt.show()
#==============================================================================
