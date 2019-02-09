import numpy as np
import cv2
from scipy import ndimage
import skimage
import PIL
from scipy import misc
import matplotlib.pyplot as plt
from pylab import *



""" basewidth = 1000
img = Image.open("images/oreo.jpg")
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save("images/oreo_resized.jpg")
 """
'''
pil_img = array(PIL.Image.open("images/oreo_resized.jpg"))
#pil_img = PIL.Image.open("images/oreo_resized.jpg").convert("L")
imshow(pil_img)

x = [100,100,700,700]
y = [300,600,300,600]
# plot the points with red star-markers
plot(x,y,'ks:')
# line plot connecting the first two points
plot(x[:2],y[:2])
# add title and show the plot
title('Plotting: "images/oreo_resized.jpg"')
show()

'''
img = cv2.imread("images/oreo_resized.jpg")
#im_g = np.array(img, dtype=float)
kernel = np.ones((3,3)) / 9

kernel = kernel[:, :, None]
#kernel = np.ones((750, 1000, 3))/9
kk = ndimage.filters.convolve(img, kernel)
#kk = misc.toimage(kk) 
#kk = matmul(img * kernel)
#kk.save("images/oreo_filtered.jpg")
plt.imshow(kk) #, cmap=plt.cm.gray
plt.show()
print(img.shape)

#img = cv2.imread('images/oreo.jpg',0)#grayscale


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
