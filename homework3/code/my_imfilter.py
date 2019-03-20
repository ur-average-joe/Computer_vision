# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:39:14 2015

@author: bxiao
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc



#% This function is intended to behave like the built in function ndimage.filter. 
#% While terms like "filtering" and
#% "convolution" might be used interchangeably, and they are indeed nearly
#% the same thing, there is a difference:
#% from 'help filter2'
#%    2-D correlation is related to 2-D convolution by a 180 degree rotation
#%    of the filter matrix.
#
#% Your function should work for color images. Simply filter each color
#% channel independently.
#
#% Your function should work for filters of any width and height
#% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
#% restriction makes it unambigious which pixel in the filter is the center
#% pixel.
#
#% Boundary handling can be tricky. The filter can't be centered on pixels
#% at the image boundary without parts of the filter being out of bounds. If
#% you look at 'help conv2' and 'help imfilter' you see that they have
#% several options to deal with boundaries. You should simply recreate the
#% default behavior of imfilter -- pad the input image with zeros, and
#% return a filtered image which matches the input resolution. A better
#% approach is to mirror the image content over the boundaries for padding.
#
#% % Uncomment if you want to simply call imfilter so you can see the desired
#% % behavior. When you write your actual solution, you can't use imfilter,
#% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
#% % computation. It might be slow.
#% output = imfilter(image, filter);


