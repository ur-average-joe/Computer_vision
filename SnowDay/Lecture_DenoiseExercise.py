from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# let's open the original image. 
pepper = misc.imread('peppers256.png',flatten=1)
# 2D fft of the pepper image using np.fft.fft2

# Shift the zero-frequency component to the center of the spectrum.

# computing magnitude by 20*np.log(np.abs(fshift))


# FFT the noisey images. 
pepper_noise = misc.imread('peppers256_noisy.png',flatten=1)
# do the above steps to the noisey images.

#
# Plotting both original and noisey images' FFT spectrum
plt.subplot(121),plt.imshow(magnitude_spectrum_p, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum_pn, cmap = 'gray')
plt.title('Noisey'), plt.xticks([]), plt.yticks([])
plt.show()

# create a uniform gray image and perform fft on it and plot the FFT spectrum



