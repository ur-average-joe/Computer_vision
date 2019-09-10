from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random

# let's open the original image. 
pepper = misc.imread('peppers256.png',flatten=1)
# 2D fft of the pepper image using np.fft.fft2
pepper_fft2 = np.fft.fft2(pepper)
# Shift the zero-frequency component to the center of the spectrum.
fshift = np.fft.fftshift(pepper_fft2)
# computing magnitude by 20*np.log(np.abs(fshift))
mag_spec_pepper = 20*np.log(np.abs(fshift))

""" 
def addPepperNoise(im):
    rows, columns = im.shape
    p = random.randrange(1, 10)/100
    output = np.zeros(im.shape, np.uint8)

    for i in range(rows):
        for j in range(columns):
            r = random.random()
            if r < p/2:
                #pepper
                output[i][j] = 0
            elif r < p:
                output[i][j] = 255
            else:
                output[i][j] = im[i][j]
        
    plt.imshow(output)
    plt.title("Image with pepper noise")
    plt.show()

    plt.imsave("peppers256_noisy.png", output)

addPepperNoise(pepper) 
"""
# FFT the noisey images. 
pepper_noise = misc.imread('peppers256_noisy.png',flatten=1)
# do the above steps to the noisey images.
pepper_noise_fft2 = np.fft.fft2(pepper_noise)
fshift_ = np.fft.fftshift(pepper_noise_fft2)
mag_spec_pepper_ = 20*np.log(np.abs(fshift_))

# Plotting both original and noisey images' FFT spectrum
plt.subplot(121)
plt.imshow(mag_spec_pepper, cmap = 'gray')
plt.title('Original img FFT Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(mag_spec_pepper_, cmap = 'gray')
plt.title('Noisey img FFT Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()

# create a uniform gray image and perform fft on it and plot the FFT spectrum
rows, columns = pepper.shape
grey = np.random.random((rows, columns))#same shape as peppers

def fftSpectrum(im):
    im = np.fft.fft2(im)
    fshift = np.fft.fftshift(im)
    magSpec = 20*np.log(np.abs(fshift))
    return magSpec

plt.subplot(121)
plt.imshow(grey, cmap="gray")
plt.title("Original")

plt.subplot(122)
plt.imshow(fftSpectrum(grey), cmap="gray")
plt.title("FFT Spectrum")
plt.show()



