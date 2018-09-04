import os
import matplotlib.pyplot as plt
import numpy as np
import random
#import scipy.ndimage


def show_slices(slices):
       """ Function to display row of image slices """
       fig, axes = plt.subplots(1, len(slices))
       for i, slice in enumerate(slices):
           axes[i].imshow(slice.T, cmap="gray", origin="lower")

def rot90(m, k=1, axis=2):
    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m

#first = np.load('data2\\1#(65, 65, 55).npy')

"""
X_before = 5
npad = ((5, 5), (0, 0), (0, 0))
first = np.pad(first, pad_width=npad, mode='constant', constant_values=0)

startz = 65//2-(55//2)
first = first[0:65,0:65, startz:startz+55]
"""


first = np.load('data2\\85#(65, 65, 55).npy')
#first = np.load('mean_img2.npy')
second = np.load('shuffled2\\45#(65, 65, 55).npy')

#first = rot90(first, 3, 0)
#first = rot90(first, 1, 2)
print(first.shape)
show_slices([
              first[int(first.shape[0]/2), :, :],
                 first[:, int(first.shape[1]/2), :],
              first[:, :, int(first.shape[2]/2)]])
plt.show()




show_slices([second[int(second.shape[0]/2), :, :],
                 second[:, int(second.shape[1]/2), :],
                 second[:, :, int(second.shape[2]/2)]])
plt.show()

