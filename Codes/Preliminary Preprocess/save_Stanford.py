import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import scipy.ndimage


def loadData(sub_ID):

    try:
        training_img = nib.load(INPUT_FOLDER + "\\00" + str(sub_ID) + "\\session_1\\anat_1\\mprage.nii.gz")
    except:
        training_img = nib.load(INPUT_FOLDER + "\\00" + str(sub_ID) + "\\session_1\\anat_1\\mprage.nii.gz")

    training_pixdim = training_img.header['pixdim'][1:4]
    
    training_data = training_img.get_data()
    
    return training_data, training_pixdim


def saveDataset(dataset, name):
    np.save("data\\" + str(name), dataset)


def show_slices(slices):
       """ Function to display row of image slices """
       fig, axes = plt.subplots(1, len(slices))
       for i, slice in enumerate(slices):
           axes[i].imshow(slice.T, cmap="gray", origin="lower")


def resample(image, pixdim, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = pixdim

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image

    
def crop_center(img,cropx,cropy, cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[startx:startx+cropx,starty:starty+cropy, startz:startz+cropz]


INPUT_FOLDER = 'Olin'
SITE_ID = 'OLIN'

labels_file = pd.read_csv('FYP_Phenotypic.csv')
print(labels_file[labels_file.SITE_ID == SITE_ID])

selected_rows = labels_file[labels_file.SITE_ID == SITE_ID]

for index, row in selected_rows.iterrows():
    print ("Saving data id: " , row['ID'])
    single_data, single_dim = loadData(row['SUB_ID'])
    
    single_resampled = resample(single_data, single_dim, [2,2,2])

    DESIRED_SHAPE = (130, 130, 110)
    cropx = single_resampled.shape[0]
    cropy = single_resampled.shape[1]
    cropz = single_resampled.shape[2]
    if DESIRED_SHAPE[0] < single_resampled.shape[0]:
        cropx = DESIRED_SHAPE[0]
    if DESIRED_SHAPE[1] < single_resampled.shape[1]:
        cropy = DESIRED_SHAPE[1]
    if DESIRED_SHAPE[2] < single_resampled.shape[2]:
        cropz = DESIRED_SHAPE[2]

    cropped_img = crop_center(single_resampled, cropx, cropy, cropz)
    
    X_before = int((DESIRED_SHAPE[0]-cropped_img.shape[0])/2)
    Y_before = int((DESIRED_SHAPE[1]-cropped_img.shape[1])/2)
    Z_before = int((DESIRED_SHAPE[2]-cropped_img.shape[2])/2)

    npad = ((X_before, DESIRED_SHAPE[0]-cropped_img.shape[0]-X_before), (Y_before, DESIRED_SHAPE[1]-cropped_img.shape[1]-Y_before), (Z_before, DESIRED_SHAPE[2]-cropped_img.shape[2]-Z_before))
    single_padded = np.pad(cropped_img, pad_width=npad, mode='constant', constant_values=0)
            
    saveDataset(single_padded, str(row['ID']) + "#" + str(single_padded.shape))

    print(str(single_data.shape) + "  --->   " + str(single_resampled.shape) + "  --->   " + str(cropped_img.shape) + "  --->   " + str(single_padded.shape))
    
    show_slices([single_padded[int(single_padded.shape[0]/2), :, :],
                 single_padded[:, int(single_padded.shape[1]/2), :],
                 single_padded[:, :, int(single_padded.shape[2]/2)]])
    plt.show()

