import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import axes
from skimage.feature import graycomatrix, graycoprops

import skimage as ski
import cv2 as cv

# Input number of convection files and tile size
#Possible Examples: 16, 21, 25, 50, 55, 61
num= 204
num1 = 16
num2 = 17
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

# Get sample image from Convection data
x_train_vis = np.zeros((num,256,256,9), dtype='float32')
y_train = np.zeros((num,256,256), dtype = 'float32')
f = open(r'/mnt/data1/ylee/for_Jason/20190523_seg_mrms_256_comp_real.bin','rb')
data = np.fromfile(f,dtype='float32')

for j in range(num1, num2):
        x_train_vis[j,:,:,:] = np.reshape(data[(j*(692224)):(j*(692224)+589824)],(256,256,9))
        y_train[j,:,:] = np.reshape(data[(626688 + j*(692224)):(626688+j*(692224)+65536)], (256,256))

# Split sample images into tiles, compute GLCMs and Haralick statistics, create image
for n in range(num1, num2):
    isamp = n
    data = x_train_vis[isamp,:,:,0]
    data *= 100
    data=data.astype(np.uint8)

    data_truth = y_train[isamp,:,:]
    data_truth *= 100
    data_truth = data_truth.astype(np.uint8)

    tiles = []
    convolve_tiles = []
    glcms = []

    contrast_value  = []
    contrast_values = []

    c_tile_means = []
    c_tile_mins  = []


    #Define the convolve masks
    #3x3 convolve mask
    kernel_1 = np.ones((3,3), np.float32)/9
    #5x5 convolve mask
    kernel_2 = np.ones((5,5), np.float32)/25
    #7x7 convolve mask
    kernel_3 = np.ones((7,7), np.float32)/49
    #9x9 convolve mask
    kernel_4 = np.ones((9,9), np.float32)/81


    #Applies the desired Kernel
    convolve_data = cv.filter2D(src = data, ddepth = -1, kernel = kernel_4)

    for r in range(0, 256, tile_size):
        for c in range(0, 256, tile_size):
            tile = data[r:r+tile_size, c:c+tile_size]
            convolve_tile = convolve_data[r:r + tile_size, c:c + tile_size]

            tiles.append(tile)
            convolve_tiles.append(convolve_tile)

            c_tile_mean = np.mean(convolve_tile)
            c_tile_min  = np.min(convolve_tile)

            c_tile_means.append(c_tile_mean)
            c_tile_mins.append(c_tile_min)

            distances=[1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm0 = graycomatrix(tile, distances=distances, angles=[0], levels=256, symmetric=False)
            glcm1 = graycomatrix(tile, distances=distances, angles=[np.pi/4], levels=256, symmetric=False)
            glcm2 = graycomatrix(tile, distances=distances, angles=[np.pi/2], levels=256, symmetric=False)
            glcm3 = graycomatrix(tile, distances=distances, angles=[3 * np.pi/4], levels=256, symmetric=False)
            glcm=(glcm0 + glcm1 + glcm2 + glcm3)/4 #compute mean matrix
            glcms.append(glcm)

            contrast = (float(graycoprops(glcm, 'contrast')))
            contrast_value.append(contrast)

    c_mean_avg = np.mean(c_tile_means)
    c_mins_avg = np.mean(c_tile_mins)

    c_tile_means = np.array(c_tile_means)
    c_tile_means = c_tile_means.reshape((int(256/tile_size), int(256/tile_size)))

    c_tile_mins = np.array(c_tile_mins)
    c_tile_mins = c_tile_mins.reshape((int(256/tile_size), int(256/tile_size)))

    c_mean_mask = (c_tile_means >= c_mean_avg + 10).astype(int)
    c_min_mask  = (c_tile_mins  >= c_mins_avg + 10).astype(int)

    contrast_max = max(contrast_value)

    for val in contrast_value:
        contrast_values.append(val/contrast_max)

    #Declare feature
    feature = 'contrast'
    feature_values =  contrast_values

    fig, ax = plt.subplots(3,3)

    ax[0,0].imshow(data, cmap='gray', origin='lower')
    ax[0,1].imshow(convolve_data, cmap='gray', origin='lower')
    ax[0,2].imshow(data_truth, cmap = 'gray', origin = 'lower')

    ax[1,0].imshow(data, cmap = 'gray', origin = 'lower')
    ax[1,1].imshow(data, cmap = 'gray', origin = 'lower')
    ax[1,2].imshow(data, cmap = 'gray', origin = 'lower')

    ax[2,0].set_visible(False)
    ax[2,1].imshow(data, cmap = 'gray', origin = 'lower')
    ax[2,2].imshow(data, cmap = 'gray', origin = 'lower')

    plt.xlim(0,256)
    plt.ylim(0,256)
    plt.axis('scaled')

    index_1 = 0
    index_2 = 0

    for r in range (0, 256, tile_size):
        index_2 = 0
        for c in range (0, 256, tile_size):
            x_offset = c
            y_offset = r

            rect_original  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_for_min   = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_for_mean  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_min_mask  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor = 'white', edgecolor = 'none', alpha = c_min_mask[index_1, index_2])

            rect_mean_mask = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor = 'white', edgecolor = 'none', alpha = c_mean_mask[index_1, index_2])


            ax[1,0].add_patch(rect_original)

            ax[1,1].add_patch(rect_min_mask)
            ax[2,1].add_patch(rect_mean_mask)

            if(c_min_mask[index_1, index_2] == 1):
                ax[1,2].add_patch(rect_for_min)

            if(c_mean_mask[index_1, index_2] == 1):
                ax[2,2].add_patch(rect_for_mean)

            index_2 += 1
        index_1 += 1
    plt.show()
