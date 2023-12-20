import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import axes
from skimage.feature import graycomatrix, graycoprops

import skimage as ski
import cv2 as cv
import pickle as pkl
###############################

<<<<<<< HEAD
=======
#Load data
>>>>>>> 0c47610... updating
filepath = r'/home/nmitchell/GLCM_project/metrics'
with open(filepath, 'rb') as file:
    metrics = pkl.load(file)

# Input number of convection files and tile size
#Possible Examples: 16, 21, 25, 50, 55, 61
#NEW: 10, *11*, 19
num= 204
<<<<<<< HEAD
num1 = 16
num2 = 17
=======
num1 = 4
num2 = 5
>>>>>>> 0c47610... updating
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

<<<<<<< HEAD
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
#    data = x_train_vis[isamp,:,:,0]
#    data *= 100
#    data=data.astype(np.uint8)

#    data_truth = y_train[isamp,:,:]
#    data_truth *= 100
#    data_truth = data_truth.astype(np.uint8)

#    data_truth_color = np.zeros(len(data_truth)*len(data_truth)*4)
#    data_truth_color = data_truth_color.reshape(len(data_truth),len(data_truth),4)

#    for i in range(len(data_truth)):
#        for j in range(len(data_truth)):
#            if(data_truth[i,j] == 100):
#                data_truth_color[i,j,:] = [1,0,0,1]
#            else:
#                data_truth_color[i,j,:] = [1,1,1,1]


#    print(metrics['Original Image'][num1])
#    print(metrics['Original Image'][10])

    data = metrics['Original Image'][isamp - 1]
    data_truth_color = metrics['Ground Truth'][isamp - 1]
    convolve_data = metrics['Convolved Image'][isamp - 1]

    tiles = []
    convolve_tiles = []
    glcms = []
    glcms_c = []

    contrast_value  = []
    contrast_values = []

    contrast_value_c  = []
    contrast_values_c = []

    c_tile_means = []
    c_tile_mins  = []


    #Define the convolve mask
    #9x9 convolve mask
 #   kernel_9x9 = np.ones((9,9), np.float32)/81

    #Apply the desired Kernel
#    convolve_data = cv.filter2D(src = data, ddepth = -1, kernel = kernel_9x9)

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

            #GLCM for Convolved Data
            glcm0_c = graycomatrix(convolve_tile, distances = distances, angles = [0],           levels = 256, symmetric = False)
            glcm1_c = graycomatrix(convolve_tile, distances = distances, angles = [np.pi/4],     levels = 256, symmetric = False)
            glcm2_c = graycomatrix(convolve_tile, distances = distances, angles = [np.pi/2],     levels = 256, symmetric = False)
            glcm3_c = graycomatrix(convolve_tile, distances = distances, angles = [3 * np.pi/4], levels = 256, symmetric = False)
            glcm_c  = (glcm0_c + glcm1_c + glcm2_c + glcm3_c)/4 #compute mean matrix
            glcms_c.append(glcm_c)

            contrast = (float(graycoprops(glcm, 'contrast').ravel()[0]))
            contrast_value.append(contrast)

            contrast_c = (float(graycoprops(glcm_c, 'contrast').ravel()[0]))
            contrast_value_c.append(contrast_c)

    c_mean_avg = np.mean(c_tile_means)
    c_mins_avg = np.mean(c_tile_mins)

    c_tile_means = np.array(c_tile_means)
    c_tile_means = c_tile_means.reshape((int(256/tile_size), int(256/tile_size)))

    c_tile_mins = np.array(c_tile_mins)
    c_tile_mins = c_tile_mins.reshape((int(256/tile_size), int(256/tile_size)))

    c_mean_mask = (c_tile_means >= c_mean_avg).astype(int)
    c_min_mask  = (c_tile_mins  >= c_mins_avg).astype(int)


    contrast_max   = max(max(contrast_value), max(contrast_value_c))
    contrast_min   = min(min(contrast_value), min(contrast_value_c))


    for val in contrast_value:
        contrast_values.append((val - contrast_min)/(contrast_max - contrast_min))

    for val in contrast_value_c:
        contrast_values_c.append((val - contrast_min)/(contrast_max - contrast_min))
=======
# Split sample images into tiles, compute GLCMs and Haralick statistics, create image
for n in range(num1, num2):
    isamp = n

    contrast_value   = metrics["Contrast Values"][isamp]
    contrast_value_c = metrics["Convolved Contrast Values"][isamp]

    contrast_values   = []
    contrast_values_c = []

    for val in contrast_value:
        contrast_values.append((val - min(contrast_value))/(max(contrast_value) - min(contrast_value)))
    for val in contrast_value_c:
        contrast_values_c.append((val - min(contrast_value_c))/(max(contrast_value_c) - min(contrast_value_c)))
>>>>>>> 0c47610... updating

    #Declare feature
    feature = 'contrast'
    feature_values   =  contrast_values
    feature_values_c = contrast_values_c
<<<<<<< HEAD
    fig, ax = plt.subplots(3,5)

    # og pic # convolve  # ground truth    # convolve       #
    # glcm   # min mask  # min on og glcm  # convolve glcm  # min on convolve glcm
    #        # mean mask # mean on og mask #                # mean on convolve glcm

    ### COLUMN 1 ###
    ax[0,0].set_title("Original Image")
    ax[0,0].imshow(data, cmap = 'gray', origin = 'lower')

    ax2 = ax[1,0]
    ax2.set_title("Original GLCM")
    ax2.imshow(data, cmap = 'gray', origin = 'lower')
=======
    fig, ax = plt.subplots(3,6)

    # og pic # convolve  # ground truth    # convolve       #                       # Infrared Image
    # glcm   # min mask  # min on og glcm  # convolve glcm  # min on convolve glcm  # Infrared Mask
    #        # mean mask # mean on og mask #                # mean on convolve glcm #

    ### COLUMN 1 ###
    ax[0,0].set_title("Original Image")
    ax[0,0].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ax2 = ax[1,0]
    ax2.set_title("Original GLCM")
    ax2.imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')
    ax2.set_xlabel("Max: " + "%.2f" % max(contrast_value) + "\n Min: " + "%.2f" % min(contrast_value))
    #SHOULD THIS BE MAX OF THE IMAGE OR MAX OF THE ENTIRE SET?
>>>>>>> 0c47610... updating

    ax[2,0].set_visible(False)

    ### COLUMN 2 ###
    ax[0,1].set_title("Convolved Image")
<<<<<<< HEAD
    ax[0,1].imshow(convolve_data, cmap = 'gray', origin = 'lower')

    ax[1,1].set_title("Min Mask")
    ax[1,1].imshow(data, cmap = 'gray', origin = 'lower')

    ax[2,1].set_title("Mean Mask")
    ax[2,1].imshow(data, cmap = 'gray', origin = 'lower')

    ### COLUMN  3 ###
    ax[0,2].set_title("Ground Truth")
    ax[0,2].imshow(data_truth_color, cmap = 'gray', origin = 'lower')

    ax[1,2].set_title("Min Mask Applied")
    ax[1,2].imshow(data, cmap = 'gray', origin = 'lower')

    ax[2,2].set_title("Mean Mask Applied")
    ax[2,2].imshow(data, cmap = 'gray', origin = 'lower')

    ### COLUMN 4 ###
    ax[0,3].set_title("Convolved Image")
    ax[0,3].imshow(convolve_data, cmap = 'gray', origin = 'lower')

    ax[1,3].set_title("Convolve GLCM")
    ax[1,3].imshow(data, cmap = 'gray', origin = 'lower')
=======
    ax[0,1].imshow(metrics["Convolved Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[1,1].set_title("Min Mask")
    ax[1,1].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[2,1].set_title("Mean Mask")
    ax[2,1].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ### COLUMN  3 ###
    ax[0,2].set_title("Ground Truth")
    ax[0,2].imshow(metrics["Ground Truth"][isamp], cmap = 'gray', origin = 'lower')

    ax[1,2].set_title("Min Mask Applied")
    ax[1,2].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[2,2].set_title("Mean Mask Applied")
    ax[2,2].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ### COLUMN 4 ###
    ax[0,3].set_title("Convolved Image")
    ax[0,3].imshow(metrics["Convolved Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[1,3].set_title("Convolve GLCM")
    ax[1,3].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')
    ax[1,3].set_xlabel("Max: " + "%.2f" % max(contrast_value_c) + "\n Min: " + "%.2f" % min(contrast_value_c))
>>>>>>> 0c47610... updating

    ax[2,3].set_visible(False)

    ### COLUMN 5 ###
    ax[0,4].set_visible(False)

    ax[1,4].set_title("Min Mask Applied")
<<<<<<< HEAD
    ax[1,4].imshow(data, cmap = 'gray', origin = 'lower')

    ax[2,4].set_title("Mean Mask Applied")
    ax[2,4].imshow(data, cmap = 'gray', origin = 'lower')

    for i in range (0,3):
        for j in range (0,5):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    plt.xlim(0,256)
    plt.ylim(0,256)
    plt.axis('scaled')

=======
    ax[1,4].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[2,4].set_title("Mean Mask Applied")
    ax[2,4].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ### COLUMN 6 ###
    ax[0,5].set_title("Infrared Image")
    ax[0,5].imshow(metrics["Infrared Image"][isamp], cmap = "gray", origin = "lower")

    ax[1,5].set_title("Infrared Mask")
    ax[1,5].set_xlim(0,64)
    ax[1,5].set_ylim(0,64)
    ax[1,5].imshow(metrics["Infrared Image"][isamp], cmap = "gray", origin = "upper")

    ax[2,5].set_visible(False)

    for i in range (0,3):
        for j in range (0,6):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

>>>>>>> 0c47610... updating
    index_1 = 0
    index_2 = 0

    for r in range (0, 256, tile_size):
        index_2 = 0
        for c in range (0, 256, tile_size):
<<<<<<< HEAD
            x_offset = c
            y_offset = r

            rect_original  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_convolve  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_for_min   = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_for_mean  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_min_mask  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor = 'white', edgecolor = 'none', alpha = c_min_mask[index_1, index_2])

            rect_mean_mask = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor = 'white', edgecolor = 'none', alpha = c_mean_mask[index_1, index_2])

            rect_for_min_c = plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            rect_for_mean_c= plt.Rectangle((x_offset, y_offset), tile_size, tile_size,
                             facecolor='red', edgecolor='none', alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])


            #GLCM applied to Original Image
            ax[1,0].add_patch(rect_original)

            #GLCM applied to Convolved Image
            ax[1,3].add_patch(rect_convolve)

            #min and mean masks only
            ax[1,1].add_patch(rect_min_mask)
            ax[2,1].add_patch(rect_mean_mask)

            #Min Mask applied to original GLCM
            if(c_min_mask[index_1, index_2] == 1):
                ax[1,2].add_patch(rect_for_min)
            #Mean Mask applied to original GLCM
            if(c_mean_mask[index_1, index_2] == 1):
                ax[2,2].add_patch(rect_for_mean)

            #Min Mask applied to convolved GLCM
            if(c_min_mask[index_1, index_2] == 1):
               ax[1,4].add_patch(rect_for_min_c)
            #Mean Mask applied to convolved GLCM
            if(c_mean_mask[index_1, index_2] == 1):
               ax[2,4].add_patch(rect_for_mean_c)
=======
            x_offset = c - 0.5
            y_offset = r - 0.5

            #GLCM Tiles Applied to Original Image
            original_GLCM  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            ax[1,0].add_patch(original_GLCM)

            #GLCM Tiles Applied to Convolved Image
            convolve_GLCM  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            ax[1,3].add_patch(convolve_GLCM)

            #Min Mask Applied to Original Image
            min_mask       = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            ax[1,1].add_patch(min_mask)

            #Mean Mask Applied to Original Image
            mean_mask      = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            ax[2,1].add_patch(mean_mask)

            #Min Mask Applied to Original GLCM
            min_applied_o  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor = 'white', edgecolor = 'none',
                             alpha = metrics["Tile Min Mask"][isamp][index_1, index_2])

            if(metrics["Tile Min Mask"][isamp][index_1, index_2] == 1):
                ax[1,2].add_patch(min_applied_o)

            #Mean Mask Applied to Original GLCM
            mean_applied_o = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor = 'white', edgecolor = 'none',
                             alpha = metrics["Tile Mean Mask"][isamp][index_1, index_2])

            if(metrics["Tile Mean Mask"][isamp][index_1, index_2] == 1):
                ax[2,2].add_patch(mean_applied_o)

            #Min Mask Applied to Convolved GLCM
            min_applied_c  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            if(metrics["Tile Min Mask"][isamp][index_1, index_2] == 1):
                ax[1,4].add_patch(min_applied_c)

            #Mean Mask Applied to Convolved GLCM
            mean_applied_c = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values_c[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            if(metrics["Tile Mean Mask"][isamp][index_1, index_2] == 1):
               ax[2,4].add_patch(mean_applied_c)

            index_2 += 1
        index_1 += 1


    index_1 = 0
    index_2 = 0

    for r in range (0, 64, int(tile_size/4)):
        index_2 = 0
        for c in range (0, 64, int(tile_size/4)):
            x_offset = c - 0.5
            y_offset = r - 0.5

            #250K Mask Applied to Infrared Image
            rect_for_IR = plt.Rectangle((x_offset, y_offset), tile_size/4, tile_size/4,
                          facecolor = 'red', edgecolor = 'none', alpha = (metrics["IR Mask"][isamp][index_1, index_2]))

            ax[1,5].add_patch(rect_for_IR)
>>>>>>> 0c47610... updating

            index_2 += 1
        index_1 += 1

    plt.show()

