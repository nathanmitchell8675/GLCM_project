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

#Load data
filepath = r'/home/nmitchell/GLCM_project/metrics'
with open(filepath, 'rb') as file:
    metrics = pkl.load(file)

# Input number of convection files and tile size
#Possible Examples: 16, 21, 25, 50, 55, 61
#NEW: 10, *11*, 19
num= 204
num1 = 2
num2 = 3
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

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

    #Declare feature
    feature_values   =  contrast_values
    feature_values_c = contrast_values_c
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

    ax[2,0].set_visible(False)

    ### COLUMN 2 ###
    ax[0,1].set_title("Convolved Image")
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

    ax[2,3].set_visible(False)

    ### COLUMN 5 ###
    ax[0,4].set_visible(False)

    ax[1,4].set_title("Min Mask Applied")
    ax[1,4].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ax[2,4].set_title("Mean Mask Applied")
    ax[2,4].imshow(metrics["Original Image"][isamp], cmap = 'gray', origin = 'lower')

    ### COLUMN 6 ###
    ax[0,5].set_title("Infrared Image")
    ax[0,5].imshow(metrics["Infrared Image"][isamp], cmap = "gray", origin = "lower")

    ax[1,5].set_title("Infrared Mask")
    ax[1,5].set_xlim(0,256)
    ax[1,5].set_ylim(0,256)
    ax[1,5].imshow(metrics["Infrared Image"][isamp], cmap = "gray", origin = "lower")
    ax[1,5].set_xlabel("Mask Value: 250K")

    ax[2,5].set_visible(False)

    for i in range (0,3):
        for j in range (0,6):

            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    index_1 = 0
    index_2 = 0

#    print(metrics["IR Mask"][isamp].shape)
    print(metrics["Infrared Image"][isamp].shape)

    for r in range (0, 256, tile_size):
        index_2 = 0
        for c in range (0, 256, tile_size):
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

            #Min Mask on Original Image
            min_mask       = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor = 'mediumseagreen', edgecolor = 'none',
                             alpha = metrics["Tile Min Mask"][isamp][index_1, index_2])

            ax[1,1].add_patch(min_mask)

            #Mean Mask on Original Image
            mean_mask      = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor = 'mediumseagreen', edgecolor = 'none',
                             alpha = metrics["Tile Mean Mask"][isamp][index_1, index_2])

            ax[2,1].add_patch(mean_mask)

            #Min Mask Applied to Original GLCM
            min_applied_o  = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

            if(metrics["Tile Min Mask"][isamp][index_1, index_2] == 1):
                ax[1,2].add_patch(min_applied_o)

            #Mean Mask Applied to Original GLCM
            mean_applied_o = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor='red', edgecolor='none',
                             alpha=feature_values[(int(256/tile_size))*int(r/tile_size) + int(c/tile_size)])

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

            #250K Mask Applied to Infrared Image
            rect_for_IR    = plt.Rectangle((x_offset, y_offset), tile_size, tile_size, facecolor = 'mediumseagreen', edgecolor = 'none',
                             alpha = (metrics["IR Mask"][isamp][index_1,index_2]).astype(int))

            ax[1,5].add_patch(rect_for_IR)


            index_2 += 1
        index_1 += 1

    plt.show()

