import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import axes
from skimage.feature import graycomatrix, graycoprops

import skimage as ski
import cv2 as cv

# Imput number of convection files and tile size
num = 204
num1 = 0
num2 = 10
tile_size = 8
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

# Get sample image from Convection
x_train_vis = np.zeros((num, 256, 256, 9), dtype = 'float32')
f = open(r'/mnt/data1/ylee/for_Jason/20190523_seg_mrms_256_comp_real.bin', 'rb')
data = np.fromfile(f, dtype = 'float32')

for j in range(num1, num2):
	x_train_vis[j,:,:,:] = np.reshape(data[(j*(692224)):(j*(692224)+589824)],(256, 256, 9))


# Split sample images into tiles, compute GLCMs and Haralick statistics, create image

#This goes through the first 10 images in the file and then splits them into tiles of size 8.
for n in range(num1, num2):
	isamp = n
	data = x_train_vis[isamp,:,:,0]
	data *= 100
	data=data.astype(np.uint8)
	tiles = []
	glcms = []

	#Plotting schematics
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(data, cmap='gray', origin='lower')
	plt.xlim(0,256)
	plt.ylim(0,256)
	plt.axis('scaled')
	plt.title(isamp)
	ax[1].imshow(data, cmap='gray', origin='lower')
	plt.xlim(0,256)
	plt.ylim(0,256)
	plt.axis('scaled')

	#Define the convolve masks
	#3x3 convolve mask
	kernel_1 = np.ones((3,3), np.float32)/9
	#5x5 convolve mask
	kernel_2 = np.ones((5,5), np.float32)/25
	#7x7 convolve mask
	kernel_3 = np.ones((7,7), np.float32)/49

	#Applies the desired Kernel
	identity = cv.filter2D(src = data, ddepth = -1, kernel = kernel_2)

	plt.imshow(data)
	plt.imshow(identity)
	plt.show()

#	for r in range(0, 256, tile_size):
#		for c in range(0, 256, tile_size):
#			tile = data[r:r+tile_size, c:c+tile_size]
#			tiles.append(tile)

#			identity = cv.filter2D(src = tile, ddepth = 1, kernel = kernel_1)
#			cv.imshow(tile)
#			cv.imshow(identity)


	#This "range" is from 0 to 256 and each step forward in the loop is the tile size.
#	for r in range (0, 256, tile_size):
#		for c in range (0, 256, tile_size):
#			x_offset = c
#			y_offset = r
		        #From what I understand, this is plotting the image. (x_offset, y_offset) is the anchor point which, in this case,
#			rect = plt.Rectangle((x_offset, y_offset), tile_size, tile_size)
#			ax[1].add_patch(rect)
#	plt.show()
