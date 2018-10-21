import cv2
import numpy as np
from os import listdir


def partition_images():
	# Extracts all the kernels from every image in the images folder.

	output_file_id = 1
	
	def get_image_names():
		# Gets all of the filenames for images in the image path.
		return sorted([f for f in listdir('./newimages')])
	

	image_names = get_image_names()
	# print(image_names[0])
	for image_name in image_names:
		image = cv2.imread('newimages/' + image_name)
		(height, width) = image.shape[:2]
		# print('the image is', height, 'high and', width, 'wide')
		kernel_height = height // 10
		kernel_width = width // 10

		for y in range(10):
			for x in range(10):
				print(output_file_id, "- Extracting image around (", kernel_height*y, ",", kernel_width*x, ")")
				roi = image[kernel_height*y:kernel_height*(y+1),kernel_width*x:kernel_width*(x+1)]
				cv2.imwrite('newkernels/' + str(output_file_id) + '.jpg', roi)
				output_file_id += 1

partition_images()
