# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-n", "--name", required=True,
	help="name to output files")
ap.add_argument("-t", "--total", type=int, default=3,
	help="# of training samples to generate")
args = vars(ap.parse_args())


def generator(image, name):

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	# construct the actual Python generator
	imageGen = aug.flow(
	    image,
	    batch_size=1,
	    save_to_dir=args["output"],
		save_prefix=os.path.splitext(name)[0],
		save_format="jpg")

		# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1
		# if we have reached the specified number of examples, break
		# from the loop
		if total == args["total"]:
			break

def rename():
	i = 0
	for filename in os.listdir(args["output"]):
		dst = args["name"] + str(i) + ".jpg"
		src = args["output"] + filename
		dst = args["output"] + dst
		os.rename(src, dst)
		i += 1
	print('[INFO] Rename %d images...' % i)

if __name__ == '__main__':
	# load the input image, convert it to a NumPy array, and then
	# reshape it to have an extra dimension
	print("[INFO] loading example image...")

	path = args["image"]
	print('[INFO] Path to images: ', path)
	fds = sorted(os.listdir(path))
	print('[INFO] Sorted images: ', fds)
	for img in fds:
		i = str(path) + str(img)
		print('[INFO] image: ', i)
		image = load_img(i)
		image = img_to_array(image)
		image = np.expand_dims(image, axis = 0)
		name = args["name"]
		generator(image, name)

	# rename images in output create_directory
	rename()
