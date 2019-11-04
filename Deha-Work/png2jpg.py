### convert all .png images to .jpg images
import glob
import os 
import cv2

img_dir = './motobike'
for png_img in os.listdir(img_dir):
	if (png_img[-4:] == ".png"):
		img_path = os.path.join(img_dir, png_img)
		img = cv2.imread(img_path)
		dst_img = png_img[:-3] + 'jpg'
		cv2.imwrite(os.path.join(img_dir, dst_img), img)
		

 
