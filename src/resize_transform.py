
import cv2
import argparse
from utilities import custom_resize_image, get_img_helper,zerop_resize_image
import glob as glob
import os
from draw_boxes import draw_boxes




def custom_resize(all_images, images_path, target_width, target_height, target_dim):
    for img_name in all_images:
        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        img_helper = get_img_helper() 
        resized_image = custom_resize_image(img_original, target_width, target_height, target_dim,img_helper)
        """ cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows() """
        cv2.imwrite(f"dataaugmented/{img_name}",resized_image)
        print("fin")


def usual_resize(all_images,images_path, target_dim):
    for img_name in all_images:

        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
        """ cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows() """
        cv2.imwrite(f"dataaugmented/{img_name}",resized_image)
        print("fin")


def zerop_resize(all_images, images_path):
    for img_name in all_images:
        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        resized_image = zerop_resize_image(img_original, target_width, target_height, target_dim)
        """ cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows() """
        cv2.imwrite(f"dataaugmented/{img_name}",resized_image)
        print("fin")



parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, nargs='+', help='Image Width')
parser.add_argument('--type', type=str, default='custom', help='resize type')
args = vars(parser.parse_args())

target_width, target_height = args['size']
resize_type = args['type']
""" target_width = 700
target_height = 500 """
target_dim = (target_width, target_height)

image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
all_image_paths = []
images_path = 'dataset/train'

# get all the image paths in sorted order
for file_type in image_file_types:
    all_image_paths.extend(glob.glob(f"{images_path}/{file_type}"))

all_images = [image_path.split(os.path.sep)[-1] for image_path in all_image_paths]

if resize_type == 'custom':
    custom_resize(all_images, images_path, target_width, target_height, target_dim)
elif resize_type == 'usual':
    usual_resize(all_images,images_path, target_dim)
elif resize_type == 'zero':
    zerop_resize(all_images,images_path)






