import cv2
import argparse
from utilities import custom_resize_image, get_img_helper,zerop_resize_image, usual_resize_image, get_tree_box
import glob as glob
import os
from draw_boxes import draw_boxes
from xml.etree import ElementTree as et
import numpy as np



def print_image (img, boxes):
    # draw the bounding boxes on the tranformed/augmented image
    annot_image, box_areas = draw_boxes(
        img, boxes, 'voc'
        )
    """ cv2.imshow('Image', annot_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows() """
    return annot_image

def custom_resize(all_images, images_path, target_width, target_height, target_dim):
    for img_name in all_images:
        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB).astype(np.float32)
        root = get_tree_box(img_name,images_path)
        img_helper = get_img_helper()
        resized_image, boxes = custom_resize_image(img_original, target_width, target_height,img_helper, root)
        annot_img = print_image(resized_image, boxes)
        cv2.imwrite(f"dataaugmented/{img_name}",annot_img)
        print("fin")
        


def usual_resize(all_images,images_path, target_width, target_height ):
    for img_name in all_images:

        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB).astype(np.float32)
        root = get_tree_box(img_name,images_path)
        resized_image, boxes = usual_resize_image(img_original, root, target_width, target_height)
        annot_img = print_image(resized_image, boxes)
        cv2.imwrite(f"dataaugmented/{img_name}",annot_img)
        print("fin")


def zerop_resize(all_images, images_path):
    for img_name in all_images:
        image_path = os.path.join(images_path, img_name)
        img_original = cv2.imread(image_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB).astype(np.float32)
        root = get_tree_box(img_name,images_path)
        resized_image, boxes = zerop_resize_image(img_original, target_width, target_height, root)
        annot_img = print_image(resized_image, boxes)
        cv2.imwrite(f"dataaugmented/{img_name}",annot_img)
        print("fin")



parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, nargs='+', help='Image Width')
parser.add_argument('--type', type=str, default='usual', help='resize type')
args = vars(parser.parse_args())

target_width, target_height = args['size']
resize_type = args['type']
""" resize_type = "zero"
target_width = 940
target_height = 427  """
target_dim = (target_width, target_height)
print(target_dim)

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
    usual_resize(all_images,images_path, target_width, target_height)
elif resize_type == 'zero':
    zerop_resize(all_images,images_path)






