from turtle import right
from skimage import io, transform
import cv2
import numpy as np
import random


def crop (width, height, image):
    h2, w2 = image.shape[:2]
    ini_h = random_point(h2-height)
    ini_w = random_point(w2-width)
    crop_image = image[ini_h:height+ini_h, ini_w:width+ini_w]
    return crop_image


def composeh_image (image_base,image_add):

    h1,w1,h2,w2,first_image,second_image = get_shape_img(image_base,image_add)
    resize_image = np.zeros((h1+h2,max(w1,w2),  3), np.uint8)
    resize_image[:h1, :w1,:3] = first_image
    resize_image[h1:h1+h2, :w2,:3] = second_image
    return resize_image

def composev_image (image_base,image_add):

    h1,w1,h2,w2,first_image,second_image = get_shape_img(image_base,image_add)
    resize_image = np.zeros((max(h1,h2), w1+w2,  3), np.uint8)
    resize_image[:h1, :w1,:3] = first_image
    resize_image[:h2, w1:w1+w2,:3] = second_image
    return resize_image

def get_shape_img(image_base, image_add):
    original_first = bool(random.getrandbits(1))
    if original_first:
            h1, w1 = image_base.shape[:2]
            h2, w2 = image_add.shape[:2]
            first_image = image_base
            second_image = image_add
    else:
        h1, w1 = image_add.shape[:2]
        h2, w2 = image_base.shape[:2]
        first_image = image_add
        second_image = image_base  
    
    return h1,w1,h2,w2,first_image,second_image

def merge_images(helper_backgroud, image_base, target_width, target_height):
    resized_helper = cv2.resize(helper_backgroud, (target_width, target_height), interpolation = cv2.INTER_AREA)
    h2, w2 = image_base.shape[:2]
    ini_h = random_point(target_height-h2)
    ini_w = random_point(target_width-w2)
    resized_helper[ini_h:h2+ini_h,ini_w:w2+ini_w,:] = image_base[0:h2,0:w2,:]
    return resized_helper


def random_point(max_value):
    value = random.randint(0,max_value)
    return value

def custom_resize_image(img_original, target_width,target_height,target_dim, img_helper):
    height, width, channels = img_original.shape 
    print("Size: "+str(width)+" x "+str(height))
    resized_image = img_original

    if width > target_width:
        resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
    elif width < target_width:
        if height > target_height:
            resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
        elif height < target_height:
            resized_image = merge_images(img_helper, img_original, target_width, target_height)
        else:
            extra_width = target_width - width
            crop_image = crop (extra_width, target_height, img_helper)
            resized_image = composev_image(img_original,crop_image)
    else:
        if height > target_height:
            resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
        elif height < target_height:
            extra_heigth = target_height - height
            crop_image = crop (target_width, extra_heigth, img_helper) 
            resized_image = composeh_image(img_original,crop_image)

    return resized_image  

def get_img_helper():
    value = random_point(9)
    img_helper = cv2.imread(f"dataset/helpers/road_{value}.jpg")   
    return img_helper

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


def complete_horizontal(size):
    complete_left = bool(random.getrandbits(1))
    left = 0
    rigth = 0
    if complete_left:
        left = size
    else:
        rigth = size
    
    return left,rigth

def complete_vertical(size):
    complete_top = bool(random.getrandbits(1))
    top = 0
    bottom = 0
    if complete_top:
        top = size
    else:
        bottom = size
    
    return top,bottom

def complete_random_border(width, height , target_width, target_height):
    left = random_point(target_width - width)
    rigth = target_width - left
    top = random_point(target_height -height)
    bottom = target_height - top

    return top, bottom, left, rigth

def zerop_resize_image(img_original, target_width,target_height,target_dim):
    height, width, channels = img_original.shape 
    print("Size: "+str(width)+" x "+str(height))
    resized_image = img_original
    color = [0, 0, 0]

    if width > target_width:
        resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
    elif width < target_width:
        if height > target_height:
            resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
        elif height < target_height:
            top, bottom, left, right = complete_random_border(width, height, target_width, target_height)
            resized_image = cv2.copyMakeBorder(img_original, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        else:
            extra_width = target_width - width
            left, right = complete_horizontal(extra_width)
            resized_image = cv2.copyMakeBorder(img_original, 0, 0, left, right, cv2.BORDER_CONSTANT,value=color)
    else:
        if height > target_height:
            resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
        elif height < target_height:
            extra_heigth = target_height - height
            top, bottom = complete_vertical(extra_heigth)
            resized_image = cv2.copyMakeBorder(img_original, top, 0, bottom, 0, cv2.BORDER_CONSTANT,value=color)
            

    return resized_image  