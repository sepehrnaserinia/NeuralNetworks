import os
from os.path import join

import cv2
import numpy as np
from PIL import Image

# Image Edit function(read, convert to Gray, Gap Fill, resize size*size(64), return img)
def image_to_normalized_square(img, size=64) :
    # convert to 1 channel
    if (img.ndim == 3) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #record the original dimensions
    width = img.shape[0]
    height = img.shape[1]
    # finding the biggest dimension
    if width >= height:
        length = img.shape[0]
    else :
        length = img.shape[1]
    # create a square matrix, the size of our biggest dimension
    im_result = np.ones((length, length),  dtype = np.uint8) * 255
    wd = int((length - width) / 2)
    hd = int((length - height) / 2)
    # fill the data from both sides, inorder to have the image in the middle & keep the rest unchanged
    im_result[wd:wd + width, hd:hd + height] = img
    # resize to size*size
    im_result = cv2.resize(im_result, (size, size), interpolation=cv2.INTER_AREA)
    return im_result

def mean_block_feature(img):
    block_size = 5
    output_size = 64
    img_result = np.empty((0, output_size))
    for image in img:
        image = extract_mean_block_feature(image, block_size)
        image = image.reshape((1, output_size))
        img_result = np.append(img_result, image, axis=0)
    return img_result

def extract_mean_block_feature(img, block_size):
    if img.shape[0] != img.shape[1]:
        return

    block_num = img.shape[0] // block_size
    img_features = np.empty((0))
    for row in range(block_num):
        for col in range(block_num):
            row_from, row_to = (block_size * row, block_size * (row + 1))
            col_from, col_to = (block_size * col, block_size * (col + 1))
            img_features = np.append(img_features, np.mean(img[row_from:row_to, col_from:col_to]))
    return img_features


def connected_components(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, im_bin = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centriods = cv2.connectedComponentsWithStats(im_bin, None, None, None, 8, cv2.CV_32S)
    im_rgb = 255 - cv2.cvtColor(im_bin, cv2.COLOR_GRAY2RGB)
    return nlabels, stats, im_rgb

def get_stat(index, stats):
    left = stats[index, cv2.CC_STAT_LEFT]
    top = stats[index, cv2.CC_STAT_TOP]
    right = stats[index, cv2.CC_STAT_WIDTH] + left
    bottom = stats[index, cv2.CC_STAT_HEIGHT] + top
    return left, top, right, bottom


def img_to_pdf(images_path, pdf_name, pdf_path):
    if not os.path.isdir(pdf_path):
        os.makedirs(pdf_path)

    imgList = []
    for img_path in images_path[1:]:
        img = Image.open(img_path, 'r')
        img = img.convert('RGB')
        imgList.append(img)

    img = Image.open(images_path[0], 'r')
    img = img.convert('RGB')
    img.save(join(pdf_path, pdf_name + ".pdf"), save_all=True, append_images=imgList)

def mergeImg(img_paths, save_filename):
    img1 = cv2.imread(img_paths[0])
    img2 = cv2.imread(img_paths[1])
    y, x, c = img1.shape
    shape = (y, 2*x, c)
    img = np.zeros(shape=shape, dtype=np.uint8)
    img[:, :x, :] = img2
    img[:, x:2*x, :] = img1
    cv2.imwrite(save_filename, img)

if __name__ == "__main__":
    print("Header File...")
