import os
from os import path, listdir, system
from os.path import isdir, join

import cv2
import numpy as np
from imgUtil import image_to_normalized_square
from fileUtil import read_matrix

def load_database(path):
    file = open(path, 'r')
    data = read_matrix(file)
    label = read_matrix(file)
    file.close()
    return (data, label)

def load_database_data(data_path, img_output_size):
    classes = [s for s in listdir(data_path) if isdir(join(data_path, s))]
    classes.sort()

    datasets = []
    for class_name in classes:
        classDir = join(data_path, class_name)
        class_dataset = [s for s in listdir(classDir) if not isdir(join(classDir, s))]
        datasets.append(class_dataset)

    # creating an empty array with three dimentions so that edited images
    # are added using np.append() function
    train_images = np.empty((0, img_output_size, img_output_size), dtype=np.uint8)
    test_images = np.empty((0, img_output_size, img_output_size), dtype=np.uint8) 
    train_labels = []
    test_labels = []

    for i, dataset in enumerate(datasets):
        class_path = join(data_path, "{0}".format(i))
        for j, filename in enumerate(dataset):
            file_path = join(class_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = image_to_normalized_square(img, size=img_output_size)
            
            # current image shape ---> (img_output_size, img_output_size)
            img = img.reshape((1, img_output_size, img_output_size))
            # adds to number of dimentions for np.append()
            # image shape after reshape ---> (1, img_output_size, img_output_size)
            if j < 0.6 * len(dataset):
                train_images = np.append(train_images, img, axis=0)
                train_labels.append(i)
            else:
                test_images = np.append(test_images, img, axis=0)
                test_labels.append(i)

    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_images, train_labels, test_images, test_labels

def load_cifar10(data_path):
    def my_unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        data = np.reshape(dic[b'data'], (10000, 3, 32, 32))
        data = np.swapaxes(data, 1, 2)
        data = np.swapaxes(data, 2, 3)
        label = np.array(dic[b'labels'])
        return data, label

    train = []
    train_label = []
    files = listdir(data_path)
    for file in files[1:6]:
        data, label = my_unpickle(join(data_path, file))
        train.append(data)
        train_label.append(label)
    train = np.concatenate(train)
    train = train.astype('float32') / 255
    train_label = np.concatenate(train_label)

    file = files[7]
    test, test_label = my_unpickle(join(data_path, file))
    test = test.astype('float32') / 255

    return train, train_label, test, test_label

if __name__ == "__main__":
    print("Header File...")
