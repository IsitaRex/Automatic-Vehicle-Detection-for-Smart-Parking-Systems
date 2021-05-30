from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import matplotlib as plt
import os
import pickle

def HoGExtraction(pickle_file_name):
    ''' 
    Input: name of the pickle file to store the features
    Output: pickle file with HoG of the images
    '''
    class_ = ["Fotos/Empty_All/" , "Fotos/Occupied_All/"]
    features = []
    labels = []
    label = -1
    for cla in class_:
        label += 1
        for fil in os.listdir(cla):
            path = cla+fil
            print(path)
            img = cv.imread(path, -1)
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
            features.append(hog_image.reshape(-1))
            labels.append(label)

    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0], 1))
    features = np.array(features)
    features = np.append(features, labels, axis = 1)

    f = open(pickle_file_name+'.pckl', 'wb')
    pickle.dump(features, f)
    f.close()

HoGExtraction("HogFeatures")