import cv2 as cv
import torch
from torch import optim, nn
from torchvision import models, transforms
import numpy as np
import matplotlib as plt
import os
import pickle

#This code is based on: https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a

class FeatureExtractor(nn.Module):

  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
	# Extract Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
	# Extract Average Pooling Layer
    self.pooling = model.avgpool
	# Flatten vector
    self.flatten = nn.Flatten()
  
  def forward(self, x):
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    return out 


def featureExtraction(model, pckl_file_name):
    '''
    Input: 
           model is the Neural Network model 
           pckl_file_name is the pickle file name where the np array of features is going to be saved for later use (the last column are the labels)
    Output:
            this function returns the np array of features and additionally saves a pickle file
    '''

    # Initialize the model
    new_model = FeatureExtractor(model)

    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    new_model = new_model.to(device)

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((128,64)),
    transforms.ToTensor()                              
    ])

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
            img = transform(img) 
            width = img.shape[1]
            height = img.shape[2]
            img = img.reshape(1, 3, width, height)
            img = img.to(device)
            with torch.no_grad():
                feature = new_model(img)
            features.append(feature.cpu().detach().numpy().reshape(-1))
            labels.append(label)

    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0], 1))
    features = np.array(features)
    print(labels.shape, features.shape)
    features = np.append(features, labels, axis = 1)
    f = open(pckl_file_name + '.pckl', 'wb')
    pickle.dump(features, f)
    f.close()
    return features

model1 = models.vgg16(pretrained=True)
model2 = models.alexnet(pretrained=True)
featureExtraction(model1, 'Image_featuresVGG16')
featureExtraction(model2, 'Image_featuresAlexNet')

# infile = open('Image_featuresVGG162.pckl','rb')
# features = pickle.load(infile)
# infile.close()