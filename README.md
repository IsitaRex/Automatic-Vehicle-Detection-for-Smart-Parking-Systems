# Automatic-Vehicle-Detection-for-Smart-Parking-Systems
The general objective of this project is to answer the question: Is it possible to develop a video-based automatic vehicle detection system by using machine- and deep-learning techniques? Which are the main difficulties to overcome for developing a system that can provide near 100% accuracy while keeping low installation and maintenance costs?. Initially, we focused on the parking space vacancy detection problem using our own images of EAFIT's University south parking lot. Solving this problem enables providing turn-by-turn guidance inside the parking lots via digital signage for all users as well as providing a parking spot vacancy map for a mobile app, so that users can choose their parking spot even before entering the parking lot. 

## Data-Set
We built our own data set available at: https://1drv.ms/u/s!Ar7c_geveKL2koApDpxAVhYsQlF99g?e=vtnh0u

The images were taken with a drone from different perspectives and then cropped in order to buil a single parking space data-set. Two classes can be found: empty parking spaces and occupied ones. Additionally, seeking to increase the data-set we performed data augmentation techniques by rotating the image, flipping it and applying both transformations. Finally we obtained 1166 images of the empty class and 1088 occupied cells, for a total of 2254 images.

## Feature Extraction
Since working with raw pixels has proved not to be optimal for image classification problems, we decided to explore two different feature extraction method looking to improve our generalization task. The first one is separating the feature extraction part from a VGG16 from the multilayer perceptron and using this flattened features as one feature set. On the other hand, the other technique is extracting the histogram of oriented gradients from our data, which is widely used in for object detection purposes and our problem can be translated into detecting a car inside a parking space or not.

* [Feature Extraction with VGG16](https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a)

## Notebooks
There is a notebook for each feature set (raw pixels, VGG16 and HoG) showing the classification results and the confusion matrices for each classifier. In order to validate the results we split the data into 80% training and 20% testing and 5-fold crossvalidation. Additionally, looking for lower complexity in our models we applied the Barnes hut-tsne as dimensionality technique to measure the performance in lower dimensions.

Classifiers used:
* Logistic Regression
* SVM (linear, polynomial and radial base)
* Random Forest
* Gradient Boosting
