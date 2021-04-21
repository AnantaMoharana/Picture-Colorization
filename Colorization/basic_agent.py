import numpy as np
from numpy.lib.function_base import diff
from skimage import io, color
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

def basic_agent(image):

    print('Basic Agent')

def loadImage(self, filename):
    with Image.open(filename).convert('RGB') as picture:
        imageWidth, imageHeight = picture.size
    pixels = picture.load()

    for i in range(int(imageWidth/2), imageWidth):
        for j in range(0,imageHeight):
            RGB_Val = int(.21*pixels[i, j][0] + .72*pixels[i, j][1] + .07*pixels[i, j][2])
            pixels[i,j] = (RGB_Val, RGB_Val, RGB_Val)
    picture.show()

def get_training_data(image): #left half of the image
    print("Getting Training Data")
    row=image.shape[0] 
    column=int(image.shape[1]/2)
    print(column)
    train_rows=[]
    
    for i in range(row):
        train_columns=[]
        #print(i)
        for j in range(column):
            #print(j)
            train_columns.append(image[i][j])
        train_rows.append(train_columns)

    training_data=np.array(train_rows)

    return training_data


def k_means():
    print("k_means")



def get_testing_data(image): #right half of the image
    print("Getting Testing Data")
    row=image.shape[0] 
    column=int(image.shape[1]/2)+1
    test_rows=[]
    
    for i in range(row):
        test_columns=[]
        for j in range(column,image.shape[1]):
            #print("working")
            test_columns.append(image[i][j])
        test_rows.append(test_columns)

    testing_data=np.array(test_rows)

    return testing_data



def Kmeans_get_centroids(left_half_training):
    print("Running Kmeans")

    #pick five random centers to use as the initial centroids
    #centroid coordinates 

    centroids=[]
    centroid_vals=[]
    for _ in range(5):
        x=random.randint(0,left_half_training.shape[0]-1)
        y=random.randint(0,left_half_training.shape[1]-1)
        print(left_half_training[x][y])
        centroids.append((left_half_training[x][y],[]))
        centroid_vals.append(left_half_training[x][y])

    minimum_distance=math.inf
    cluster_assignment=[-1,-1,-1]
    for i in range(left_half_training.shape[0]):
        for j in range(left_half_training.shape[1]):

            #compute the distance for each of the centroids

            for val in centroid_vals:
                distance=color_distance(left_half_training[i][j],val)

                if distance<minimum_distance:
                    minimum_distance=distance
                    cluster_assignment=val

            for centroid in centroids:
                if np.array_equal(centroid[0],cluster_assignment):
                    centroid[1].append
                    break
            
    



    print("progress")
    
    

def averge_recalculation(centroid):
    print("Recomputing Clusters")


def color_distance(start, end): #formula from lecture 20 notes
    #print("Color Distance From Lecture 20 Notes")

    red=2*(start[0]-start[0])**2

    green=4*(start[1]-start[1])**2

    blue=3*(start[2]-start[2])**2

    distance=math.sqrt(red+green+blue)

    return distance










if __name__ == '__main__':
    print("Main Method")
    image=io.imread('flowerpic.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')
    training_data=get_training_data(image)

    Kmeans_get_centroids(training_data)
    #testing_data=get_testing_data(image)
    #print(image)
    #image=color.rgb2gray(image)

    #io.imshow(training_data) 
    io.show()

    #io.imshow(testing_data)
    #io.show()
    
