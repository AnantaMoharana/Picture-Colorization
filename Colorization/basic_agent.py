import numpy as np
from numpy.lib.function_base import diff
from skimage import io, color
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

def basic_agent(left_half_training):

    #print('Basic Agent')
    centroids_with_coordinates, centroid_vals=k_means(left_half_training)

    for cluster in centroids_with_coordinates:
        pixel=cluster[0]
        for spot in cluster[2]:
            left_half_training[spot[0]][spot[1]]=pixel
    io.imshow(left_half_training) 
    io.show()

    
            



def loadImage(filename):
    with Image.open(filename).convert('RGB') as picture:
        imageWidth, imageHeight = picture.size
    pixels = picture.load()

    for i in range(int(imageWidth/2), imageWidth):
        for j in range(0,imageHeight):
            RGB_Val = int(.21*pixels[i, j][0] + .72*pixels[i, j][1] + .07*pixels[i, j][2])
            pixels[i,j] = (RGB_Val, RGB_Val, RGB_Val)
    picture.show()

def get_training_data(image): #left half of the image
    #("Getting Training Data")
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


def k_means(left_half_training): #psoibbly delete later


    #print("k_means")
    #get the first 5 ranodm centroids
    centroids=[]
    centroid_vals=[]
    for _ in range(5):
        x=random.randint(0,left_half_training.shape[0]-1)
        y=random.randint(0,left_half_training.shape[1]-1)
        print(left_half_training[x][y])
        centroids.append((left_half_training[x][y],[]))
        centroid_vals.append(left_half_training[x][y])
    
    #get the clusters assinged to each centroid
    while True:
        #print("break")

        centroids, centroid_vals=KMeans_get_clusters(left_half_training, centroids, centroid_vals)

        prior_centroid=centroid_vals

        centroids,centroid_vals=averge_recalculation(centroids, centroid_vals)

        if np.array_equal(prior_centroid, centroid_vals):
            centroids_with_coordinates=[]
            for centroids in centroid_vals:
                centroids_with_coordinates.append((centroids,[],[]))
            
            centroids_with_coordinates, centroid_vals=KMeans_get_coordinates(left_half_training, centroids_with_coordinates, centroid_vals)
            return centroids_with_coordinates, centroid_vals
            break



        
def KMeans_get_coordinates(left_half_training,centroids_with_coordinates, centroid_vals):
    #print("Running Kmeans")
   # minimum_distance=math.inf
    cluster_assignment=[-1,-1,-1]
    for i in range(left_half_training.shape[0]):
        for j in range(left_half_training.shape[1]):

            #compute the distance for each of the centroids
            minimum_distance=math.inf
            for val in centroid_vals:
                distance=color_distance(left_half_training[i][j],val)

                if distance<minimum_distance:
                    minimum_distance=distance
                    cluster_assignment=val

            for centroid in centroids_with_coordinates:
                if np.array_equal(centroid[0],cluster_assignment):
                    centroid[1].append(left_half_training[i][j])
                    centroid[2].append((i,j))
                    break
            
    return centroids_with_coordinates, centroid_vals


    #print("progress")
    


    
            



def get_testing_data(image): #right half of the image
    #print("Getting Testing Data")
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



def KMeans_get_clusters(left_half_training,centroids, centroid_vals):
    #print("Running Kmeans")
   # minimum_distance=math.inf
    cluster_assignment=[-1,-1,-1]
    for i in range(left_half_training.shape[0]):
        for j in range(left_half_training.shape[1]):

            #compute the distance for each of the centroids
            minimum_distance=math.inf
            for val in centroid_vals:
                distance=color_distance(left_half_training[i][j],val)

                if distance<minimum_distance:
                    minimum_distance=distance
                    cluster_assignment=val

            for centroid in centroids:
                if np.array_equal(centroid[0],cluster_assignment):
                    centroid[1].append(left_half_training[i][j])
                    break
            
    return centroids, centroid_vals


    #print("progress")
    
    

def averge_recalculation(centroids, centroid_vals):    #compute new Centroids
    #print("Recomputing Clusters")
    new_centers_kmeans=[]
    new_centroids=[]
    for colors  in centroids:
        counter = 0
        red=0
        green=0
        blue=0
        for rgb in colors[1]:
            red+=rgb[0]
            green+=rgb[1]
            blue+=rgb[2]
            counter+=1
        new_center=[int(red/counter),int(green/counter),int(blue/counter)]
        new_centers_kmeans.append(np.array(new_center))
        new_centroids.append((np.array(new_center),[]))
    
    return  new_centroids, new_centers_kmeans




        


def color_distance(start, end): #formula from lecture 20 notes
    #print("Color Distance From Lecture 20 Notes")

    red=2*(start[0]-end[0])**2

    green=4*(start[1]-end[1])**2

    blue=3*(start[2]-end[2])**2

    distance=math.sqrt(red+green+blue)

    return distance










if __name__ == '__main__':
    print("Main Method")
    image=io.imread('flowerpic.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')
    training_data=get_training_data(image)

    #test_vals=[[1,2,3],[7,8,3],[6,7,2]]
    #centroid_test=[([1,2,3],[[9,8,7],[8,7,6]]),([7,8,3],[[5,4,3],[2,1,3]]),([6,7,2],[[5,1,2],[3,5,6]])]

    #centroid, new_centers_kmeans=averge_recalculation(centroid_test, test_vals)
    #centroids_with_coordinates, centroid_vals=k_means(training_data)
    #print("Basic Agent Left Half time")
    #KMeans_get_clusters(training_data)
    #testing_data=get_testing_data(image)
    #print(image)
    #image=color.rgb2gray(image)
    basic_agent(training_data)
    #io.imshow(training_data) 
    #io.show()

    #io.imshow(testing_data)
    #io.show()
    
