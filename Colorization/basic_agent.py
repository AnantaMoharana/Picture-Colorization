import numpy as np
from numpy.lib.function_base import diff
from skimage import io, color
import matplotlib.pyplot as plt
import random
import math
from queue import PriorityQueue
import collections
from PIL import Image

def basic_agent(left_half_training, right_testing_data):

    print('Basic Agent')
    b=left_half_training.shape
    print(b)
    c=right_testing_data.shape
    print(c)
    io.imshow(left_half_training) 
    io.show()
    io.imshow(right_testing_data) 
    io.show()
    right_rows=len(right_testing_data[0])
    right_columns=len(right_testing_data[1])
    Left_training_grey_scale=np.copy(left_half_training)
    set_to_grey_scale(Left_training_grey_scale)
    set_to_grey_scale(right_testing_data)
    io.imshow(right_testing_data) 
    io.show()
    print(len(right_testing_data[0]))
    print(len(right_testing_data[1]))

    centroids_with_coordinates, centroid_vals=k_means(left_half_training)
    print(centroid_vals)

    for cluster in centroids_with_coordinates:
        pixel=cluster[0]
        for spot in cluster[2]:
            left_half_training[spot[0]][spot[1]]=pixel
    io.imshow(left_half_training) 
    io.show()

    left_grey_patches=[]

    for i in range(1,b[0]-1):
        for j in range(1, b[1]-1):
            left_grey_patches.append(Left_training_grey_scale[i-1:i+2,j-1:j+2])

    #surrouind the rihgt half with black pixels
    border=[]

    for i in range(c[1]+2):
        border.append(np.array([0,0,0]))

    working_right_side=[]
    working_right_side.append(border)

    for i in range(0,c[0]):
        placements=[]
        placements.append(np.array([0,0,0]))
        for j in range(0,c[1]):
            placements.append(right_testing_data[i][j])
        placements.append(np.array([0,0,0]))
        working_right_side.append(placements)
    
    
    working_right_side.append(border)
    working_right_side=np.array(working_right_side)
    io.imshow(working_right_side) 
    io.show()
    w=right_testing_data.shape
    
    progress=0
    for i in range(1,w[0]-1):
        for j in range(1, w[1]-1):
            right_pixel_patch=right_testing_data[i-1:i+2,j-1:j+2]
            print((progress/((w[0]-1)*(w[1]-1)))*100,"% Done")
            q = PriorityQueue()
            for patch in left_grey_patches:
                similarity=get_patch_similarity(patch, right_pixel_patch)
                #print(similarity)
                q.put((similarity,(patch[1][1][0],patch[1][1][1],patch[1][1][2])))
            

            six_patches=[]

            num=6

            while num:
                six_patches.append(q.get()[1])
                num-=1
           

            occurrences = collections.Counter(six_patches)
            #print("Progress")

            color=get_most_frequent(occurrences)
            tie=check_tie(occurrences,color)

            if not tie:
                spot_color=get_appropriate_color(centroid_vals, color)
                #print(spot_color)
                right_testing_data[i][j]=spot_color
            if tie:
                color=right_testing_data[i][j]

                spot_color=[-1,-1,-1]
                mindist=math.inf
                for patch in six_patches:
                    patch2=[patch[0],patch[1],patch[2]]
                    dist=color_distance(patch2, color)
                    if dist<mindist:
                        mindist=dist
                        spot_color=patch2

                spot_color2=[-1,-1,-1]
                mindist2=math.inf
                for centroid in centroid_vals:
                    distance=color_distance(spot_color, centroid)
                    if distance<mindist2:
                        mindist2=distance
                        spot_color2=centroid
                right_testing_data[i][j]=spot_color2

            progress+=1

            if progress%1600==0:
                io.imshow(right_testing_data) 
                io.show()


    io.imshow(right_testing_data) 
    io.show()

    #some cod3 to join th3 two parts togehter



            



    #prep the right half by adding a border to the array
def get_appropriate_color(centroid_vals,color):
    spot=[color[0],color[1],color[2]]
    dist=math.inf
    center_val=[-1,-1,-1]
    for center in centroid_vals:
        similarity=color_distance(spot, center)
        if similarity<dist:
            dist=similarity
            center_val=center

    return center_val

def get_most_frequent(occurences):
    get_most_frequent=(-1,-1,-1)
    freq=-1
    
    for key in occurences.keys():
        if occurences[key]>freq:
            freq=occurences[key]
            get_most_frequent=key
    return get_most_frequent
def check_tie(occurrences,color):
    tie=False
    for key in occurrences.keys():
        if key==color:
            continue
        if occurrences[color]==occurrences[key]:
            tie=True
            return tie
    return tie

    print("Progress")
def get_patch_similarity(left, right):
    similarity=0
    for i in range(0,len(left)):
        for j in range(0,len(left)):
            similarity+=color_distance(left[i][j], right[i][j])

    return similarity
    #print("Implement")
            
def set_to_grey_scale(image_data):
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            pixel=image_data[i][j]
            red=pixel[0]
            green=pixel[1]
            blue=pixel[2]

            RGB=int((.21*red)+(.72*green)+(.07*blue))

            image_data[i][j]=[RGB,RGB,RGB]


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

        if centroid_verification(prior_centroid, centroid_vals):
            centroids_with_coordinates=[]
            for centroids in centroid_vals:
                centroids_with_coordinates.append((centroids,[],[]))
            
            centroids_with_coordinates, centroid_vals=KMeans_get_coordinates(left_half_training, centroids_with_coordinates, centroid_vals)
            return centroids_with_coordinates, centroid_vals
            break

def centroid_verification(prior_centroid,centroid_vals):
    verified=True

    for i in range(0,5):
        for j in range(0,3):
            difference=abs(prior_centroid[i][j]-centroid_vals[i][j])
            if difference>=5:
                verified=False
                return verified
    return verified



        
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

    return int(distance)










if __name__ == '__main__':
    print("Main Method")
    image=io.imread('smaller_flower.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')
    training_data=get_training_data(image)
    testing_data=get_testing_data(image)

    #test_vals=[[1,2,3],[7,8,3],[6,7,2]]
    #centroid_test=[([1,2,3],[[9,8,7],[8,7,6]]),([7,8,3],[[5,4,3],[2,1,3]]),([6,7,2],[[5,1,2],[3,5,6]])]

    #centroid, new_centers_kmeans=averge_recalculation(centroid_test, test_vals)
    #centroids_with_coordinates, centroid_vals=k_means(training_data)
    #print("Basic Agent Left Half time")
    #KMeans_get_clusters(training_data)
    #testing_data=get_testing_data(image)
    #print(image)
    #image=color.rgb2gray(image)
    basic_agent(training_data,testing_data)
    #io.imshow(training_data) 
    #io.show()

    #io.imshow(testing_data)
    #io.show()