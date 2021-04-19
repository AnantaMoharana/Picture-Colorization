import numpy as np
from numpy.lib.function_base import diff
from skimage import io, color
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def basic_agent(image):

    print('Basic Agent')


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


    def loadImage(self, filename):
        with Image.open(filename).convert('RGB') as picture:
            imageWidth, imageHeight = picture.size
        pixels = picture.load()

        for i in range(int(imageWidth/2), imageWidth):
            for j in range(0,imageHeight):
                RGB_Val = int(.21*pixels[i, j][0] + .72*pixels[i, j][1] + .07*pixels[i, j][2])
                pixels[i,j] = (RGB_Val, RGB_Val, RGB_Val)
        picture.show()




if __name__ == '__main__':
    print("Main Method")
    image=io.imread('flowerpic.jpg')
    training_data=get_training_data(image)
    testing_data=get_testing_data(image)
    #print(image)
    #image=color.rgb2gray(image)

    io.imshow(training_data) 
    io.show()

    io.imshow(testing_data)
    io.show()
    
