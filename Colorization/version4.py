import numpy as np
from skimage import io, color
import math
import random

def improved_agent(leftHalfColor, leftHalfGrey, rightHalfGrey):


    training_set=get_model_training_set(leftHalfColor,leftHalfGrey)



    #print("Progress")
    weights1=np.random.normal(0,0.5,(9, 3))

    weights2=np.random.normal(0,0.5,(3, 3))


    for _ in range(1500):
        training_outputs=[]
        actual_vals=[]
        training_values=[]
        for training in training_set:

            input_layer=training[0]

            actual_value=training[1]/255

            z,z2,z3,z4forward_propagation(input_layer, weights1,weights2)


        
    #print("Progress")



        








def forward_propagation(inputs, weights1, weights2):

    z=np.dot(inputs,weights1)

    z2=sigmoid(z)

    z3=np.dot(z2,weights2)

    output=sigmoid(z3)

    return  z, z2, z3, output




    #print("Progress")

def backward_propragation():





def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    f = 1/(1+np.exp(-x))
    df = f * (1 - f)
    return df
    #print("Progress")


def get_model_training_set(color,leftHalfGrey):


    training_set=[]
    for x in range(1, color.shape[0]-1):
        for y in range(1, color.shape[1]-1):

            midRight = leftHalfGrey[x + 1][y]
            midLeft = leftHalfGrey[x - 1][y]
            upperMid = leftHalfGrey[x][y + 1]
            lowerMid = leftHalfGrey[x][y - 1]
            lowerRight = leftHalfGrey[x + 1][y + 1]
            upperLeft = leftHalfGrey[x - 1][y - 1]
            upperRight = leftHalfGrey[x + 1][y - 1]
            lowerLeft = leftHalfGrey[x - 1][y + 1]
            mid = leftHalfGrey[x][y]

            actual=color[x][y]
            patches=[
                [upperLeft[0] /255, upperMid[0] /255, upperRight[0] /255, midLeft[0] /255, midRight[0] /255, lowerRight[0] /255, lowerMid[0] /255, lowerLeft[0] /255, mid[0]/255 ],
                 [upperLeft[1]/255 , upperMid[1]/255 , upperRight[1]/255 , midLeft[1]/255 , midRight[1]/255 , lowerRight[1]/255 , lowerMid[1]/255 , lowerLeft[1]/255 , mid[1]/255 ],
                 [upperLeft[2] /255, upperMid[2] /255, upperRight[2] /255, midLeft[2] /255, midRight[2] /255, lowerRight[2] /255, lowerMid[2] /255, lowerLeft[2] /255, mid[2]/255 ]
            ]

            training_set.append((np.array(patches),np.array(actual)))

    return training_set


    #print("Get Training")

def get_training_data(image):  # left half of the image
    row = image.shape[0]
    column = int(image.shape[1] / 2)
    train_rows = []

    for i in range(row):
        train_columns = []
        for j in range(column):
            train_columns.append(image[i][j])
        train_rows.append(train_columns)

    training_data = np.array(train_rows)
    return training_data


def get_testing_data(image):  # right half of the image
    row = image.shape[0]
    column = int(image.shape[1] / 2) + 1
    test_rows = []

    for i in range(row):
        test_columns = []
        for j in range(column, image.shape[1]):
            test_columns.append(image[i][j])
        test_rows.append(test_columns)

    testing_data = np.array(test_rows)
    return testing_data


def set_to_grey_scale(image_data):  # de-color an image
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            pixel = image_data[i][j]
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]

            RGB = int((.21 * red) + (.72 * green) + (.07 * blue))

            image_data[i][j] = [RGB, RGB, RGB]

def colorpic(rightHalfGrey,weights1,weights2):
    for x in range(1, rightHalfGrey.shape[0]-1):
        for y in range(1, rightHalfGrey.shape[1]-1):

            midRight = rightHalfGrey[x + 1][y]
            midLeft = rightHalfGrey[x - 1][y]
            upperMid = rightHalfGrey[x][y + 1]
            lowerMid = rightHalfGrey[x][y - 1]
            lowerRight = rightHalfGrey[x + 1][y + 1]
            upperLeft = rightHalfGrey[x - 1][y - 1]
            upperRight = rightHalfGrey[x + 1][y - 1]
            lowerLeft = rightHalfGrey[x - 1][y + 1]
            mid = rightHalfGrey[x][y]

            #actual=color[x][y]
            patches=[
                [upperLeft[0] /255 , upperMid[0]  /255, upperRight[0] /255, midLeft[0]/255 , midRight[0] /255, lowerRight[0] /255, lowerMid[0] /255, lowerLeft[0] /255, mid[0]/255 ],
                 [upperLeft[1] /255 , upperMid[1]  /255, upperRight[1] /255, midLeft[1]/255 , midRight[1] /255, lowerRight[1] /255, lowerMid[1] /255, lowerLeft[1] /255, mid[1]/255 ],
                 [upperLeft[2] /255 , upperMid[2]  /255, upperRight[2] /255, midLeft[2]/255 , midRight[2] /255, lowerRight[2] /255, lowerMid[2] /255, lowerLeft[2] /255, mid[2]/255 ]
            ]

            first_layer, output=forward_propagation(np.array(patches), weights1, weights2)

            rightHalfGrey[x][y]=output[0]*255

            #training_set.append((np.array(patches),np.array(actual)))

    io.imshow(rightHalfGrey)
    io.show()        

#def weight_derivative(actual, predicted,training_set):
    #return -2*np.dot(training_set,np.sum(np.subtract(actual,predicted)))/len(predicted)

if __name__ == '__main__':


    image = io.imread('40x40flower.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')

    training_data = get_training_data(image)
    testing_data = get_testing_data(image)

    leftHalfGreyScale = np.copy(training_data)
    set_to_grey_scale(leftHalfGreyScale)

    # Run the improved agent code
    improved_agent(training_data, leftHalfGreyScale, testing_data)