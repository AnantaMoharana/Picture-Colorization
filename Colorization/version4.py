import numpy as np
from skimage import io, color
import math
import random

def improved_agent(leftHalfColor, leftHalfGrey, rightHalfGrey):


    training_set=get_model_training_set(leftHalfColor,leftHalfGrey)



    #print("Progress")
    #Initialize the weights using wah
    weights1=np.random.normal(0,0.5,(5, 9)) 

    weights2=np.random.normal(0,0.5,(3, 5))
    adj1_total = np.zeros((3,5),dtype = float)
    adj2_total = np.zeros((5,9),dtype = float)
    i=0
    epochs=3000
    for e in range(epochs):
        total_error_sum=0
        for train in training_set:



            input_layer=train[0]
            

            actual_output=[[train[1][0]/255],[train[1][1]/255],[train[1][2]/255]]
            actual_output=np.array(actual_output)

            #print(weights1.shape)
            #print(weights2.shape)
            #print(input_layer.shape)

            hidden,output=forward_propagation(input_layer,weights1,weights2)

            #print(output.shape)

            cost=np.sum(np.square(output,actual_output))
            total_error_sum+=cost
            
            #compute the derivative between the output and hidden derivatives
            deriv1=output_hidden_derivative(hidden,output,actual_output,weights2)
            adj1_total+=deriv1
            deriv2=calc_cost_deriv_2(input_layer,hidden,output,actual_output,weights2,weights1)
            adj2_total+=deriv2
        
        #print("Epoch :")
        print("Epoch and Error:",e,total_error_sum)
            
        adj1_total=adj1_total * (1/len(training_set))
        adj2_total=adj2_total * (1/len(training_set))

        weights1=weights1 - 0.003*adj2_total
        weights2=weights2 - 0.003*adj1_total

    colorpic(rightHalfGrey, weights1, weights2)



        #print("progress",i)
        #i+=1

def input_hidden_derivative(input_layer,hidden,output,actual_output,weights1,weights2):

    sigmoid_wight1_and_intput=sigmoid_derivative(np.dot(weights1,input_layer.T))

    part1=np.dot(sigmoid_wight1_and_intput,input_layer)

    sumation= 2*np.subtract(output,actual_output)*sigmoid_derivative(np.dot(weights2,hidden))

    part2=np.dot(weights2.T,sumation)

    a = np.zeros((5, 5))

    part2=np.fill_diagonal(a,part2)

    part2=a

    derivative=np.dot(part2,part1)

    return derivative

    
def output_hidden_derivative(hidden,output,actual_output,weights2):
    z=np.dot(weights2,hidden)
    z=sigmoid_derivative(z)
    z2=2*np.subtract(output,actual_output.T)
    
    dot=np.dot(z2,z)

    derivative=np.dot(dot,hidden.T)

    return derivative


def forward_propagation(inputs, weights1, weights2):

    z=np.dot(weights1,inputs.T)

    hidden_layer=sigmoid(z)

    z2=np.dot(weights2,hidden_layer)

    output_layer=sigmoid(z2)

    return  hidden_layer,output_layer




    #print("Progress")

def backward_propragation():
    print("BackWard")





def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    f = 1/(1+np.exp(-x))
    df = f * (1 - f)
    return df
    #print("Progress")


def get_model_training_set(color,leftHalfGrey):
    io.imshow(color)
    io.show()   

    io.imshow(leftHalfGrey)
    io.show()   
    print(color[1][1])


    training_set=[]
    for x in range(1, color.shape[0]-1):
        for y in range(1, color.shape[1]-1):
            #z=leftHalfGrey[x-1:x+2,y-1:y+2].flatten()
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
            #print(actual)
            patches=[
                [upperLeft[0] /255, upperMid[0] /255, upperRight[0] /255, midLeft[0] /255, midRight[0] /255, lowerRight[0] /255, lowerMid[0] /255, lowerLeft[0] /255, mid[0]/255 ]
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

            RGB = (.21 * red) + (.72 * green) + (.07 * blue)

            image_data[i][j] = RGB

def colorpic(rightHalfGrey,weights1,weights2):

    #out=np.copy(rightHalfGrey)
    for x in range(1, rightHalfGrey.shape[0]-1):
        for y in range(1, rightHalfGrey.shape[1]-1):

            midLeft = rightHalfGrey[x - 1][y]
            midRight = rightHalfGrey[x + 1][y]
            upperMid = rightHalfGrey[x][y + 1]
            lowerMid = rightHalfGrey[x][y - 1]
            lowerRight = rightHalfGrey[x + 1][y + 1]
            upperLeft = rightHalfGrey[x - 1][y - 1]
            upperRight = rightHalfGrey[x + 1][y - 1]
            lowerLeft = rightHalfGrey[x - 1][y + 1]
            mid = rightHalfGrey[x][y]

            #actual=color[x][y]
            patches=[
                [upperLeft[0] /255 , upperMid[0]  /255, upperRight[0] /255, midLeft[0]/255 , midRight[0] /255, lowerRight[0] /255, lowerMid[0] /255, lowerLeft[0] /255, mid[0]/255 ]
            ]

            first_layer, output=forward_propagation(np.array(patches), weights1, weights2)

            #out[x][y]=[int(output[0]*255),int(output[1]*255),int(output[2]*255)]
            rightHalfGrey[x][y]=[int(output[0]*255),int(output[1]*255),int(output[2]*255)]
            #print(rightHalfGrey[x][y])

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
    print(testing_data[1][1])
    set_to_grey_scale(testing_data)

    leftHalfGreyScale = np.copy(training_data)
    set_to_grey_scale(leftHalfGreyScale)

    # Run the improved agent code
    improved_agent(training_data, leftHalfGreyScale, testing_data)