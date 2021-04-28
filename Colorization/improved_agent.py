import numpy as np
from skimage import io, color
import math
import random



def improved_agent(leftHalfColor, leftHalfGrey, rightHalfGrey):

    # Get size values for easy access
    rightHalfSize=rightHalfGrey.shape
    rightRows=len(rightHalfGrey[0])
    rightColumns=len(rightHalfGrey[1])

    leftHalfSize = leftHalfGrey.shape
    leftRows=len(leftHalfGrey[0])
    leftColumns=len(leftHalfGrey[0])


    # Define a neural network
    inputLayer = np.array(
        [[0] * 24]
    )
    outputLayer = np.array(
        [[0] * 3]
    )
   # hiddenLayer1_weights = np.array(
   #     [random.random()*24]
   # )
    hiddenLayer1_bias = np.array(
        [random.random(),random.random(),random.random()]
    )
    #hiddenLayer2_weights = np.array(
    #    [random.random()*3]
    #)
    hiddenLayer2_bias = np.array(
        [[random.random()]*3]
    )



    # Loop through entire left side (this is TRAINING)
    for x in range(1, leftRows-1):
        for y in range(1, leftColumns-1):

            # Find our 8 surrounding squares (B&W) to train with
            midRight=leftHalfGrey[x+1][y]
            midLeft=leftHalfGrey[x-1][y]
            upperMid=leftHalfGrey[x][y+1]
            lowerMid=leftHalfGrey[x][y-1]
            lowerRight=leftHalfGrey[x+1][y+1]
            upperLeft=leftHalfGrey[x-1][y-1]
            upperRight=leftHalfGrey[x+1][y-1]
            lowerLeft=leftHalfGrey[x-1][y+1]

            ## Add the squares to our input layers:
            #inputLayer = np.array(
            #              [upperLeft[0], upperLeft[1], upperLeft[2],
            #              upperMid[0], upperMid[1], upperMid[2],
            #              upperRight[0], upperRight[1], upperRight[2],
            #              midLeft[0], midLeft[1], midLeft[2],
            #              midRight[0], midRight[1], midRight[2],
            #              lowerRight[0], lowerRight[1], lowerRight[2],
            #              lowerMid[0], lowerMid[1], lowerMid[2],
            #              lowerLeft[0], lowerLeft[1], lowerLeft[2]])
#
            ## Find our middle COLOR pixel to train with
            middleColor = leftHalfColor[x][y]

            inputLayer = np.array(
                         [ [upperLeft[0], upperMid[0], upperRight[0], midLeft[0], midRight[0], lowerRight[0], lowerMid[0],lowerLeft[0]], 
                          [upperLeft[1], upperMid[1], upperRight[1], midLeft[1], midRight[1], lowerRight[1], lowerMid[1],lowerLeft[1]],
                          [upperLeft[2], upperMid[2], upperRight[2], midLeft[2], midRight[2], lowerRight[2], lowerMid[2],lowerLeft[2]]])
            
            dim=inputLayer.shape
            hiddenLayer1_weights=[]

            for i in range(dim[1]):
                vec=[]
                for j in range(dim[0]):
                    vec.append(random.random())
                hiddenLayer1_weights.append(vec)
            hiddenLayer1_weights = np.array(hiddenLayer1_weights)
            # Now train, we want to associate the surrouning B&W pixels with a color pixel.
            #each vector should i
            # RUN FORWARD THROUGH INPUT LAYER
            print(inputLayer)
            print(hiddenLayer1_weights.shape)
            print(hiddenLayer1_bias.shape)
            print(np.matmul(inputLayer, hiddenLayer1_weights))

            hiddenLayer = sumTwoLists(np.matmul(inputLayer, hiddenLayer1_weights), hiddenLayer1_bias)
            hiddenlayerWithActivation= sigmoid_util(hiddenLayer)

            hiddenLayer2_weights=[]
            for i in range(hiddenlayerWithActivation.shape[0]):
                hiddenLayer2_weights.append(random.random())

            hiddenLayer2_bias=[random.random()]
            outputLayer = sumTwoLists(np.matmul(hiddenlayerWithActivation, hiddenLayer2_weights), hiddenLayer2_bias)

            for i in range(len(outputLayer)):
                outputLayer[i]=sigmoid(outputLayer[i])

            #outputLayer = sigmoid_util(outputLayer)

            print(outputLayer)


            # NOW TIME FOR STOCHASTIC GRADIENT DESCENT, AND BACK PROPAGATION W/ ERROR LOSS FUNCTION

            print("implement here stochasticGradientDescent()")





def get_training_data(image):  # left half of the image
    row = image.shape[0]
    column = int(image.shape[1] / 2)
    print(column)
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

def set_to_grey_scale(image_data): # de-color an image
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            pixel=image_data[i][j]
            red=pixel[0]
            green=pixel[1]
            blue=pixel[2]

            RGB=int((.21*red)+(.72*green)+(.07*blue))

            image_data[i][j]=[RGB,RGB,RGB]

# We will use the sigmoid activation function
def sigmoid_util(n):
    size=n.shape
    for i in range(size[0]):
        for j in range(size[1]):
            n[i][j]= 1 / (1 + math.exp(-n[i][j]))

    return n


#def sigmoid(list):
#    if len(list) == 1: # nested list error sometimes
#        list = list[0]
#    return [sigmoid_util(element) for element in list]
def sigmoid(num):

    return 1/(1+math.exp(-num))




def stochasticGradientDescent():
    print("implement here")


def dotProduct(list1, list2):
    list1 = np.asarray(list1)
    list1 = list1.reshape(1, 24)
    return np.array(list1)*np.array(list2)


def sumTwoLists(list1, list2):
    result = list1
    for i in range(0, len(list1)):
        result[i]=np.add(result[i],list2)
    return result






if __name__ == '__main__':
    image = io.imread('super_small_flower.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')

    training_data = get_training_data(image)
    testing_data = get_testing_data(image)

    leftHalfGreyScale= np.copy(training_data)
    set_to_grey_scale(leftHalfGreyScale)

    # Run the improved agent code
    improved_agent(training_data, leftHalfGreyScale, testing_data)




