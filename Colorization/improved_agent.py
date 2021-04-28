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
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]
    )
    outputLayer = np.array(
        [[0],
        [0],
        [0]]
    )
    hiddenLayer1_weights = np.array(
        [[random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()]]
    )
    hiddenLayer1_bias = np.array(
        [[random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()]]
    )
    hiddenLayer2_weights = np.array(
        [[random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()]]
    )
    hiddenLayer2_bias = np.array(
        [[random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()],
         [random.random()]]
    )



    # Loop through entire left side (this is TRAINING)
    for x in range(0, leftRows):
        for y in range(0, leftColumns):

            # Find our 8 surrounding squares (B&W) to train with
            midRight=leftHalfGrey[x+1][y]
            midLeft=leftHalfGrey[x-1][y]
            upperMid=leftHalfGrey[x][y+1]
            lowerMid=leftHalfGrey[x][y-1]
            lowerRight=leftHalfGrey[x+1][y+1]
            upperLeft=leftHalfGrey[x-1][y-1]
            upperRight=leftHalfGrey[x+1][y-1]
            lowerLeft=leftHalfGrey[x-1][y+1]

            # Add the squares to our input layers:
            inputLayer = [upperLeft, upperMid, upperRight, midLeft, midRight, lowerRight, lowerMid, lowerLeft]

            # Find our middle COLOR pixel to train with
            middleColor = leftHalfColor[x][y]


            # Now train, we want to associate the surrouning B&W pixels with a color pixel.

            # RUN FORWARD THROUGH INPUT LAYER
            hiddenLayer = sumTwoLists(dotProduct(inputLayer, hiddenLayer1_weights),  hiddenLayer1_bias)

            hiddenlayerWithActivation= sigmoid(sum(hiddenLayer))

            outputLayer = sum(dotProduct(hiddenlayerWithActivation, hiddenLayer2_weights) + hiddenLayer2_bias)


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
def sigmoid(n):
    return 1 / (1 + math.pow(math.e, -n))


def stochasticGradientDescent():
    print("implement here")


def dotProduct(list1, list2):
    return sum([x * y for x, y in zip(list1, list2)])


def sumTwoLists(list1, list2):
    result = [[list1[i][j] + list2[i][j] for j in range
    (len(list1[0]))] for i in range(len(list1))]
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

