import numpy as np
from skimage import io, color
import math
import random


def improved_agent(leftHalfColor, leftHalfGrey, rightHalfGrey):
    # Get size values for easy access
    rightHalfSize = rightHalfGrey.shape
    rightRows = len(rightHalfGrey[0])
    rightColumns = len(rightHalfGrey[1])

    leftHalfSize = leftHalfGrey.shape
    leftRows = len(leftHalfGrey[0])
    leftColumns = len(leftHalfGrey[0])

    # Define a neural network

    ##LAYER1: INPUT LAYER
    inputLayer = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    ##LAYER2: HIDDEN LAYER IN BIAS
    hiddenLayer1_bias = np.array(
        [random.random(),random.random(),random.random()]
    )

    ##LAYER2: HIDDEN LAYER IN WEIGHTS
    dim = inputLayer.shape
    hiddenLayer1_weights = []
    for i in range(dim[1]):
        vec = []
        for j in range(dim[0]):
            vec.append(random.random())
        hiddenLayer1_weights.append(vec)
    hiddenLayer1_weights = np.array(hiddenLayer1_weights)

    ##LAYER2: HIDDEN LAYER OUT WEIGHTS
    hiddenLayer2_weights = []
    for i in range(dim[0]):
        hiddenLayer2_weights.append(random.random())

    ##LAYER2: HIDDEN LAYER OUT BIAS
    hiddenLayer2_bias = [random.random()]

    ##LAYER3: OUTPUT LAYER
    outputLayer = np.array(
        [0, 0, 0]
    )



    ## Prep our patches array and actual color values for training.
    actualValues = []
    patches = []

    for x in range(1, leftRows - 1):
        for y in range(1, leftColumns - 1):
            # Find our 8 surrounding squares (B&W) to train with
            midRight = leftHalfGrey[x + 1][y]
            midLeft = leftHalfGrey[x - 1][y]
            upperMid = leftHalfGrey[x][y + 1]
            lowerMid = leftHalfGrey[x][y - 1]
            lowerRight = leftHalfGrey[x + 1][y + 1]
            upperLeft = leftHalfGrey[x - 1][y - 1]
            upperRight = leftHalfGrey[x + 1][y - 1]
            lowerLeft = leftHalfGrey[x - 1][y + 1]
            mid = leftHalfGrey[x][y]

            ## Find our middle COLOR pixel to train with
            actualValues.append(leftHalfColor[x][y])

            ## Add the squares to our input layers:
            patches.append([
                [upperLeft[0] / 255, upperMid[0] / 255, upperRight[0] / 255, midLeft[0] / 255, midRight[0] / 255, lowerRight[0] / 255, lowerMid[0] / 255, lowerLeft[0] / 255, mid[0] / 255],
                 [upperLeft[1] / 255, upperMid[1] / 255, upperRight[1] / 255, midLeft[1] / 255, midRight[1] / 255, lowerRight[1] / 255, lowerMid[1] / 255, lowerLeft[1] / 255, mid[1] / 255],
                 [upperLeft[2] / 255, upperMid[2] / 255, upperRight[2] / 255, midLeft[2] / 255, midRight[2] / 255, lowerRight[2] / 255, lowerMid[2] / 255, lowerLeft[2] / 255, mid[2] / 255]
            ])




    ## Parameters for training, feel free to edit to tune the model
    epochs = 1000
    w0 = .2
    learningRate = .7

    # NOW TRAIN
    for _ in range(epochs):

        ## GRAB A RANDOM SAMPLE AKA 'Stochastic'
        rand = random.randint(0, len(actualValues)-1)

        ## Fetch the input patches:
        inputLayer = np.array(patches[rand])

        ## Grab the resulting middle color (actual)
        actualColor = actualValues[rand]

        ## Now train, we want to associate the surrounding B&W pixels with a color pixel.
        hiddenLayer = sumTwoLists(np.matmul(inputLayer, hiddenLayer1_weights), hiddenLayer1_bias)
        hiddenlayerWithActivation = sigmoid_util(hiddenLayer)
        outputLayer = sumTwoLists(np.matmul(hiddenlayerWithActivation, hiddenLayer2_weights), hiddenLayer2_bias)
        outputLayer = [sigmoid(outputLayer[0]), sigmoid(outputLayer[1]), sigmoid(outputLayer[2])]

        ## Compute the loss (error function) & learning rate
        error = loss(outputLayer, actualColor)

        ## NOW TIME FOR STOCHASTIC GRADIENT DESCENT
        
        # work in progress
        hiddenLayer1_weights = hiddenLayer1_weights - (learningRate * GRADIENT)





def stochasticGradientDescent():
    print("implement here")



# Using SUM-SQUARE ERROR to compute loss (SUM of ALL (y-y0)^2)
def loss(predicted, actual):
    error = 0
    for i in range(0, len(predicted)):
        error = error + math.pow((predicted[i] - actual[i]), 2)
    return error


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


# We will use the sigmoid activation function
def sigmoid_util(n):
    size = n.shape
    for i in range(size[0]):
        for j in range(size[1]):
            n[i][j] = 1.0 / (1.0 + math.exp(-n[i][j]))
    return n


def sigmoid(num):
    return 1 / (1 + math.exp(-num))


def dotProduct(list1, list2):
    list1 = np.asarray(list1)
    list1 = list1.reshape(1, 24)
    return np.array(list1) * np.array(list2)


def sumTwoLists(list1, list2):
    result = list1
    for i in range(0, len(list1)):
        result[i] = np.add(result[i], list2)
    return result


if __name__ == '__main__':
    image = io.imread('nasa.jpeg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')

    training_data = get_training_data(image)
    testing_data = get_testing_data(image)

    leftHalfGreyScale = np.copy(training_data)
    set_to_grey_scale(leftHalfGreyScale)

    # Run the improved agent code
    improved_agent(training_data, leftHalfGreyScale, testing_data)

