import numpy as np
from skimage import io, color



def improved_agent(leftHalfColor, leftHalfGrey, rightHalfGrey):

    leftHalfSize=leftHalfGrey.shape
    rightHalfSize=rightHalfGrey.shape

    rightRows=len(rightHalfGrey[0])
    rightColumns=len(rightHalfGrey[1])


    print("implement here")


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



if __name__ == '__main__':
    image = io.imread('super_small_flower.jpg')
    image = color.convert_colorspace(image, 'RGB', 'RGB')

    training_data = get_training_data(image)
    testing_data = get_testing_data(image)

    leftHalfGreyScale= np.copy(training_data)
    set_to_grey_scale(leftHalfGreyScale)

    # Run the improved agent code
    improved_agent(training_data, leftHalfGreyScale, testing_data)



