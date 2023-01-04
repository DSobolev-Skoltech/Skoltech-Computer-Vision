import numpy as np
import cv2 as cv
import operator
from skimage.segmentation import clear_border
import torch
import math
from imutils.perspective import four_point_transform
from tensorflow.keras.models import load_model

'''We are getting numbers in this chank of code'''

#Getting threshold
def predict_mask(image):
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #mask = np.zeros((img_grey.shape[:2]), np.uint8)
    #let's use hsl representation, cause light can help to properly distinguish between black ans white
    HLS = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    LIGHT = HLS[:, :, 1]

    mask_hls = (LIGHT < 100)
    mask_int = mask_hls.astype(np.uint8)

    contours, hierarchy = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #find contours in the picture

    areas = [cv.contourArea(c) for c in contours] #let's make array of areas of contours

    sort_areas_inds= sorted(range(len(areas)), key=lambda k: areas[k], reverse=True) #from largest to smallest
    largest=sort_areas_inds[:10]
    
    sudokus_cont=[]
    mask1 = np.zeros((img_grey.shape[:2]), np.uint8)
    for i in range(len(largest)):
        if (areas[largest[i]] > 0.5 * areas[largest[0]])and(areas[largest[i]]>550000) : #this condition helps to find second sudoku
            cv.drawContours(mask1,[contours[largest[i]]], 0, 1, -1) #adding largest contours
            sudokus_cont.append(contours[largest[i]])
    mask = np.bool_(mask1)
    # loading train image:
    #train_img_4 = cv.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    #rf = joblib.load('/autograder/submission/random_forest.joblib')
    return mask, sudokus_cont, img_grey

def get_cropped(sudokus_cont, image_grey): #returns arrray of sudoku fields from image

    sudokus=[] #list of images corresponding to cropped sudokus
    for i in sudokus_cont:
        #print(i)
        epsilon = 0.1* cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, epsilon, True)
        #print(approx)
        #print(np.ravel(approx).shape)
        #print(type(approx))
        cropped_sudoku= four_point_transform(image_grey, np.ravel(approx).reshape(4,2))
        sudokus.append(cropped_sudoku)
    return sudokus

def splitBoxes (img):
    for pieces in img:
        pieces = cv.resize(pieces, (450,450))
        rows = np.vsplit(pieces, 9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                boxes.append(box)
    return boxes

def intializePredictionModel():
    model = load_model('myModel.h5')
    return model


def getPrediction (boxes):
    model = intializePredictionModel()
    result = []
    for image in boxes:
        #Image preparation
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv.resize(img, (28,28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        #Predict
        predictions = model.predict(img)
        #classIndex = model.predict_classes(img)
        classIndex = np.argmax(predictions, axis = -1)
        probabilityValue = np.amax(predictions)
        #print(classIndex, probabilityValue)
        #Saving
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(-1)
    boards = np.array(result)
    boards = np.reshape(boards, (9,9))
    boards =[np.int16(boards)]
    return boards, result

def predict_image(img):
    mask, sudokus_cont, img_grey = predict_mask(img)
    sudokus = get_cropped(sudokus_cont, img_grey)
    boxes = splitBoxes (sudokus)
    digits, result = getPrediction (boxes)
    return mask, digits


# img2 = cv.imread('homework_2/train_0.jpg')
# mask, digits = predict_image(img2)
# # imgGRAY = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

# cv.imshow("Image Mask", mask)
# cv.waitKey(0)



# cv.imshow('image', mask)
# cv.waitKey(0)
# cv.destroyAllWindows()

# mask, contours = predict_mask(cv.imread ('homework_2/train_0.jpg'))
# sudokus = get_cropped (contours, imgGRAY)
# boxes = splitBoxes(sudokus)
# result, boards = getPrediction(boxes)

# print(boards)
# print(len(boxes))

# cv.imshow('image', boxes[5])
# cv.waitKey(0)
# cv.destroyAllWindows()
#for imgs in sudokus:
# for i in boxes:
#     cv.imshow('image', boxes[i])
#     cv.waitKey(0)
#     cv.destroyAllWindows()


#Getting the biggest contour
# def biggestContour(img, contours):
#     biggest = np.array([])
#     max_area = 0
#     for i in contours:
#         area = cv.contourArea(i)
#         if area > 50:
#             peri = cv.arcLength(i, True)
#             approx = cv.approxPolyDP(i, 0.02*peri, True)
#             if area > max_area and len(approx) == 4:
#                 biggest = approx
#                 max_area = area
#     return biggest, max_area
# biggest, maxArea = biggestContour(contours)
# print(biggest)

