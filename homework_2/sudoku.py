import numpy as np
import cv2 as cv
import operator
from skimage.segmentation import clear_border
import torch
import math
from imutils.perspective import four_point_transform
from tensorflow.keras.models import load_model
import sudukoSolver
'''We are getting numbers in this chank of code'''

def mask_prediction(image):

    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imgGREY = cv.cvtColor(imgRGB, cv.COLOR_RGB2GRAY)
    
    #4 th seminar has a great potential to make a nice mask to distinguish sudoku
    HLS = cv.cvtColor(imgRGB, cv.COLOR_RGB2HLS)
    LIGHT = HLS[:, :, 1]
    maskHLS = (LIGHT < 100)
    maskINT = maskHLS.astype(np.uint8)
    contours, useless_param = cv.findContours(maskINT, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    
    contour_areas = [cv.contourArea(c) for c in contours] 
    
    sort_areas_inds = sorted(range(len(contour_areas)), key = lambda k: contour_areas[k], reverse=True) 
    largest_contours=sort_areas_inds[:10]

    
    sudokus_contour=[]
    zero_mask = np.zeros((imgGREY.shape[:2]), np.uint8)
    for i in range(len(largest_contours)):
        if (contour_areas[largest_contours[i]] > 0.5 * contour_areas[largest_contours[0]]) and (contour_areas[largest_contours[i]] > 550000) : #the second condition will find second sudoku if any
            cv.drawContours(zero_mask, [contours[largest_contours[i]]], 0, (255,0,0), -2) #adding largest contours
            sudokus_contour.append(contours[largest_contours[i]])
    
    mask = np.bool_(zero_mask)
    return mask


def preProcess(img):
    height, width = 3000, 3000
    img = cv.resize(img, (width,height))
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
    imgThreshold = cv.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    contours, hieracrchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # imgThreshold = cv.cvtColor(imgThreshold, cv.COLOR_BGR2RGB)
    return imgThreshold, contours

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype = np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def biggestContour(img, contours):
    biggest = np.array([])
    height, width = 3000, 3000
    img = cv.resize(img, (width,height))
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    if biggest.size != 0:
            biggest = reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[1]], [img.shape[1], img.shape[0]]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
            imgBlank = np.zeros((450, 450, 3), np.uint8)
            imgWarpColored = cv.cvtColor(imgWarpColored, cv.COLOR_RGB2GRAY)
    return biggest, max_area, imgWarpColored

def splitBoxes_1 (img):
    img = cv.resize(img, (450,450))
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

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


def splitBoxes_2 (img):
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
    boards = [np.int16(boards)]
    return boards, result



def predict_image(img):
    mask = mask_prediction(img)
    imgThreshold, contours = preProcess(img)
    biggest, max_area, imgWarpColored = biggestContour(img, contours)
    boxes = splitBoxes_1 (imgWarpColored)
    digits, result = getPrediction (boxes)
    return mask, digits, result, biggest

a, b, result, biggest = predict_image(cv.imread('train/train_5.jpg'))
for i in range(len(result)):
    # replace hardik with shardul
    if result[i] == -1:
        result[i] = 0
print(result)

imgBlank = np.zeros((450, 450, 3), np.uint8)

#Display numbers
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv.LINE_AA)
    return img

result = np.asarray(result)

posArray = np.where(result > 0, 0, 1)

sudoku_board = np.array_split(result,9)

print(sudoku_board)

imgSolvedDigits = imgBlank.copy()
imgDetectedDigits = imgBlank.copy()

try:
    sudukoSolver.solve(sudoku_board)
except:
    pass



flatList = []
for sublist in sudoku_board:
    for item in sublist:
        flatList.append(item)
solvedNumbers = flatList*posArray

imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv.line(img, pt1, pt2, (255, 255, 0),2)
        cv.line(img, pt3, pt4, (255, 255, 0),2)
    return 

heightImg = 450
widthImg = 450

img = cv.resize(cv.imread('train/train_5.jpg'), (widthImg, heightImg))

pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix = cv.getPerspectiveTransform(pts1, pts2)  # GER
imgInvWarpColored = img.copy()
imgInvWarpColored = cv.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
imgDetectedDigits = drawGrid(imgDetectedDigits)
imgSolvedDigits = drawGrid(imgSolvedDigits)

cv.imshow('result', imgSolvedDigits)
cv.waitKey(0)