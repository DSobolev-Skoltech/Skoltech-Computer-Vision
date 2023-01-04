
import cv2 as cv
import numpy as np
from imutils.perspective import four_point_transform
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
import joblib
import operator
import joblib
import random


"""Huge part pf the mask prediction is got from stackoverflow"""
#Mask and Sudoku contour prediction

def mask_prediction(image):

    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imgGREY = cv.cvtColor(imgRGB, cv.COLOR_RGB2GRAY)
    
    #4 th seminar has a great potential to make a nice mask to distinguish sudoku
    HLS = cv.cvtColor(imgRGB, cv.COLOR_RGB2HLS)
    LIGHT = HLS[:, :, 1]
    maskHLS = (LIGHT < 100)
    maskINT = maskHLS.astype(np.uint8)
    #Regular contour finder
    contours, useless_param = cv.findContours(maskINT, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    #array of contours
    contour_areas = [cv.contourArea(c) for c in contours] 
    #Sort values
    sort_areas_inds = sorted(range(len(contour_areas)), key = lambda k: contour_areas[k], reverse=True) 
    largest_contours=sort_areas_inds[:10]

    sudokus_contour=[]
    zero_mask = np.zeros((imgGREY.shape[:2]), np.uint8)
    for i in range(len(largest_contours)):
        if (contour_areas[largest_contours[i]] > 0.5 * contour_areas[largest_contours[0]]) and (contour_areas[largest_contours[i]] > 550000) : #the second condition will find second sudoku if any
            cv.drawContours(zero_mask, [contours[largest_contours[i]]], 0, (255,0,0), -2) #adding largest contours
            sudokus_contour.append(contours[largest_contours[i]])
    mask = np.bool_(zero_mask)
    return mask, sudokus_contour

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



#Got the thing with four point transform from Data science friends
#Gives us cropped image of a sudoku
def crop(sudokus_cont, imgGREY): 
    sudokus=[] 
    for i in sudokus_cont:
        epsilon = 0.1* cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, epsilon, True)
        try:
            cropped_sudoku = four_point_transform(imgGREY, np.ravel(approx).reshape(4,2))
        except:
            cropped_sudoku=imgGREY
        sudokus.append(cropped_sudoku)
    return sudokus

#taking every single number from sudoku for further digit recognition
def get_single_cell(y, x, maps):
    dx=maps.shape[1]//9
    dy=maps.shape[0]//9
    digit_img = maps[y*dy:(y+1)*dy, x*dx:(x+1)*dx]
    return digit_img

#getting clear sudoku image
def preProcess(image): 
    picture = cv.GaussianBlur(image,(9, 9),3)
    thresh = cv.threshold(picture, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    imageTHRSH = clear_border(thresh)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(imageTHRSH, kernel, iterations = 1)
    dilation = cv.dilate(erosion, kernel, iterations = 1)
    img_masked=dilation
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gray = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
    img_masked=gray
    return img_masked

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

  # process crop width and height for max available dimension
def center_crop(img, dim=(64, 64)):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_point_x, mid_point_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    cw2=min(cw2, ch2)
    ch2=cw2
    crop_img = img[mid_point_y - cw2:mid_point_y + cw2, mid_point_x - ch2:mid_point_x + ch2]
    return crop_img

'''Model has been raking and over thought from three different sources:
https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
https://towardsdatascience.com/open-cv-based-sudoku-solver-powered-by-rust-df256653d5b3
https://gist.github.com/qgolsteyn/7da376ced650a2894c2432b131485f5d'''
def intializePredictionModel():
    model = load_model('/autograder/submission/cnn_classifier.h5')
    return model

def predict_image(image):
    model = intializePredictionModel()
    if (len(image.shape) == 2):
        image=cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    sudokus_contour=mask_prediction(image)[1]
    imgGREY=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_array=crop(sudokus_contour, imgGREY)
    boards=[]
    for img in img_array:
        board=-np.ones((9, 9), dtype="int")
        table_img = preProcess(img)
        for y in range(0, 9):
            for x in range(0, 9):
                digit_box=get_single_cell(y, x, table_img)
                digit_box_big=center_crop(digit_box, dim=(digit_box.shape[0] - 0.31 * digit_box.shape[0], digit_box.shape[1] - 0.31 * digit_box.shape[0]))
                fillness=cv.countNonZero(digit_box_big) / float(digit_box_big.shape[0] * digit_box_big.shape[1])
                digit_box=tf.image.resize(tf.expand_dims(digit_box_big, axis=2), (28,28))
                ita = img_to_array(digit_box)
                ita = np.expand_dims(ita, axis=0)
                pred = model.predict(ita).argmax(axis=1)[0]
                if fillness<0.1:
                    pred=-1
                board[y, x] = pred
        boards.append(board)
    digits = boards
    mask=mask_prediction(image)[0]
    return mask, digits

