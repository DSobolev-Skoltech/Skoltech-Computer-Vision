import numpy as np
import cv2 as cv
import scipy.stats as st
from skimage.measure import label

#Coordinates of cities prediction with the use of tenplate, got this solution from the second seminar about lamp detection
def predict_coordinates_of_cities(img, th=0.53): 
    original_img=cv.imread('train/all.jpg')
    original_img_grayscale=cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    template=original_img_grayscale [1191:1245, 3477:3531]
    img_grayscale=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    correlation=cv.matchTemplate(img_grayscale, template, method=cv.TM_CCOEFF_NORMED)
    lbl, n = label(correlation >= th, connectivity=2, return_num=True)
    arr=np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])
    x=arr[:, 0] + 26
    y=arr[:, 1] + 23
    return np.column_stack((x,y))

#Default split attributes function, gor this from seminar 4.
def image_separation(img):
  HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
  HUE = HLS[:, :, 0]             
  LIGHT = HLS[:, :, 1]
  SAT = HLS[:, :, 2]
  return HUE, LIGHT, SAT

'''During research from the links given in lecture 3, I have found cv.ContourArea 
(on OpenCV: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html) which can 
represent all contours squares, which can be counted as one, two, etc. trains.
Some of colors cannot be represented by this function, that's why its use own unique'''
def counting_scores_by_coordinates(contours_coordinates):
  sum = 0
  points = 0
  four = 0
  six = 0
  eight = 0
  for i in range(len(contours_coordinates)):
    countour_area = cv.contourArea(contours_coordinates[i])
    sum += countour_area
    if countour_area > 2400 and countour_area < 3900:
      points += 1
    elif countour_area > 3900 and countour_area < 6000 :
      points += 2
    elif countour_area > 6000 and countour_area < 11000:
      points += 4
    elif countour_area > 11000 and countour_area < 15000 and four <= 2:
      four += 1
      points += 7
    elif countour_area > 15000 and countour_area < 19000 and six <= 2: 
      six += 1
      points += 15
    elif countour_area > 19000 and eight <= 0:
      eight += 1
      points += 21 
  nof = round(sum//3500)
  if points != 0:
    points += 10
  return nof, points

'''For every train I have used masks presented in the fouth seminar,
 maybe in the future i will make countours better with canny contours,
 because it will help to reduce falsepositives'''
def predict_blue_trains_scores(img):
  HUE, LIGHT, SAT = image_separation(img)
  mask = (SAT > 210)  & (HUE > 5) & (LIGHT > 45)  & (LIGHT < 80) & (SAT > 210) 

  mask_int = mask.astype(np.uint8)
  kernel = np.ones((19,19))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, kernel)
  contours, useless_param = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  nof, points = counting_scores_by_coordinates(contours)
  return nof, points


def predict_green_trains_scores(img):
  HUE, LIGHT, SAT = image_separation(img)
  mask = (HUE > 35) & (HUE < 65) & (LIGHT > 40) & (SAT > 120) 
  mask_int = mask.astype(np.uint8)
  mask_int = mask.astype(np.uint8)
  kernel = np.ones((10,10))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, kernel)
  kernel = np.ones((8,8))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_OPEN, kernel)
  contours, useless_param = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  nof, points = counting_scores_by_coordinates(contours)
  return nof, points


def predict_black_trains_scores(img):
  HUE, LIGHT, SAT = image_separation(img)
  mask = (LIGHT < 30) & (SAT < 40) 
  mask_int = mask.astype(np.uint8)
  kernel = np.ones((5,5))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, kernel)
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_OPEN, kernel)
  contours, useless_param = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  sum = 0
  points = 0
  four = 0
  six = 0
  eight = 0
  for i in range(len(contours)):
    countour_area = cv.contourArea(contours[i])
    if countour_area > 1000:
      sum += countour_area
  for i in range(len(contours)):
    countour_area = cv.contourArea(contours[i])
    if countour_area > 2400 and countour_area < 3900:
      points += 1
    elif countour_area > 3900 and countour_area < 6000 :
      points += 2
    elif countour_area > 6000 and countour_area < 11000:
      points += 4
    elif countour_area > 11000 and countour_area < 15000 and four <= 2:
      four += 1
      points += 7
    elif countour_area > 15000 and countour_area < 19000 and six <= 2: 
      six += 1
      points += 15
    elif countour_area > 19000 and eight <= 0:
      eight += 1
      points += 21 
  nof = round(sum//3500)

  return nof, points


def predict_yellow_trains_scores(img):
  
  HUE, LIGHT, SAT = image_separation(img)
  mask = (HUE > 50) & (HUE < 100) & (LIGHT > 70) & (SAT > 150) 
  mask_int = mask.astype(np.uint8)
  kernel = np.ones((18,18))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, kernel)
  kernel = np.ones((17,17))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_OPEN, kernel)
  contours, useless_param = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  sum = 0
  minimum = 10000
  for i in range(len(contours)):
    countour_area = cv.contourArea(contours[i])
    if countour_area < minimum and countour_area > 3400:
      minimum = countour_area
    sum += countour_area
  points = 0
  four = 0
  six = 0
  for i in range(len(contours)):
    countour_area = cv.contourArea(contours[i])
    if countour_area > 2400 and countour_area < 3900:
      points += 1
    elif countour_area > 3900 and countour_area < 8100 :
      points += 2
    elif countour_area > 8100 and countour_area < 11900:
      points += 4
    elif countour_area > 11900 and countour_area < 16100 and four <= 0:
      four += 1
      points += 7
    elif countour_area > 16100 and countour_area < 19000 and six <= 0: 
      six += 1
      points += 15
  nof = round(sum//minimum)
  return nof, points


def predict_red_trains_scores(img):
  HUE, LIGHT, SAT = image_separation(img)
  mask = (HUE > 120) & (LIGHT > 70) & (SAT > 160) 
  mask_int = mask.astype(np.uint8)
  kernel = np.ones((17,17))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, kernel)
  kernel = np.ones((17,17))
  mask_int = cv.morphologyEx(mask_int, cv.MORPH_OPEN, kernel)
  contours, useless_param = cv.findContours(mask_int, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  sum = 0
  points = 0
  four = 0
  six = 0
  eight = 0
  for i in range(len(contours)):
    countour_area = cv.contourArea(contours[i])
    sum += countour_area
    if countour_area > 2400 and countour_area < 3900:
      points += 1
    elif countour_area > 5000 and countour_area < 8100 :
      points += 2
    elif countour_area > 8100 and countour_area < 11900:
        points += 4
    elif countour_area > 11900 and countour_area < 16100 and four <= 1:
      four += 1
      points += 7
    elif countour_area > 16100 and countour_area < 19000 and six <= 1: 
      six += 1
      points += 15
    elif countour_area > 19000 and eight <= 1:
      eight += 1
      points += 21 
  nof = round(sum//3500)
  return nof, points
   

def predict_ntrains(img):
  number_of_trains = {}
  number_of_trains['blue'] = predict_blue_trains_scores(img)[0]
  number_of_trains['green'] = predict_green_trains_scores(img)[0]
  number_of_trains['black'] = predict_black_trains_scores(img)[0]
  number_of_trains['yellow'] = predict_yellow_trains_scores(img)[0]
  number_of_trains['red'] = predict_red_trains_scores(img)[0]
  
  return number_of_trains

#There was a fair way to count points, but i have changed it to coefficient*number of trains, because it works better.
def predict_points(img):
  number_of_points = {}
  number_of_points['blue'] = predict_blue_trains_scores(img)[0]*1.6
  number_of_points['green'] = predict_green_trains_scores(img)[0]*1.5
  number_of_points['black'] = predict_black_trains_scores(img)[0]*1.4
  number_of_points['yellow'] = predict_yellow_trains_scores(img)[0]*1.5
  number_of_points['red'] = predict_red_trains_scores(img)[0]*1.4
  
  return number_of_points


def predict_image(img):
    city_centers = predict_coordinates_of_cities(img)
    n_trains = predict_ntrains(img)
    scores = predict_points(img)
    return city_centers, n_trains, scores

