# Skoltech-Computer-Vision
This repository is a set of three homeworks demonstrating my computer vision skills based on libraries such as openCV


HOMEWORK 1
Overview
The goal of the homework is to count scores for the Ticket to Ride Europe board game. 

ann_1.jpg

During the game, players connect different cities to complete some destination tickets. Every player has a set of train markers for claiming a route. At the end of the game, players got scores for claiming every route (=path between two neighboring cities) depending on its length:

ann_2.jpg

(left - the number of markers, right - the score). A full version of the rules can be found here Links to an external site..

Our goal is to create a system to count these scores automatically.

 

Data
To create an algorithm, we'll provide several "training examples" here.

To test your algorithm, we'll use a different set of images (the same map, the same angle of view).

 

Submission
Your submissions will be graded automatically via gradescope Links to an external site.(your code will be automatically launched to process test images). The technical information is detailed below - read it when you are about to submit it. By now, you can start with jupyter to play with images and adjust some parts of your algorithm.

 

Submission specifications
To submit your solution, you need to:

(1) Name your file as `ticket_to_ride.py`. Other filenames won't be recognized.
(2) Name the primary pipeline function (input-output specifications you could find below) as `predict_image`.
(3) Upload (submit) your .py file to Homework 1 of our MA030348 Links to an external site.course.
`predict_image` function takes only 1 argument - image array. We load test images via

 img = cv2.imread(path_to_img)
and pass the array strictly to your function. Test images have (1) the same map, (2) the same image orientation and view angle, (3) trains of the same five predefined colors, and (4) similar lightning conditions.
Then, the output of the `predict_image` function consists of 3 items:

centers, n_trains, scores = predict_image(img)
`centers` is a 2-dim python list with coordinates corresponding to the city (circle points on the map) coordinate pairs. The order of centers does not matter. Example: `[[1650, 1739], [237, 3046], [1451, 890]]`. Note that (spatial) coordinates should correspond to the original resolution of the image read by `cv2.imread`.
`n_trains` is a python dict consisting of 5 items - the number of trains (and not the empty cells) on the image corresponding to each color. If there are no trains of "black" color, the right answer is 0, and it should also be added to the dict. Color names are strictly defined as "blue", "green", "black", "yellow", and "red" and should be used as dict keys. Example: `{"blue": 34, "green": 42, "black": 0, "yellow": 0, "red": 28}`.
`scores` is a python dict consisting of 5 items - scores of each player (color) on the image. The format is the same as above. Example: `{"blue": 59, "green": 67, "black": 0, "yellow": 32, "red": 0}`. Scores are calculated for each color for the whole image (legend above should help), i.e., a sum of all the combinations of separate completed routes contribute to the score of each color
The example of a pipeline with the correct input-output specification is uploaded to the hw1 folder as a template `.py` file.

If you use template matching and crop your template from train images: train images will be available (with the same names) inside the platform container while your script is being evaluated. So if you used to load an image via

img_template = cv2.imread('train/black_red_yellow.jpg')
the new way of loading template image will be (add `/autograder/source/` at the beginning)

img_template = cv2.imread('/autograder/source/train/black_red_yellow.jpg')
 

The platform specifications
The platform provides you with python3.10 environment (activated by default) and preinstalled pip packages we need to solve the problem: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `opencv` (4.6.0). Please, ensure you use the same version of `opencv` while preparing the solution.

To execute your function platform use a single node with 4 CPU cores and 6.0 Gb RAM. Please, ensure you constrain your solution with 6 Gb of RAM. (e.g., do not run template matching on the full resolution image :D) Overall, downsampling your image three, four, or even more times is a good idea.

The limit of time on a single submit is 40 minutes. Please, ensure your script is fast enough to process 10 test images; otherwise, your submission won't be scored. If you have your algorithm working < 4 min on the single image (for all five colors) within a similar constrain of 4CPU, 6Gb RAM - you are (probably) fine:)

The number of submissions is unlimited.

Tracebacks and Warnings (if any) could be seen on your submission page.

Still have any questions? Please, ask them by telegram, e-mail, or in the comments.

 

Grading details
Our python templates will contain a set of functions with predefined signatures. We'll compare outputs of your function with "gold standard" answers and then assign scores using the following scores split for each test image. 

2.5 points for city detection (=coordinates of their centers): 2.5 * TP / (TP + FP + FN) where TP, FP, FN is the number of True Positive, False Positive, False Negative. TP is defined as the closest "hit" of any your center
5.0 points per image for trains number of each color. For every single color (colors are equal in grading, even if the number of trains for a specific color is 0):
1.0 if the number error is +-1
0.8 if +-2
0.6 if +-3
0.4 if +-4
0.2 if +-5
0.0 if the absolute error >= 6
2.5 points for the final score based on the aggregation of errors for each color. For every single color:
0.5 if the number error is +-1
0.4 if +-3
0.3 if +-5
0.2 if +-7
0.1 if +-9
0.0 if the absolute error >= 11
10.0 points for each of the 10 test images will result in 100.0 points (max). Your final score for every (successful) submission will be displayed on the leader board. The separate scores for every test image will be available on your submission info page.

Score scaling and bonus points
The leaderboard score will then be scaled on the `=== 100% score benchmark ===` score so that you can obtain bonus points. And finally, this scaled score will be cast into your homework1 grade (with bonus points, if any).

Penalty for code similarity
Be aware that the platform automatically performs a pretty smart code similarity comparison. Share your solutions on your own risk of losing homework scores:)
We will discount the scores for the parties with the high code similarity by the factor of N, where N is the number of party members.

Penalty for the security leaks abuse
If you (apparently) find any security leaks, exploit them at your own risk of losing homework scores:)
We set your homework1 score to 0 if we detect any sign of security leaks abuse.

Penalty for "hardcoding"
We expect you to create solutions based on computer vision algorithms. And the goal of our course is that you learn how to use them. So in this homework, we expect you to use computer vision algorithms to detect the cities, trains, and routes. We will not accept the solutions where you detect them "by hand." (In practice, it often means that you have hardcoded all possible coordinates in several hundred lines of python code.) The hardcoded parts of your solution will not contribute to the final score (for these sub-parts, the score will be zeroed).

 

Debugging and evaluating on the train set
We have provided you with the evaluation script called `evaluation.py`. It is available in the Homework 1's folder. You can test your pipeline at any stage of developing a solution. To do so, you need to satisfy the following conditions:

(1) follow all the specifications for the submission file: (a) it should be named `ticket_to_ride.py`, (b) it should contain a function named `predict_image`, (c) but now you should use your local paths, e.g., `img_template = cv2.imread('train/black_red_yellow.jpg')`;
(2) `evaluate.py` should be in the same folder (working directory) with your file `ticket_to_ride.py`;
(3) folder `train` with all its content should be in the same folder with (2).



HOMEWORK 2 
Overview
Goal: build a CV algorithm to solve Sudoku.

Part 1: Find tables 

Step 1: find some keypoints (Otsu thresholding, edges, Hough lines, corners of 9x9 table, etc.)
Step 2: find 9x9 tables, estimate the Sudoku-ness of every table (Hough lines, regular structures, etc.)
Step 3: apply Projective Transform for every found table
Part 2: Recognize digits

Step 4: divide the table into separate cells (optionally: remove table artifacts)
Step 5: build digit classifier on MNIST or manually (semi-supervised) annotated train data: feature extractor (e.g. HoG) + classifier (SVM, Random Forest, NN, etc.)
Legacy approach. It is less robust, but still working pretty fine. (It is not recommended to use template matching approach)

Step N: manually create templates using normalized images
Step N+1: apply match_template Links to an external site.to estimate correlation with your templates.
Step N+2: make a decision for each cell.
Part 3: [extra] Solve sudoku -> Draw solution

Step 6: We will provide you with a function to solve sudoku (github link Links to an external site.). You need to aggregate input in the right format.
Step 7: Plot solved sudoku on the original image. This step is optional and will result in bonus points.
 

Data
To create an algorithm, we'll provide several "training examples" here. 

To test your algorithm, we'll use a different set of images.

 

Submission specifications
To submit your solution, you need to:

(1) Name your file as `sudoku.py`. Other filenames won't be recognized.
(2) Name the primary pipeline function (input-output specifications you could find below) as `predict_image`.
(3) Upload (submit) your .py file to Homework 2 of our MA030348 course.
 

`predict_image` specifications:
`predict_image` function takes only 1 argument - image array. We load test images via

 img = cv2.imread(path_to_img)
and pass the array strictly to your function. Test images have the same sudoku table orientation. The output consists of 2 items:

mask, digits = predict_image(img)
`mask` is a 2-dim numpy array with bool values. Mask should contain only {0, 1} values corresponding to all full sudoku tables on the image. Mask should have the same spatial size as the original image. Examples:
  ;  ;  

`digits` is a python list consisting of 9x9 numpy arrays (preferable with int16 type) - predicted digits for every sudoku table. Empty cells should be encoded as `-1`. The order of numpy arrays in the case of several tables is not important. Example:
[
    np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
              [-1, -1, -1,  8,  9,  4, -1, -1, -1],
              [-1, -1, -1,  6, -1,  1, -1, -1, -1],
              [-1,  6,  5,  1, -1,  9,  7,  8, -1],
              [-1,  1, -1, -1, -1, -1, -1,  3, -1],
              [-1,  3,  9,  4, -1,  5,  6,  1, -1],
              [-1, -1, -1,  8, -1,  2, -1, -1, -1],
              [-1, -1, -1,  9,  1,  3, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
]
To create a mask from contours/points, you can use `cv2.fillPoly`.

Also, see `_template_sudoku.py` in the Homework 2 folder.

 

The platform specifications
You are provided with the same options as in Homework 1: 40 minutes of submission processing time, 4 CPU cores, 6 GB RAM.

Platform provides you with python3.10 environment (activated by default) and preinstalled pip packages we need to solve the problem: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `opencv`, `python-mnist`, `torch`, `tensorflow`, `torchvision`, `keras`, `imutils`.

All train images could be found here

'/autograder/source/train/'
All additional files (e.g. model weights), that you have submitted along with scripts, could be found here:

'/autograder/submission/'
 

Submission of the complete algorithm
To submit a "complete algorithm" (finding table -> recognizing digits -> solving sudoku -> visualizing solution) use a standard canvas submission.

Upload a single `.zip` file containing a fully reproducible jupyter notebook (please, include all model weights or templates you need). One should assume, that the `train` folder (folder with the training images) is on the same directory level as the running notebook.

The notebook should demonstrate a visual solution for the training images chosen by the student.

Processed images will be evaluated manually, on subjective visual perception of TAs. It is expected to be filled with drawn predicted digits in empty cells of the original image.


Grading
Sudoku 9x9 tables segmentation. The output is binary masks of the same size as the original image. (45% points based on IoU). IoU > 0.9 will be considered as 1.0 since the Ground Truth masks are not perfectly annotated.
Digit recognizer. The output is a python list with 9x9 numpy arrays with digits (including empty cells as a specified symbol `-1`). (45% points based on Accuracy).
Complete algorithm: filled solution of sudoku. (Extra 11.11% points, manually graded)
 

Score scaling and bonus points
The leaderboard score will then be scaled on the `=== 100% score benchmark ===` score so that you can obtain bonus points. And finally, this scaled score will be cast into your Homework 2 grade (with bonus points, if any).

Penalty for code similarity
Be aware that the platform automatically performs a pretty smart code similarity comparison. Share your solutions on your own risk of losing homework scores:)
We will discount the scores for the parties with the high code similarity by the factor of N, where N is the number of party members.

Penalty for the security leaks abuse
If you (apparently) find any security leaks, exploit them at your own risk of losing homework scores:)
We set your Homework 2 score to 0 if we detect any sign of security leaks abuse.

Debugging and evaluating on the train set
We have provided you with the evaluation script called `evaluation.py`. It is available in the Homework 2's folder. You can test your pipeline at any stage of developing a solution. To do so, you need to satisfy the following conditions:

(1) follow all the specifications for the submission file: (a) it should be named `sudoku.py`, (b) it should contain a function named `predict_image`, (c) but now you should use your local paths, e.g., some_image = cv2.imread('train/train_8.jpg').
(2) `evaluate.py` should be in the same folder (working directory) with your file `ticket_to_ride.py`;
(3) folder `train` with all its content should be in the same folder with (2).
Now, just run the evaluation script:

$> python evaluate.py
It will return you the training scores, estimated evaluation time, or the full traceback if you have any error. Having ground truth values for the train images in the `train` folder, you may also debug a pipeline yourself.


HOMEWORK 3

Overview
Goal: find locations of a given item on a store shelf.

Legend: there is a request from the international retail group that wants a CV system to verify the layout of products on its shelves. Some brands are buying out the top spots on shelves for their goods, so it's crucial to have them at the right places. You will develop a system to control merchandiser work in this assignment. If this system satisfies the customer, we will deploy it across thousands of shops.

tg_image_2507334198.jpeg

You both agreed that it's enough to write a simple function, computing item locations for a demo project. This function should get two images.

The "Gallery" image is the same as above (only without red boxes around juice).

"Query" image will contain one item cropped to its margins. For example:

dob2.jpeg

 

The output should be an array of locations [(x_min, y_min, width, height), ...].

Values are in the scale 0-1 (yolo format). (x_min, y_min) is left-upper corner of a bounding box, x_min = x_min *in pixels* / image_width.

The end of the year is approaching soon, so it's better to finish work before bookkeeping goes off-season. (Deadline is 16th 19th of December)

 

If this task is too hard: start with a simple classifier if a gallery image has a query item. Return [(0,0,1,1)] if it does, [] otherwise. It will give you some points. The next step is to return precise bounding boxes. 

If this task is too easy: there is a real-world problem with "production-grade" photos. However, not all of the items are made equal. Some may be more flexible (e.g., a pack of chips), but this case is hard to solve with classical CV approaches. We are suggesting trying to solve a case with tiny items. There is a picture with "АКТИВИЯ" yogurt (`train_extreme.jpg` and `template_extreme.jpg`) waiting for you. The latter is not required to get 100 points for this assignment but may be helpful if you want to face some out-of-(study)sandbox problems.

 

Data
To create an algorithm, we'll provide several "training examples" here. File `template_X*.jpg` is the query for the image `image_X.jpg`. Sometimes, there are several queries for the same image.

To test your algorithm, we'll use a different set of images.

 

Submission specifications
To submit your solution, you need to:

(1) Name your file as `retrieval.py`. Other filenames won't be recognized.
(2) Name the primary pipeline function (input-output specifications you could find below) as `predict_image`.
(3) Upload (submit) your .py file to Homework 3 of our MA030348 course.
 

`predict_image` specifications:
`predict_image` function takes two arguments: image array and query image array. We load test images via

 img = cv2.imread(path_to_img)
and pass the arrays strictly to your function. The output consists of one python list:

bboxes_list = predict_image(img, query)
`bboxes_list` is a python list, empty or consisting of the bounding boxes. A bounding box is a python tuple consisting of 4 float numbers `(x_min, y_min, width, height)`. `x_min`, `y_min`, `width`, and `height` are relative to the image shape, i.e., scaled into [0, 1] interval by the width or height of the image.

Also, see `_template_solution.py` in the Homework 3 folder.

 

The platform specifications
We provide you with the same options as in Homework 1 or 2: 40 minutes of submission processing time, 4 CPU cores, 6 GB RAM.

The platform provides you with python3.10 environment (activated by default) and preinstalled pip packages we need to solve the problem: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `opencv`.

 

Grading
You get up to 10% points for each of the ten image-query pairs in a hidden test.

For every pair, you get 2% of the total score for the right guess if a gallery image has a query item: output `[]`  if no query in the gallery, output `[(0, 0, 1, 1)]` if there is a query in the gallery.
For every pair, you get
8% of the total score if all bounding boxes are correct and there are no false positive proposals;
6% of the total score in the case of FN + FP = 1;
4% of the total score in the case of FN + FP = 2;
2% of the total score in the case of FN + FP = 3;
0% of the total score in the case of FN + FP >= 4.
 

Score scaling and bonus points
The leaderboard score will then be scaled on the `=== 100% score benchmark ===` score so that you can obtain bonus points. And finally, this scaled score will be cast into your Homework 3 grade.

Penalty for code similarity
Be aware that the platform automatically performs a pretty smart code similarity comparison. Share your solutions on your own risk of losing homework scores:)
We will discount the scores for the parties with the high code similarity by the factor of N, where N is the number of party members.

Penalty for the security leaks abuse
If you (apparently) find any security leaks, exploit them at your own risk of losing homework scores:)
We set your Homework 3 score to 0 if we detect any sign of security leaks abuse.
