import cv2
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth


#first version, was slitely changed, score 56.
"""resourse https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects"""
def comparing_images(img1, img2, a=6, b=0.08, c=8, d=1.2):
    MIN_MATCH_COUNT = 10
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # queryImage
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # trainImage

    orb = cv2.SIFT_create(15000, a, b, c, d)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    x = np.array([kp2[0].pt])

    for i in range(len(kp2)):
        x = np.append(x, [kp2[i].pt], axis=0)
    print(x)
    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    s = [None] * n_clusters_
    for i in range(n_clusters_):
        l = ms.labels_
        d, = np.where(l == i)
        print(d.__len__())
        s[i] = list(kp2[xx] for xx in d)

    des2_ = des2

    results = []
    points = []
    for i in range(n_clusters_):
        kp2 = s[i]
        l = ms.labels_
        d, = np.where(l == i)
        des2 = des2_[d, ]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        des1 = np.float32(des1)
        des2 = np.float32(des2)

        matches = flann.knnMatch(des1, des2, 2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                
                good.append(m)

        if len(good) > 3:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

            if M is None:
                print ("No Homography")
            else:
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape

                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                dst = cv2.perspectiveTransform(pts,M)
                # points.append(np.int32(dst))
                points.append((dst[0][0][0], dst[0][0][1], np.abs(dst[2][0][0]-dst[0][0][0]), np.abs(dst[2][0][1]-dst[0][0][1])))
                img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                singlePointColor=None,
                                matchesMask=matchesMask,  # draw only inliers
                                flags=2)

                img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
                results.append(img3)
                
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
    return points

#Version number two, with separated functions, resourse is the same + OpenCV tutorials
def preprocess_and_find_features(img, sift):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) != 2 else img
    kp, desc = sift.detectAndCompute(img, None)

    return img, (kp, desc)

def ratio(matches, thresh):
    return [m for m, n in matches if m.distance < thresh * n.distance]

def find_dst_pts(pts, kp1, kp2, good):
    src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    return cv2.perspectiveTransform(pts, M) 

def find_good_matches(des1, des2, thresh):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=10))
    matches = flann.knnMatch(des1, des2, k=2)

    return ratio(matches, thresh)

def predict_image(img: np.ndarray, query: np.ndarray) -> list:

    results = []

    sift = cv2.SIFT_create()

    img1, (kp1, des1) = preprocess_and_find_features(query, sift)
    img2, (kp2, des2) = preprocess_and_find_features(img, sift)

    pts = np.float32([[0, 0], [0, img1.shape[0]-1], [img1.shape[1]-1, img1.shape[0]-1], [img1.shape[1]-1, 0]]).reshape(-1, 1, 2)

    good = find_good_matches(des1, des2, 0.7)
    if len(good) > 32:
        dst = find_dst_pts(pts, kp1, kp2, good)

        img2 = cv2.fillPoly(img2, [np.int32(dst)], 0)

        results.append(
                        (
                            dst[0, 0, 0] / img.shape[1], 
                            dst[0, 0, 1] / img.shape[0], 
                            np.abs(dst[0, 0, 0] - dst[2, 0, 0]) / img.shape[1], 
                            np.abs(dst[0, 0, 1] - dst[1, 0, 1]) / img.shape[0]
                        )
                    )

        for _ in range(30):
            img2, (kp2, des2) = preprocess_and_find_features(img2, sift)
            
            good = find_good_matches(des1, des2, 0.88)
            if len(good) > 32:
                dst = find_dst_pts(pts, kp1, kp2, good)

                if len(dst) == 4 and \
                    np.isclose(np.linalg.norm(dst[0, 0] - dst[1, 0]) / np.linalg.norm(dst[2, 0] - dst[3, 0]), 1, 0.1) and \
                    np.isclose(np.linalg.norm(dst[0, 0] - dst[3, 0]) / np.linalg.norm(dst[2, 0] - dst[1, 0]), 1, 0.1):
                    img2 = cv2.fillPoly(img2, [np.int32(dst)], 0)

                    results.append(
                                    (
                                        dst[0, 0, 0] / img.shape[1], 
                                        dst[0, 0, 1] / img.shape[0], 
                                        np.abs(dst[0, 0, 0] - dst[2, 0, 0]) / img.shape[1], 
                                        np.abs(dst[0, 0, 1] - dst[1, 0, 1]) / img.shape[0]
                                    )
                                )
        return results
    else:
        return results