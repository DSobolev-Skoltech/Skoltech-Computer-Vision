import cv2 as cv
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

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

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    list_of_bboxes = comparing_images(query, img)
    return list_of_bboxes


# print(predict_image(cv.imread('homework_3/train/train_0.jpg'), cv.imread('homework_3/train/template_0_0.jpg')))