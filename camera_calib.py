# https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch04.html

import numpy as np
import cv2
import glob

''' Dimensions of the chessboard '''
chessboard_pattern = (8,10)
chessboard_internal_pattern = (6,8)
chessboard_pattern_size_mm = 34

''' Path to the image to undistort '''
distorted_image_1 = './images/img_014.jpg'

''' Defining the world coordinates for 3D points. Object points are (0,0,0), (1,0,0), (2,0,0), ..., (6,8,0) '''
objp = np.zeros((chessboard_internal_pattern[0] * chessboard_internal_pattern[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_internal_pattern[0], 0:chessboard_internal_pattern[1]].T.reshape(-1, 2)

''' Scaling the object points by the pattern size '''
# objp = objp * chessboard_pattern_size_mm

''' Arrays to store object points and image points from all the images '''
objpoints = [] # 3d points for chessboard images (world coordinate frame)
imgpoints = [] # 2d points for chessboard images (camera coordinate frame)

''' Path of chessboard images used for caliberation '''
image_list = glob.glob('./images/*.jpg')

''' Termination Criteria '''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for image in image_list:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_internal_pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If corners are found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine pixel coordinates for given 2d points
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the chessboard corners
        cv2.drawChessboardCorners(img, chessboard_internal_pattern, corners2, ret)
        cv2.imshow('img', img)

        if image == distorted_image_1:
            cv2.imwrite('outputs/chess.png', img)
        cv2.waitKey(1)
cv2.destroyAllWindows()

########## CAMERA CALIBRATION #####################
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Camera Calibrated: ', ret)
print("Camera matrix : \n", cameraMatrix)
print("Distortion coefficient: \n", dist)
# print("Rotation Vectors:  \n", rvecs)
# print("Translation Vectors:  \n", tvecs)

img = cv2.imread(distorted_image_1)
cv2.imwrite('outputs/original.png', img)
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

##################### Undistort Image ####################
''' Sample 1'''
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('outputs/undistorted_calibresult.png', dst)

#####################  Undistort Image with Remapping ####################
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('outputs/undistorted_calibresult_mapping.png', dst)

##################### Reprojection error ##################
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
