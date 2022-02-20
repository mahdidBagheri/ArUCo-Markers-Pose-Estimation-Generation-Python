'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time


def drawAxis(frame, matrix_coefficients, rvec, tvec):
    frame = frame.copy()


    K = matrix_coefficients
    R = cv2.Rodrigues(rvec)[0]
    t = np.reshape(tvec, (3,1))
    P = np.concatenate((R,t),axis=1)
    z = 1

    X1 = np.array([[0],[0],[0], [1]])
    x1 = (1/z) * np.matmul( np.matmul(K,P), X1 )
    frame = cv2.circle(frame, (int(x1[0,0]/x1[2,0]), int(x1[1,0]/x1[2,0])), 5, (255,0,0),-1 )

    X2 = np.array([[0.01], [0], [0], [1]])
    x2 = (1 / z) * np.matmul(np.matmul(K, P), X2)
    frame = cv2.circle(frame, (int(x2[0, 0] / x2[2, 0]), int(x2[1, 0] / x2[2, 0])), 5, (255, 0, 0), -1)

    X3 = np.array([[0], [0.01], [0], [1]])
    x3 = (1 / z) * np.matmul(np.matmul(K, P), X3)
    frame = cv2.circle(frame, (int(x3[0, 0] / x3[2, 0]), int(x3[1, 0] / x3[2, 0])), 5, (0, 255, 0), -1)

    X4 = np.array([[0], [0], [0.01], [1]])
    x4 = (1 / z) * np.matmul(np.matmul(K, P), X4)
    frame = cv2.circle(frame, (int(x4[0, 0] / x4[2, 0]), int(x4[1, 0] / x4[2, 0])), 5, (0, 0, 255), -1)

    return frame

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    frame_2 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    # cv2.aruco_dict = cv2.aruco.Dictionary_get(1)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            frame = cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            frame_2 = drawAxis(frame, matrix_coefficients, rvec, tvec)

    return frame, frame_2

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output, output2 = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()