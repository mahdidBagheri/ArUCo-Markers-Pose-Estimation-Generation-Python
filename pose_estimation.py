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

def drawCube(frame, poly_top, poly_right, poly_left, poly_front, poly_behind, p1, p2, p3, p4, p5, p6, p7, p8):
    frame = cv2.fillPoly(frame, [poly_top], color=(0, 255, 0))

    if(p6[0,0] < p7[0,0]):
        frame = cv2.fillPoly(frame, [poly_left], color = (0,0,255))
    else:
        frame = cv2.fillPoly(frame, [poly_right], color = (0,0,255))

    if(p5[0,0] < p6[0,0]):
        frame = cv2.fillPoly(frame, [poly_front], color = (255,0,0))
    else:
        frame = cv2.fillPoly(frame, [poly_behind], color = (255,0,0))


    return frame


def drawAxis(frame, matrix_coefficients, rvec, tvec):
    frame = frame.copy()


    K = matrix_coefficients
    R = cv2.Rodrigues(rvec)[0]
    t = np.reshape(tvec, (3,1))
    P = np.concatenate((R,t),axis=1)
    z = 1

    X1 = np.array([[0],[0],[0], [1]])
    x_1 = (1/z) * np.matmul( np.matmul(K,P), X1 )
    x1 = int(x_1[0,0]/x_1[2,0])
    y1 = int(x_1[1,0]/x_1[2,0])
    p1 = np.array([[x1],
                   [y1]])
    frame = cv2.circle(frame, (x1,y1), 5, (255,0,0),-1 )

    X2 = np.array([[0.01], [0], [0], [1]])
    x_2 = (1 / z) * np.matmul(np.matmul(K, P), X2)
    x2 = int(x_2[0,0]/x_2[2,0])
    y2 = int(x_2[1,0]/x_2[2,0])
    p2 = np.array([[x2],
                   [y2]])
    frame = cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)

    X3 = np.array([[0], [0.01], [0], [1]])
    x_3 = (1 / z) * np.matmul(np.matmul(K, P), X3)
    x3 = int(x_3[0,0]/x_3[2,0])
    y3 = int(x_3[1,0]/x_3[2,0])
    p3 = np.array([[x3],
                   [y3]])
    frame = cv2.circle(frame, (x3, y3), 5, (255, 0, 0), -1)

    X4 = np.array([[0], [0], [0.01], [1]])
    x_4 = (1 / z) * np.matmul(np.matmul(K, P), X4)
    x4 = int(x_4[0,0]/x_4[2,0])
    y4 = int(x_4[1,0]/x_4[2,0])
    p4 = np.array([[x4],
                   [y4]])
    frame = cv2.circle(frame, (x4, y4), 5, (255, 0, 0), -1)

    oo = np.array([[x1],
                   [y1]])

    x_ = np.array([[x3 - x1],
                   [y3 - y1]])

    y_ = np.array([[x2 - x1],
                   [y2 - y1]])

    z_ = np.array([[x4 - x1],
                   [y4 - y1]])

    p5 = oo + z_ + y_
    p6 = oo + z_ + y_ + x_
    p7 = oo + z_ + x_
    p8 = oo + y_ + x_

    poly_top = np.concatenate((np.transpose(p4),
                               np.transpose(p5),
                               np.transpose(p6),
                               np.transpose(p7)))

    poly_right = np.concatenate((np.transpose(p1),
                               np.transpose(p2),
                               np.transpose(p5),
                               np.transpose(p4)))

    poly_left = np.concatenate((np.transpose(p3),
                               np.transpose(p7),
                               np.transpose(p6),
                               np.transpose(p8)))

    poly_front = np.concatenate((np.transpose(p2),
                               np.transpose(p5),
                               np.transpose(p6),
                               np.transpose(p8)))

    poly_behind = np.concatenate((np.transpose(p1),
                               np.transpose(p3),
                               np.transpose(p7),
                               np.transpose(p4)))

    frame = drawCube(frame, poly_top, poly_right, poly_left, poly_front, poly_behind, p1, p2, p3, p4, p5, p6, p7, p8)


    # print(f"{R[0,0]:0.2f}, {R[0,1]:0.2f}, {R[0,2]:0.2f}, {R[1,0]:0.2f}, {R[1,1]:0.2f}, {R[1,2]:0.2f}, {R[2,0]:0.2f}, {R[2,1]:0.2f}, {R[2,2]:0.2f}, ")

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