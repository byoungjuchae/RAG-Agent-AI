
import cv2
import numpy as np

def calibrate_camera(images_paths, nx=9, ny=6):
    objpoints = []
    imgpoints = []

    for image_path in images_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret_val, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)

        if ret_val:
            objpoints.append(np.zeros((ny * 7, 3), dtype=np.float32))
            imgpoints.append(corners)

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

    return camera_matrix, distortion_coeffs

def undistort_images(images_paths, nx=9, ny=6):
    objpoints = []
    imgpoints = []

    for image_path in images_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret_val, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)

        if ret_val:
            objpoints.append(np.zeros((ny * 7, 3), dtype=np.float32))
            imgpoints.append(corners)

    camera_matrix, distortion_coeffs = calibrate_camera(images_paths, nx, ny)
    image_undistorted = cv2.undistort(image, camera_matrix, distortion_coeffs, None, camera_matrix)

    return image_undistorted

