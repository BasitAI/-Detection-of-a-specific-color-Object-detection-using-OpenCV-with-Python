# Detection of a specific color Object detection using OpenCV with Python

# The color detection process is mostly in demand in computer vision.
# A color detection algorithm identifies pixels in an image that match a specified color or color range.
# The color of detected pixels can then be changed to distinguish them from the rest of the image.
# This process can be easily done using OpenCV.
# In this Red Object Detection Using OpenCV

# **********/***********/***********/*****************/**************/************
# Import the OpenCV Library
import cv2
import numpy as np

# Define Video Capture object
cap = cv2.VideoCapture(0)

while True:

    # return a true value if the frame exists otherwise False
    Success,frame = cap.read()

    # Convert color format from BGR to HSV
    imgHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define Range of Red color in HSV
    L_Bound = np.array([160, 50, 50])  # Setting the Red lower limit
    U_Bound = np.array([180, 255, 255])  # Setting the Red upper limit
    # Threshold the HSV image to detect only Red Color.This will give the color to mask.
    mask = cv2.inRange( imgHsv, L_Bound, U_Bound)
    # Bitwise AND mask and Original Image
    Red = cv2.bitwise_and(frame, frame, mask= mask)
    # Horizontal  Stacking
    hStack = np.hstack([frame, Red])
    # To display the Red object output along with original image in the Horizontal Stack
    cv2.imshow('Red Object Detection', hStack)

    if cv2.waitKey(1) == 30:
        break

cap.release()

cv2.destroyAllWindows()
