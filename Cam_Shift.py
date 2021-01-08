import cv2
import numpy as np

video = cv2.VideoCapture("Duck.mp4")
tracked_object = cv2.imread("Duck_photo.png")  # sub_image to be Tracked
# Take First Frame from Adobe Premiere and Open in MS Paint
# Determine the Dimensions of the tracked object
y = 119
x = 500
width = 526 - y
height = 873 - x
tracked_hsv = cv2.cvtColor(tracked_object, cv2.COLOR_BGR2HSV)  # Convert image to HSV Colors
tracked_hist = cv2.calcHist([tracked_hsv], [0], None, [180], [0, 180])
# For this application we only need the histogram of Hue.
# It ranges from 0 to 179 (180 values)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while True:
    _, frame = video.read()
    video_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert Video to HSV
    # Performing back projection on the histogram of the Region of interest to find what has a similar histogram in the
    # video
    mask1 = cv2.calcBackProject([video_hsv], [0], tracked_hist, [0, 180], 1)
    rect, tracking_window = cv2.CamShift(mask1, (x, y, width, height), term_criteria) # Apply the Cam shift algorithm
    # Define the Parameters of the Tracking Window
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    cv2.polylines(frame, [points], True, (0, 0, 255), 2)  # To make the rectangle rotate with the object

    cv2.imshow("mask", mask1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(120) # Delay 40ms between frames
    if key == 27:  # Press Esc button to exit program
        break
video.release()
cv2.destroyAllWindows()
