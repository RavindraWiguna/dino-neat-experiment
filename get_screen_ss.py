import cv2
from PIL import ImageGrab
import numpy as np

def nothing(x):
    pass

def showTrackbar():
    #Create Trackbar
    cv2.namedWindow("Track")
    cv2.createTrackbar("x1", "Track", 0, 1920, nothing)
    cv2.createTrackbar("y1", "Track", 0, 1080, nothing)
    cv2.createTrackbar("w", "Track", 128, 1920, nothing)
    cv2.createTrackbar("h", "Track", 256, 1080, nothing)
    # cv2.createTrackbar("Method", "Track", 0, 1, nothing)


def main():
    isRun = True
    showTrackbar()
    while isRun:
        #get value from trackbar
        x1 = cv2.getTrackbarPos("x1", "Track")
        y1 = cv2.getTrackbarPos("y1", "Track")
        w = cv2.getTrackbarPos("w", "Track")
        h = cv2.getTrackbarPos("h", "Track")
        ss_region = (x1, y1, (x1+w), (y1+h))
        ss_img = ImageGrab.grab(ss_region)
        # print(type(ss_img))
        frame = np.array(ss_img)[:, :, 0]
        # print(frame.shape)
        cv2.startWindowThread()
        cv2.imshow('test', frame)
        # print("don't close it with mouse!")
        k = cv2.waitKey(1) & 0xFF
        if(k == 27):
            isRun = False
            cv2.destroyAllWindows()
            break

    

if __name__=="__main__":
    main()