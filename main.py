from PIL import ImageGrab
import cv2
import numpy as np
# import pyautogui

# pyautogui.keyDown(pyautogui.KEYBOARD_KEYS)
# print(pyautogui.KEY_NAMES) # they are the same

def main():
    isRun = True
    x, y, w, h = 77, 306, 471, 89
    x2, y2 = x+w, y+h
    goreg = cv2.imread('game_over.jpg')
    goreg = goreg[:, :, 0]
    while isRun:        
        ss_region = (x, y, x2, y2)
        ss_img = ImageGrab.grab(ss_region)
        # print(type(ss_img))
        frame = np.array(ss_img)[:, :, 0]
        gameOverBtn = frame[ 36:62, 237:265] # don't change the value (very sensitive)
        
        # print(frame.shape)
        # cv2.startWindowThread()
        cv2.imshow('test', frame)
        # print("don't close it with mouse!")
        k = cv2.waitKey(1) & 0xFF
        # check for esc pressed or died
        if(k == 27 or np.sum(cv2.subtract(gameOverBtn, goreg)) < 500):
            isRun = False
            cv2.destroyAllWindows()
            break

if __name__=="__main__":
    main()