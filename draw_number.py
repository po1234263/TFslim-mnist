import numpy as np
import cv2
import os
import re
import test_for_input

drawing = False
mode = True
ix, iy = -1, -1

def draw_num(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.circle(img, (x, y), 3, (255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False


img = np.zeros((140, 140, 1), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_num)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_NEAREST)
        image_dir = os.listdir('image_set')
        pattern = '[0-9]+'
        num_in_image_set = []
        for im in image_dir:
            num_in_image_set.append(re.findall(pattern, im)[0])
        num_in_image_set = np.array(num_in_image_set, dtype = np.int32)
        num_in_image_set.sort()
        if(len(num_in_image_set) > 0):
            idx = num_in_image_set[-1] + 1
        else:
            idx = 0
        cv2.imwrite('image_set/img_%d.jpg' %(idx), img)
        test_for_input.simple_test(img)
        break
cv2.destroyAllWindows()
