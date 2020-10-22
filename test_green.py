import cv2
import numpy as np
frame = cv2.imread('b.png')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_red = np.array([70,100,100]) 
upper_red = np.array([88, 200, 200]) 
mask = cv2.inRange(hsv, lower_red, upper_red) 
res = cv2.bitwise_and(frame,frame, mask= mask) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))

hole=res.copy()
cv2.floodFill(hole,None,(0,0),255) # 找到洞孔
hole=cv2.bitwise_not(hole)
filledEdgesOut=cv2.bitwise_or(res,hole)
filledEdgesOut=cv2.add(res,hole)
closing = cv2.morphologyEx(filledEdgesOut, cv2.MORPH_CLOSE, kernel, iterations=2)
closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
filledEdgesOut = cv2.cvtColor(filledEdgesOut, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result_b.jpg', closing)
cv2.imwrite('result_b1_withoutclose.jpg', filledEdgesOut)
# cv2.imshow("filledEdgesOut",filledEdgesOut)
# cv2.imshow('closing', closing)
# cv2.imshow('frame',frame) 
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)
# cv2.imshow('hsv',hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()