
import numpy as np
import argparse
import imutils
import cv2

bg = cv2.imread('blank.png')
fg = cv2.imread('pig.png')

bgGray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
fgGray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

sub = bgGray.astype("int32") - fgGray.astype("int32")
sub = np.absolute(sub).astype("uint8")
cv2.imshow("sub",sub)
cv2.imwrite('sub.png', sub)
cv2.waitKey(10)

thresh = cv2.erode(sub, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=1)
rec, thresh1 = cv2.threshold(thresh,40,255,cv2.THRESH_BINARY)
cv2.imshow('tresh1', thresh1)
cv2.imwrite('tresh.png', thresh1)

cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 
(minX, minY) = (np.inf, np.inf)
(maxX, maxY) = (-np.inf, -np.inf)
 
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
 
	# reduce noise by enforcing requirements on the bounding box size
	if w > 10 and h > 10:
		# update our bookkeeping variables
		minX = min(minX, x)
		minY = min(minY, y)
		maxX = max(maxX, x + w - 1)
		maxY = max(maxY, y + h - 1)

# draw a rectangle surrounding the region of motion
cv2.rectangle(fg, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Output", fg)
cv2.imwrite('output.png', fg)
cv2.imshow("bg", bg)
cv2.waitKey(0)