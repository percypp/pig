#!/usr/bin/python
# file encoding: utf-8
import os
import cv2
import sys
import time
import shutil
import threading
import itertools
import numpy as np
import tkinter as tk
import pyrealsense2 as rs
from PIL import Image, ImageTk
from tkinter import filedialog

def get_critiacal_frames(start, lower, upper, cframes):
	beforemask=1
	totallightup=0
	#1280 * 720
	y = 0
	x = int(1280 / 3)
	w = int(1280 / 3)
	h = int(720 / 3)
	for i in range(len(cframes)):
		cframes[i] = cframes[i][y:y+h, x:x+w]
	for i in range(30):
		nowmask=  cv2.inRange(cframes[i], lower, upper)/255
		totallightup+=(nowmask-beforemask>0)
		beforemask=nowmask
	avgmask=np.sum(totallightup)/30
	print("avg: ",avgmask)
	for i in range(max(start,30),len(cframes)):
		nowmask=  cv2.inRange(cframes[i], lower, upper)/255
		lightup=(nowmask-beforemask>0)
		if(np.sum(lightup)>2*avgmask):  #need more check
			return i 
		beforemask=nowmask
	print("fail to find critical")
	return 0

def readbag(filename):
	pipeline = rs.pipeline()
	config = rs.config()
	rs.config.enable_device_from_file(config, filename)
	
	pipeline.start(config)

	colorizer = rs.colorizer()

	align_to = rs.stream.color
	align = rs.align(align_to)

	previous_timestamp = 0
	collected_frames = list()

	while True:
		
		frames = pipeline.wait_for_frames()

		aligned_frames = align.process(frames)
	
		color_frame = aligned_frames.get_color_frame()
		
		depth_frame = aligned_frames.get_depth_frame()

		current_timestamp = int(color_frame.get_timestamp())

		color_image = np.asanyarray(color_frame.get_data())

		depth_image = np.asanyarray(depth_frame.get_data())

		collected_frames.append(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

		cv2.imshow('', cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

		key = cv2.waitKey(1)

		if key & 0xFF == ord('q') or key == 27:
			cv2.destroyAllWindows()
			return collected_frames

		if previous_timestamp > current_timestamp:
			cv2.destroyAllWindows()
			return collected_frames

		previous_timestamp = current_timestamp

def refresh_image(canvas, img, image_id):
    try:    	
        pil_img = Image.fromarray(img).resize((400,400), Image.ANTIALIAS)
        Img = ImageTk.PhotoImage(pil_img)
        canvas.itemconfigure(image_id, image=Img)
    except IOError:  # missing or corrupt image file
        Img = None
    # repeat every half sec
    canvas.after(500, refresh_image, canvas, Img, image_id) 

def find_tablet(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green = np.array([60, 50, 90])
	upper_green = np.array([88, 200, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	res  = cv2.bitwise_and(frame, frame, mask = mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))

	hole = res.copy()
	cv2.floodFill(hole, None, (0, 0), 255)
	hole = cv2.bitwise_not(hole)
	filtedEdgesOut = cv2.bitwise_or(res, hole)
	filtedEdgesOut = cv2.add(res, hole)
	ero = cv2.erode(filtedEdgesOut, np.ones((2, 2), np.uint8), iterations = 10)
	closing = cv2.morphologyEx(ero, cv2.MORPH_CLOSE, kernel, iterations=2)
	closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
	cv2.imshow('clos', closing)
	cv2.waitKey(0)

def openfile():
	bagfile = filedialog.askopenfilename( initialdir=os.getcwd(), title="select file", filetypes=(("bag files", "*.bag"), ("all files", "*.*")))
	collected_frames = readbag(bagfile)
	critical = np.zeros((720, 1280), dtype=int)

	if "board" not in bagfile: # pig bag
		lower = np.asarray([250,250,250])
		upper = np.asarray([255,255,255])
		time = get_critiacal_frames(30, lower, upper, collected_frames.copy())
		critical = collected_frames[time]
		directory = os.getcwd() + "/point clouds/"
		if not os.path.exists(directory):
			os.makedirs(directory)
	else:              # green plane bag
		find_tablet(collected_frames[len(collected_frames)//2].copy())

		# 取第一個frame找出綠版和紅點
			

def main():
	root = tk.Tk()
	root.geometry("500x450+100+100")
	button = tk.Button(root, text="Open", command=openfile)
	button.pack()
	exit = tk.Button(root, text="Exit", command=root.destroy)
	exit.pack()
	root.mainloop()

if __name__ == '__main__':
	main()