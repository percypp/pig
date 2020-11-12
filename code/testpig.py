# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:43:33 2020

@author: gordon
"""
import cv2
import sys
import numpy as np
sys.path.append(".")
import read_write as rw
import img_tool as imt
import greenplane_reddot as gprd
import pointcloud as pc
filename="20c2"
debug=False
camera=filename[-2:]
red=[]
rec=[]
tfr=[]
rw.readallframes("C:/Users/gordon/Documents/ALLBAG/view_one/"+filename+".bag",0,100,red,rec,tfr )
print("finish read")
for i in range(100):
    if(tfr[i]-tfr[0]>3100):
        index=i
        break
allp=rw.savetopointcloud("",red[index],rec[index],re=True)
vx,vy,vz,p=rw.load_vec(filename)
allp=pc.transform(allp,vx,vy,vz,p,debug=debug)
rw.savepointtofile(allp,filename+"pig.ply")