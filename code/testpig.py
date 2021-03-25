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
import removebg_pig as rbgp
filename="23c2"
debug=False
camera=filename[-2:]
red=[]
rec=[]
tfr=[]
path="Users/percychien/Desktop/project/pig-main/test/"
path="D:/data/ALLBAG/view_three/"
rw.readallframes(path+filename+".bag",0,300,red,rec,tfr )
print("finish read")

def save(x):
    num = x+1
    x = (x*1000) + 1600
    for i in range(len(red)):
        if(tfr[i]-tfr[0]>x):
            index=i
            break
    vx,vy,vz,p=rw.load_vec(filename)
    background = np.load(filename+"_bg.npy")
    background=pc.non_zero_mean(red[2:10])
    #finalmask = rbgp.get_pig_mask(red[index], background)
    allp=rw.savetopointcloud("",red[index],rec[index],re=True)
    allp=pc.transform(allp,vx,vy,vz,p,debug=debug)
    cv2.imwrite(filename+"pig"+str(num)+".jpg", rec[index])
    rw.savepointtofile(allp,filename+"pig"+str(num)+".ply")
    print(num,"finish")

for i in range(5):
    save(i)