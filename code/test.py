# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:44:07 2020

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
camera=filename[-2:]
debug=True
red=[]
rec=[]
tfr=[]
rw.readallframes("C:/Users/gordon/Documents/ALLBAG/view_one/"+filename+".bag",0,50,red,rec,tfr )
print(tfr)
bg=imt.getbackground(red[0:2])
rbg=imt.remove_background(bg)
red=[]
rec=[]
tfr=[]
rw.readallframes("C:/Users/gordon/Documents/ALLBAG/view_one/board"+camera+".bag",0,50,red,rec,tfr )
gpd,gpc=rbg.remove_bg(red[20],rec[20])
gp=imt.green_mask(gpc)
rd=imt.red_mask(gpc)

if(debug==True):
    cv2.imshow('My Image1', rec[1])
    cv2.imshow('My Image2', gp*200)
    cv2.imshow('My Image3', rd*200)

gp,reddot=gprd.find_green_plane_red_dot(gp,rd,gpc,debug)
re=pc.findvec(gp,reddot,red[29],rec[29],debug=debug)
vx=re[0]
vy=re[1]
vz=re[2]
p=re[3]
print(vx,vy,vz)
allp=rw.savetopointcloud("",red[20],rec[20],re=True)
allp=pc.transform(allp,vx,vy,vz,p,debug=debug)
if(debug):
    rw.savepointtofile(allp,"board"+camera+"after.ply")
rw.save_vec(filename,vx,vy,vz,p)
np.save(filename+"_bg",bg)
print(rw.load_vec(filename))
if(debug):
    cv2.waitKey(0)
    cv2.destroyAllWindows()