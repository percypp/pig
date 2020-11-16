# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:22:33 2020

@author: gordon
"""
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
sys.path.append(".")
import pointcloud as pc
class connect_c:
    def __init__(self,ID):
        self.ID=ID
        self.pnum=1
        self.weight=0
        self.i=0
        self.j=0
        self.d=0
    def add_point(self,i,j,d):
        self.i+=i
        self.j+=j
        self.d+=d
        self.weight+=1/(d+500)
        self.pnum+=1
    def finish(self):
        self.i=self.i/self.pnum
        self.j=self.j/self.pnum
        self.d=self.d/self.pnum
        return self.weight
    def show(self):
        print("i",self.i,"j",self.j,"d",self.d)
        
        
    
def where_is_pig(mask,red_img):
    num_labels_p, labels_im_p=cv2.connectedComponents(mask)
    allCandidate=[]
    for i in range(num_labels_p):
        allCandidate.append(0)
    print(num_labels_p)
    print(labels_im_p.shape)
    for i in range(labels_im_p.shape[0]):
        for j in range(labels_im_p.shape[1]):
            label=labels_im_p[i,j]
            if(label==0):
                continue
            if(allCandidate[label]==0):
                allCandidate[label]=connect_c(label)
            else :
                allCandidate[label].add_point(i,j,red_img[i,j])
    bignum=0
    for i in range(1,num_labels_p):
        tnum=allCandidate[i].finish()
        #print(tnum)
        if(tnum>bignum):
            bigc=allCandidate[i]
            bignum=tnum
    return [bigc,labels_im_p==bigc.ID]
def near_pig_center(new_mask,depth_img,ci,cj,cd):
    re_mask=np.zeros(depth_img.shape)
    print("pig:xyd",ci,cj,cd)
    center=pc.to_real_xyz(ci,cj,cd)
    print(center)
    for i in range(new_mask.shape[0]):
        for j in range(new_mask.shape[1]):
            if(pc.dis_two_p_square(center,pc.to_real_xyz(i,j,depth_img[i,j]))<500000):
                re_mask[i,j]=1
    return re_mask*new_mask
def autosmoothing(depth):
    depth[depth>5000]=0
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            temp=0
            if(depth[i,j]!=0):
                temp=depth[i,j]
            if(depth[i,j]==0):
                depth[i,j]=temp
        for j in range(0,depth.shape[1],-1):
            temp=0
            if(depth[i,j]!=0):
                temp=depth[i,j]
            if(depth[i,j]==0):
                depth[i,j]=temp
    return depth
               
    plt.imshow(((depth==0)*200).astype(np.int32), interpolation='nearest')
def get_pig_mask(depth_f,bgdepth_f):
    depth_f[depth_f==0]=5000
    bgdepth_f[bgdepth_f>=5000]=5000
    mask=(depth_f.astype(np.int32)-bgdepth_f.astype(np.int32))<-50
    plt.imshow((mask*200).astype(np.int32), interpolation='nearest')
    pig_place,new_mask=where_is_pig(mask.astype(np.uint8),depth_f.astype(np.double))
    #plt.imshow((new_mask*200).astype(np.int32), interpolation='nearest')
    re_mask=near_pig_center(new_mask,depth_f,pig_place.i,pig_place.j,pig_place.d)
    #plt.imshow((re_mask*200).astype(np.int32), interpolation='nearest')
    return re_mask