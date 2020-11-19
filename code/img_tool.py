# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:52:14 2020

@author: gordon
"""
import numpy as np
def getbackground(dep):
    allframe=np.asarray(dep)
    allframe[allframe==0]=9999
    bg=np.sort(allframe,axis=0)[(int)(len(allframe)*0.85)]
    return bg   
def green_mask(cframe):
    newp=cframe.astype(np.float)
    newp=newp*newp
    truecolor=np.asarray([[[0,1,0.4]]])
    mask=(np.sum((newp*truecolor),axis=2)/np.sqrt(np.sum(newp*newp,axis=2)*np.sum(truecolor*truecolor,axis=2)))>0.9
    mask2=(np.sum(cframe,axis=2)>20)
    return np.logical_and(mask,mask2).astype(np.uint8)
def red_mask(cframe):
    newp=cframe.astype(np.float)
    truecolor=np.asarray([[[1,0.2,0.2]]])
    newp=newp*newp
    mask=(np.sum((newp*truecolor),axis=2)/np.sqrt(np.sum(newp*newp,axis=2)*np.sum(truecolor*truecolor,axis=2)))>0.8
    mask2=(np.sum(cframe,axis=2)>20)
    return np.logical_and(mask,mask2).astype(np.uint8)

    
class remove_background:
    def __init__(self,bg):
        self.bg=bg.astype(np.int32)
        
    def remove_bg_mask(self,depf):
        return np.logical_and((depf.astype(np.int32)-self.bg)<-55,depf<3500)
    def remove_bg( self,d,c):
        mask=self.remove_bg_mask(d)
        return (d*mask,c*np.expand_dims(mask,axis=2))
