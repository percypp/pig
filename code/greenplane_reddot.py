# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:20:36 2020

@author: gordon
"""
import cv2
import numpy as np
class Points():
    def __init__(self,inid=0,place=(),num=0):
        self.place=place
        self.num=num
        self.id=inid
    def add(self,i,j):
        self.place[0]+=i
        self.place[1]+=j
        self.num+=1
    def __lt__(self, other):
        return self.num<other.num
    def __gt__(self,other):
        return self.num>other.num
    def __ne__(self,other):
        return self.num!=other.num
    def __eq__(self,other):
       # print("in")
        return self.num==other.num
class Connect_component():
    def __init__(self,rdnum,gpnum):
        self.gp=[]
        self.rd=[]
        for i in range(rdnum+1):
            self.rd.append(Points())
        for i in range(gpnum+1):
            self.gp.append(Points())
    def addrd(self,rid,i,j):
        if(self.rd[rid]==Points()):
            #print("add rd")
            self.rd[rid]=Points(rid,[i,j],1)
        else :
            self.rd[rid].add(i,j)
    def addgp(self,gid,i,j):
        if(self.gp[gid]==Points()):
            self.gp[gid]=Points(gid,[i,j],1)
        else :
            #print("re")
            self.gp[gid].add(i,j)    


    def gplane(self):
        return max(self.gp)
    def rdot(self,gplane):
        self.rd.sort(reverse=True)
        re=[]
        for i in range(len(self.rd)):
            if(len(self.rd[i].place)==0):
                continue
            core=np.asarray(self.rd[i].place)/(self.rd[i].num)
            core=core.astype(int)
           # print(core)
            if(gplane[core[0],core[1]]!=0 and self.rd[i].num>5):
                re.append(core)
            if(len(re)==4):
                break 
        return re


  
        
        
    
def find_green_plane_red_dot(gpfilter,rdfilter,p,debug=True):
    num_labels_g, labels_im_g=cv2.connectedComponents(gpfilter)
    num_labels_r, labels_im_r=cv2.connectedComponents(rdfilter)
    #print((gpfilter+rdfilter)>0)
    cc=Connect_component(num_labels_r,num_labels_g)
    #find green plane and red dot exist in same labels_im_con class
    #for i in range(gpfilter.shape[0])

    for i in range (labels_im_g.shape[0]):
        for j in range(labels_im_r.shape[1]):
            if(labels_im_g[i,j]!=0): # 
                cc.addgp(labels_im_g[i,j],i,j)
            if(labels_im_r[i,j]!=0): # 
                cc.addrd(labels_im_r[i,j],i,j)
    gpid=cc.gplane().id #find connect_component which content green plane and 
    gplane=(labels_im_g==gpid)
    im2, contours, hierarchy = cv2.findContours(gplane.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(np.sum(im2))
    hull=[cv2.convexHull(contours[0], False)]
    print(hull)
    drawing = np.zeros((gplane.shape[0], gplane.shape[1], 3), np.uint8)
    color = (0, 255, 0);
    cv2.drawContours(drawing, hull, 0, color, -1)
    print(np.sum(drawing/255))

    redDot=cc.rdot(np.sum(drawing,axis=2))
    print(redDot)
    if(debug==False):
        return (gplane,redDot)
    mask=np.asarray([gplane])
    mask=np.moveaxis(mask,0,-1)
    img= (p*mask).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)
    
    if(debug):
        cv2.imshow('My Image ori',cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
        cv2.imshow('My Image mask',img)
        cv2.imshow('My Image mask2',drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (gplane,redDot)

