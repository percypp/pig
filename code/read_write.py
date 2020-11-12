# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:36:57 2020

@author: gordon
"""
import pyrealsense2 as rs

import time
import numpy as np
def to_real_xyz(yi,xi,d):
    x=-(xi/640-1.00078)*0.69243*d
    y=-(yi/360-1.00139)*0.38887*d
    z=d
    return  x,y,z
def savetopointcloud(file_name,depth_frame,color_frame,re=False):
    points=[]
    for y in range(depth_frame.shape[0]):
        for x in range(depth_frame.shape[1]):
            d=depth_frame[y,x]
            if(d==0):
                continue
            xyz=to_real_xyz(y,x,d)
            color=color_frame[y,x,:]
            if(re==False):
                points.append("%f %f %f %d %d %d\n"%(xyz+(color[2],color[1],color[0])))
            else:
                points.append(np.asarray((xyz+(color[0],color[1],color[2]))))
    if(re==True):                  
        return np.asarray(points)
    file=open(file_name,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

def savepointtofile(p,file_name):
    points=[]
    for i in p:
        i=tuple(i)
        if(len(i)==3):
            i=i+(0,0,255)
        points.append("%f %f %f %d %d %d\n"%i)
    file=open(file_name,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
%s
'''%(len(points),"".join(points)))
    file.close()
def readallframes(path,start,maxframes,redepth,recolor,timestamp):
    time.sleep(0.01)
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path)
    #align_to = rs.stream.color # or also depth
    #align = rs.align(align_to)
    
    framenumber=0
    # Start streaming from file
    pipeline.start(config)
    count=0
    frame_num=0
    tmp_frame_num=0
    i=0
    while True:
        frames = pipeline.wait_for_frames()
       
        # Get depth frame
        color_frame=frames.get_color_frame()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        # Colorize depth frame to jet colormap
        depth_frame=frames.get_depth_frame()
        if(i>start):
            redepth.append(np.asarray(depth_frame.get_data()).copy())
            recolor.append(np.asarray(color_frame.get_data()).copy())
            tmp_frame_num=depth_frame.get_frame_number()
            timestamp.append(depth_frame.get_timestamp())
        i+=1
        if(tmp_frame_num<frame_num):
            break
        else:
            frame_num=depth_frame.get_frame_number()
    
       # print("depth: ",depth_frame.get_frame_number(),"color: ",color_frame.get_frame_number())
        count+=1
        if(count>maxframes):
            break
   # return redepth,recolor
def save_vec(filename,vx,vy,vz,p):
    a = np.asarray([ vx, vy, vz ,p])
    np.save(filename+"_vec",a)
def load_vec(filename):
    a=np.load(filename+"_vec.npy")
    return [a[0,:],a[1,:],a[2,:],a[3,:]]