# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:49:37 2020

@author: gordon
"""
import numpy as np
import read_write as rw
def to_real_xyz(yi,xi,d):
    x=-(xi/640-1.00078)*0.69243*d
    y=-(yi/360-1.00139)*0.38887*d
    z=d
    return  x,y,z
def dis_two_p_square(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[1]-b[1])**2)
def remove_bad_point(points):
    re=[]
    npoint=points[:,0:3]
    avg=np.average(npoint,axis=0)
    tf=np.linalg.norm(npoint-avg,axis=1)<500
    for i in range(len(tf)):
        if(tf[i]):
            re.append(points[i])
    return np.asarray(re)
def unit_vector(vector):
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def points_transform(vx,vy,vz,d,points):
    re=[]
    mat=np.asarray([vx/np.linalg.norm(vx),vy/np.linalg.norm(vy),vz/np.linalg.norm(vz)])
    invmat=np.linalg.inv(mat)
    for p in points:
        re.append(np.matmul((np.asarray(p)-d),invmat))
    return re
def non_zero_mean(frames):
    arr=np.asarray(frames)
    avg=np.sum(arr,axis=0)/np.sum( arr!=0 ,axis=0)
    return np.nan_to_num(avg,0)

def PCA(data, correlation = False, sort = True):
    mean = np.mean(data, axis=0)
    data_adjust = data - mean
#: the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T) 
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if sort:
    #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
    return eigenvalues, eigenvectors
def best_fitting_plane(points):
    w, v = PCA(points)
#: the normal of the plane is the last eigenvector
    normal = v[:,2]
#: get a point from the plane
    point = np.mean(points, axis=0)
    
    a, b, c = normal
    d = -(np.dot(normal, point))
    return (point, normal , a, b, c, d)
def findvec(gpmask,reddot,depth,picture,debug=False):
    points=rw.savetopointcloud("test.ply",gpmask*depth,picture,re=True)
    print("len before remove",len(points))
    points=remove_bad_point(points)
    print("len after remove",len(points))
    point,Normalvector,a,b,c,d=best_fitting_plane(points[:,0:3])
    if(Normalvector[2]<0):
        Normalvector=-Normalvector
    line=np.expand_dims(np.arange(-200,200),axis=1)*np.asarray([Normalvector])+np.asarray(point)
    if(debug):
        rw.savepointtofile(line.tolist(),"line.ply")
        rw.savepointtofile(points,"plane.ply")
    up=0
    upid=-1
    down=2000
    downid=-1
    left=2000
    leftid=-1
    right=0
    rightid=-1
    for i in range(len(reddot)):
        if(reddot[i][0]<left):
            leftid=i
            left=reddot[i][0]
        if(reddot[i][0]>right):
            rightid=i
            right=reddot[i][0]
        if(reddot[i][1]<down):
            downid=i
            down=reddot[i][1]
        if(reddot[i][1]>up):
            upid=i
            up=reddot[i][1]
    v1=np.asarray(to_real_xyz(reddot[upid][0],reddot[upid][1],0.1))
    ans1=v1*(-(d)/np.sum(v1*np.asarray([a,b,c])))
    v1=np.asarray(to_real_xyz(reddot[downid][0],reddot[downid][1],0.1))
    ans2=v1*(-d/np.sum(v1*np.asarray([a,b,c])))
    v1=np.asarray(to_real_xyz(reddot[leftid][0],reddot[leftid][1],0.1))
    ans3=v1*(-d/np.sum(v1*np.asarray([a,b,c])))
    v1=np.asarray(to_real_xyz(reddot[rightid][0],reddot[rightid][1],0.1))
    ans4=v1*(-d/np.sum(v1*np.asarray([a,b,c])))     
    print(angle_between((ans1-ans2),(ans3-ans4))/np.pi*180,np.linalg.norm((ans1-ans2)),np.linalg.norm((ans3-ans4)))
     
    return [(ans1-ans2)/np.linalg.norm(ans1-ans2),(ans3-ans4)/np.linalg.norm(ans3-ans4),Normalvector,(ans1+ans2+ans3+ans4)/4]
def transform(allpoints,vx,vy,vz,mid,debug=True):
    
    point_after_transform=points_transform(vx,vy,vz,mid,allpoints[:,0:3])
    point_after_transform_new = []
    allcolor=[]
    for  i in range(len(point_after_transform)):
        if(np.linalg.norm(point_after_transform[i])<4000):
            point_after_transform_new.append(point_after_transform[i])
            allcolor.append(allpoints[i,3:6])
    return np.concatenate((point_after_transform_new,allcolor),axis=1)
    

    