#-*-coding: utf-8-*-
import os
import cv2
import time
import math
import decimal
import argparse
import colorsys
import operator
import itertools
import matplotlib
import statistics
import numpy as np
import open3d as o3d
from scipy import signal
from scipy import optimize
from math import sin, cos, pi
import matplotlib.pylab as plt
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


ridge = []
bodycurve = []
midHeight = []
botHeight = []
sArr = np.array(0)
rArr = np.array(0)
outer = np.array(0)
xF, yF, zF = int(), int(), int()
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input")
args = vars(parser.parse_args())

def display(pts):
	u = []
	for i in range(1000):
		u.append([i,0,0])
		u.append([0,i,0])
		u.append([0,0,i])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(u+pts)
	o3d.visualization.draw_geometries([pcd])			


def readpoints(argv):
	f = open(args["input"])
	lines = f.readlines()
	pts = []
	Begin = False
	property = -1
	for line in lines:
		l = line[:-1].split()
		if 'end_header' in line:
			Begin = True
		if 'property' in line:
			property += 1
		if len(l) > 5 and Begin:
			x, y, z = float(l[0]), float(l[1]), float(l[2])
			pts.append([x, y, z])
	f.close()
	return pts

def distance(A, B):
	return np.linalg.norm(list( map(operator.sub, A, B)))

def adjust(pts):
	
	pts.sort(key = lambda x: x[0])
	halves = pts[len(pts)//2][0]
	horizontal = []
	for p in pts:
		if abs(p[0] - halves) < 1:
			horizontal.append(p)

	dist = 0
	pair = []
	for each in horizontal:
		for another in horizontal:
			if another == each:continue
			if distance(each, another) > dist:
				pair = []
				dist = distance(each, another)
				pair.append(each)
				pair.append(another)

	# 計算旋轉矩陣
	a, b = pair
	diff = [d/int(distance(a, b)) for d in list( map(operator.sub, a, b))]
	Cos = np.dot([0,1,0], diff)/np.linalg.norm(diff)
	Sin = math.sqrt(1-Cos**2)
	R = np.array([[1,0,0],[0,Cos,-Sin],[0,Sin,Cos]])
	pts = np.matmul(R, np.array(pts)[:].T).T
	return pts.tolist()

def transection(pts): # 找出體高中心橫切面
	pts.sort(key = lambda x: x[0], reverse=True)
	pts.sort(key = lambda x: x[1], reverse=True)
	# 沿身長方向排序並取出頭尾座標值
	head, tail = int(pts[0][1]), int(pts[-1][1])
	X = []
	global ridge
	global midHeight
	global botHeight
	minima, maxima = 1e4, -1e4
	middle = int()
	for j in range(head, tail, -1):
		indices = [x for x, y, _ in pts if abs(y-j) < 10]
		if len(indices) > 1:
			indices.sort()
			minima = indices[0]  # bottom
			maxima = indices[-1] # top
			X.append(j)
			midHeight.append([(minima+maxima)/2,j]) #中部高度
			botHeight.append([minima,j]) #底部高度
			ridge.append([maxima, j])
		if j == (head + tail)//2:
			middle = (indices[0] + indices[-1])/2
	return X, middle

def arclength(arc):
	arc_length = 0
	for i in range(1, len(arc)-1):
		A, B = arc[i-1], arc[i]
		x = np.linalg.norm(A[0]-B[0])
		y = np.linalg.norm(A[1]-B[1])
		z = np.linalg.norm(A[2]-B[2])
		norm = x ** 2 + y ** 2 + z ** 2
		arc_length += math.sqrt(norm)
	return arc_length

def centerline(pts): # 找出中心線 
	X, middle = transection(pts)
	Y = np.array([middle]*len(X))

	mesh = np.array(pts)
	droite, gauche = [], []

	# 計算對稱分群
	for tup in list(zip(X, Y)):
		x = np.where(abs(mesh[:,0].astype(int) - int(tup[1].item())) < 20)
		y = np.where(abs(mesh[:,1].astype(int) - int(tup[0])) < 20)
		if np.intersect1d(x, y).shape[0]:
			right, left = -1e3, 1e3
			for k in [*np.intersect1d(x, y)]:
				if mesh[k,2] > right:
					right = mesh[k,2]
				if mesh[k,2] < left:
					left = mesh[k,2]
			droite.append([tup[1].item(), tup[0], right])
			gauche.append([tup[1].item(), tup[0], left])

	# 計算分群點中間點群
	cline = list()
	global outer
	outer = np.array(droite + gauche)
	for a, b in zip(droite, gauche):
		cline.append([e/2 for e in list( map(operator.add, a, b) )])
	cline = np.array(cline)

	# 把中心點集進行迴歸
	lin_reg = LinearRegression()
	cube_reg = PolynomialFeatures(degree = 3)
	cline_y = cube_reg.fit_transform(cline[:,1].reshape(-1,1))
	lin_reg.fit(cline_y, np.array(cline[:,2]).reshape(-1,1))
	cline_z = lin_reg.predict(cube_reg.fit_transform(cline[:,1].reshape(-1,1)))

	newcline = [] # 計算中心線(水平)
	for rx, ry, rz in zip(cline[:,0], cline[:,1], cline_z):
		newcline.append([rx, ry, rz.item()])
	
	ridgeline = []
	Bouldering  = False
	for k in range(len(ridge)-1):
		if Bouldering == False and ridge[k][1] == newcline[0][1]:
			Bouldering = True
		if Bouldering == True:	
			ridgeline.append(ridge[k])
	spine = []
	for hill, plain in zip(ridgeline, newcline):
		spine.append([hill[0], hill[1], plain[2]])

	# 針對交集點集進行迴歸
	quartic_reg = PolynomialFeatures(degree = 6)
	spine_y = quartic_reg .fit_transform(np.array(spine)[:,1].reshape(-1,1))
	lin_reg.fit(spine_y, np.array(spine)[:,0].reshape(-1,1))
	spine_x  = lin_reg.predict(quartic_reg.fit_transform(np.array(spine)[:,1].reshape(-1,1)))
	global bodycurve
	for x , arc in zip(spine_x, spine):
		y, z = arc[1], arc[2]
		bodycurve.append([x.item(), y ,z])

	return bodycurve

def removefoot(pts): # 體高與胸深的輔助函式
	pts = np.array(pts)
	pts = pts[pts[:,1].argsort()]
	mean = np.mean(pts[:,0])
	std = np.std(pts[:,0])
	remains, removed = [], []
	for p in pts:
		if mean - p[0] < std:
			remains.append(p)
		else:	
			removed.append(p)
	return remains, removed

def fit_plane(cluster):
	# do fit
	cluster = np.array(cluster)
	tmp_A = []
	tmp_b = []
	for i in range(len(cluster)):
		tmp_A.append([cluster[i,0], cluster[i,1], 1])
		tmp_b.append(cluster[i,2])
	b = np.matrix(tmp_b).T
	A = np.matrix(tmp_A)
	fit = (A.T * A).I * A.T * b
	errors = b - A * fit
	residual = np.linalg.norm(errors)
	collected = []
	for i in range(len(cluster)):
		x, y, _ = cluster[i]
		z = fit[0]*x + fit[1]*y + fit[2]
		collected.append([x, y, z])
	return collected

def realloc(pts):
	# 將點雲重新分配
	mug = []
	curve = np.array(bodycurve[:])
	curve = curve[curve[:,1].argsort()].tolist()
	for i in range(1,len(curve)-1):
		if curve[i] != curve[i-1]:
			offset = list( map(operator.sub, curve[i], curve[i-1]))
			scatter = []
			for p in range(0,len(pts)-1,2):
				residual = list( map(operator.sub, pts[p], curve[i-1]))
				result = np.dot(offset, residual)
				if abs(result) < 1: scatter.append(pts[p])
			
			if len(scatter) > 3:
				collected = fit_plane(scatter[:])
				mug += collected
	return mug

def chestlocation(pts): # 計算後腳跟位置
	global bodycurve

	chest = []
	cond1, cond2 = False, False
	nofoot, foot3D = removefoot(pts[:])
	former1, former2 = 0, 0
	curve = bodycurve.copy()
	curve.sort(key = lambda c: c[1], reverse=True)
	for i in range(len(curve)-1):
		front, rear = curve[i:i+2]
		vector = list( map(operator.sub, front, rear))
		if front != rear:vector /= np.linalg.norm(vector)
		ring, band = [], []
		for j in range(len(nofoot)-1):
			offset = list( map(operator.sub, nofoot[j], rear))
			if abs(np.dot(vector, offset)) < 1:
				ring.append(nofoot[j])
			
		for j in range(len(pts)-1):
			offset = list( map(operator.sub, pts[j], rear))
			if abs(np.dot(vector, offset)) < 1:
				band.append(pts[j])
		# not yet解決頭方向的問題:
		if len(ring) != 0:
			ring.sort(key = lambda r: r[0])
			band.sort(key = lambda b: b[0])
			if ring[0][0] == band[0][0] and former1 > former2:
				cond1 = True
			former1, former2 = ring[0][0], band[0][0]
			norm = np.linalg.norm(ring[0][0]-ring[-1][0])
			if norm > 300 and norm < 400:
				cond2 = True
			if cond1 and cond2:
				chest += ring
				break
	#display(curve+chest+foot3D)
	return np.array(chest), foot3D

def widthline(index):
	line = []
	global bodycurve
	A, B = bodycurve[index-1], bodycurve[index]
	drc = [A[0]-B[0], A[1]-B[1], A[2]-B[2]]
	for x, y, z in outer:  
		a, b, c, = drc[0], drc[1], drc[2]
		plane = a * (x - B[0]) + b * (y - B[1]) + c * (z - B[2]) # 平面方程式
		if abs(plane) < 1:
			line.append([x, y, z])
	counter = 0
	line.sort(key = lambda e: e[2], reverse=True)
	while counter < 10 :
		une, deux, trois = line[0]
		uno, dos, tres = line[-1]
		middle = [e/7 for e in list( map(operator.add, [une, deux, trois], [6*uno, 6*dos, 6*tres]) )]
		line.append(middle)
		counter += 1
	return line[0], line[-1]

def bodylength(pts): #計算體長
	bodycurve = centerline(pts)
	savepointtofile(bodycurve, '.\\體長.ply')
	return arclength(bodycurve) 

def distanceToPlane(p0,n0,p):
	return np.dot(np.array(n0),np.array(p)-np.array(p0))

def residualsCircle(parameters,dataPoint):
	global sArr, rArr
	global xF, yF, zF
	r, s, Ri = parameters
	point  = [xF, yF, zF]
	planePointArr = s*sArr + r*rArr + np.array(point)
	distance = [ np.linalg.norm( planePointArr-np.array([x, y, z])) for x,y,z in dataPoint]
	res = [(Ri-dist) for dist in distance]
	return res

def residualsPlane(parameters,dataPoint):
    px,py,pz,theta,phi = parameters
    nx,ny,nz =sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)
    distances = [distanceToPlane([px,py,pz],[nx,ny,nz],[x,y,z]) for x,y,z in dataPoint]
    return distances

def Bestfitting(pts):
	# fit ellipse in 3D!
	px = np.sum(pts[:,0])/len(pts)
	py = np.sum(pts[:,1])/len(pts)
	pz = np.sum(pts[:,2])/len(pts)
	estimate = [px, py, pz, 0, 0] 
	bestFitValues, ier = optimize.leastsq(residualsPlane, estimate, args=(pts), maxfev=20000)
	global xF, yF, zF
	xF, yF, zF, tF, pF = bestFitValues
	point  = [xF, yF, zF]
	normal = [sin(tF)*cos(pF), sin(tF)*sin(pF), cos(tF)]
	# Fitting a circle inside the plane
	# creating two inplane vectors
	global sArr, rArr
	sArr = np.cross(np.array([1,0,0]),np.array(normal)) # assuming that normal not parallel x!
	sArr = sArr/np.linalg.norm(sArr)
	rArr = np.cross(sArr,np.array(normal))
	rArr = rArr/np.linalg.norm(rArr)
	
	estimateCircle = [0, 0, 335] # px, py, pz and zeta, phi
	bestCircleFitValues, ier = optimize.leastsq(residualsCircle, estimateCircle,args=(pts), maxfev=20000)
	rF, sF, RiF = bestCircleFitValues

	centerPointArr = sF*sArr + rF*rArr + np.array(point)
	synthetic = [list(centerPointArr + RiF*cos(phi)*rArr + RiF*sin(1.95*phi)*sArr) for phi in np.linspace(0, 2*pi, 15)]
	return synthetic

def bodygirth(pts): #計算胸圍&胸深
	bodycurve = centerline(pts)
	chest, foot = chestlocation(pts)
	synthetic = Bestfitting(chest)
	savepointtofile(synthetic, '.\\胸圍.ply')
	#display(synthetic + bodycurve + foot)
	
	synthetic.sort(key = lambda p: p[0])
	chest_depth = np.linalg.norm(synthetic[0][0]-synthetic[-1][0])
	synthetic.sort(key = lambda p: p[2])
	chest_width = np.linalg.norm(synthetic[0][2]-synthetic[-1][2])
	# 橢圓周長：L=2πb+4(a-b)
	bodygirth = pi * chest_depth + 4 * abs(chest_width-chest_depth)
	return chest_depth, bodygirth

def scapular(pts): #計算體高(順便計算體寬)
	global bodycurve
	curve = bodycurve[:]
	curve.sort(key = lambda p: p[1])
	frontwidth, rearwidth = 0, 0
	bodyheight = int()
	pts.sort(key = lambda p: p[0])
	lowest = pts[0][0]
	for i in range(1, len(curve)-1, 2):
		A, B = curve[i-1], curve[i]
		unit = list( map(operator.sub, B, A))/ distance(A, B)
		section = []
		for p in pts:
			offset = list( map(operator.sub, p, A))
			equation = np.dot(offset, unit)
			if abs(equation) < 1:
				section.append(p)
				pts.remove(p)
		if len(section) > 1:
			section.sort(key = lambda s: s[2]) # 根據z方向做排序、找出體寬
			keft, right = section[0][2], section[-1][2]
			width = abs(keft-right)
			if i < len(curve)//2 and width > frontwidth:
				frontwidth = width
				section.sort(key = lambda s: s[0], reverse=True) # 根據x方向做排序、找出體高
				bodyheight = section[0][0]-lowest
			elif i > len(curve)//2 and width > rearwidth:
				rearwidth  = width

	return frontwidth, rearwidth, bodyheight


def footgirth(pts): #計算管圍
	global bodycurve
	curve = bodycurve.copy()
	_, foot = removefoot(pts)
	curve.sort(key = lambda c: c[1], reverse=True)
	droite = []
	front, rear = [], []
	for f in foot:
		for c in range(0, len(curve)//2):
			t = curve[c]
			if abs(f[1]-t[1])<1 and (f[2] < t[2]):
				front.append(f.tolist())
		for c in range(len(curve)//2, len(curve)-1):
			t = curve[c]
			if abs(f[1]-t[1])<1 and (f[2] < t[2]):
				rear.append(f.tolist())
	# 計算前管圍
	allmean = []
	front.sort(key = lambda f: f[0])
	fbottom, ftop = math.floor(front[0][0]), math.ceil(front[-1][0])
	for value in range(fbottom, ftop):
		frontslice = np.array([f for f in front if abs(f[0]-value)<1])
		if len(frontslice) > 5:
			candidate = Bestfitting(frontslice)
			candidate.sort(key = lambda p: p[2])
			diameter = np.linalg.norm(candidate[0][2]-candidate[-1][2])
			allmean.append(pi * diameter)
	allmean.sort()
	frontgirth = min(allmean)
	for girth in allmean:
		if girth > 100 and girth < 200:
			frontgirth = girth
	allmean.clear()
	# 計算後管圍
	reargirth = int()
	rear.sort(key = lambda r: r[0])
	rbottom, rtop = math.floor(rear[0][0]), math.ceil(rear[-1][0])
	for value in range(rbottom, rtop):
		rearslice = np.array([r for r in rear if abs(r[0]-value)<1])
		if len(rearslice) > 5:
			candidate = Bestfitting(rearslice)
			candidate.sort(key = lambda p: p[2])
			diameter = np.linalg.norm(candidate[0][2]-candidate[-1][2])
			allmean.append(pi * diameter)
	
	reargirth = min(allmean)
	for girth in allmean:
		if girth > 100 and girth < 200:
			reargirth = girth
	return frontgirth, reargirth

def measure(pts):
	print('體長長度 : ' + "{:.2f}".format(bodylength(pts)/10) + 'cm')
	frontwidth, rearwidth, bodyheight = scapular(pts[:])
	print('前寬長度 : ' + "{:.2f}".format(frontwidth/10) + ' cm')
	print('後寬長度 : ' + "{:.2f}".format(rearwidth/10) + ' cm')
	print('體高長度 : ' + "{:.2f}".format(bodyheight/10) + ' cm')
	chest_depth, Bgirth = bodygirth(pts)
	print('胸深高度 : ' + "{:.2f}".format(chest_depth/10) + ' cm')
	print('胸圍長度 : ' + "{:.2f}".format(Bgirth/10) + ' cm')
	frontgirth, reargirth = footgirth(pts)
	print('前管圍長度 : ' + "{:.2f}".format(frontgirth/10) + ' cm')
	print('後管圍長度 : ' + "{:.2f}".format(reargirth/10) + ' cm')

def savepointtofile(pts, file_name):
	points = list()
	for i in range(len(pts)-1):
		x, y, z = pts[i][0], pts[i][1], pts[i][2]
		r, g, b = 0, 255, 0
		points.append("%f %f %f %d %d %d\n"%(x, y, z, r, g, b))
	file = open(file_name,"w")
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

def main():
	pts = readpoints(args["input"])
	pts = adjust(pts)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pts)
	o3d.io.write_point_cloud(".\\adjust.ply", pcd)
	measure(pts)

if __name__ == '__main__':
	main()
