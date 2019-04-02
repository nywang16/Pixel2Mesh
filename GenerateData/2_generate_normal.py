# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:22:12 2018

@author: wnylol
"""

import numpy as np
from scipy.spatial import ConvexHull

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def readFaceInfo(obj_path):
	vert_list = np.zeros((1,3), dtype='float32') # all vertex coord   
	face_pts = np.zeros((1,3,3), dtype='float32') # 3 vertex on triangle face
	face_axis = np.zeros((1,3,3), dtype='float32') # x y z new axis on face plane
	with open(obj_path, 'r') as f:
		while(True):
			line = f.readline()
			if not line:
				break
			if line[0:2] == 'v ':
				t = line.split('v')[1] # ' 0.1 0.2 0.3'
				vertex = np.fromstring(t, sep=' ').reshape((1,3))
				vert_list = np.append(vert_list, vertex, axis=0)
			elif line[0:2] == 'f ':
				t = line.split() # ['f', '1//2', '1/2/3', '1']
				p1,p2,p3 = [int(t[i].split('/')[0]) for i in range(1,4)]

				points = np.array([vert_list[p1], vert_list[p2], vert_list[p3]])
				face_pts = np.append(face_pts, points.reshape(1,3,3), axis=0)

				###!!!!!!!!!!!!###
				v1 = vert_list[p2] - vert_list[p1]	# x axis
				v2 = vert_list[p3] - vert_list[p1]
				f_n = np.cross(v1, v2)		# z axis, face normal
				f_y = np.cross(v1, f_n)		# y axis
				new_axis = np.array([unit(v1), unit(f_y), unit(f_n)])
				face_axis = np.append(face_axis, new_axis.reshape((1,3,3)), axis=0)

	face_pts = np.delete(face_pts, 0, 0)
	face_axis = np.delete(face_axis, 0, 0)
	return face_pts, face_axis

def generate_normal(pt_position, face_pts, face_axis):
	pt_normal = np.zeros_like(pt_position, dtype='float32')

	for points, axis in zip(face_pts, face_axis):
		f_org = points[0] # new axis system origin point
		f_n = axis[2] 
        
		face_vertex_2d = np.dot(points - f_org, axis.T)[:,:2]

		# check if a valid face	 
		n1,n2,n3 = [np.linalg.norm(face_axis[i]) for i in range(3)]
		if n1<0.99 or n2<0.99 or n3<0.99:
			continue
		# check if 3 point on one line
		t = np.sum(np.square(face_vertex_2d), 0)
		if t[0]==0 or t[1]==0:
			continue

		transform_verts = np.dot(pt_position - f_org, axis.transpose())
		vert_idx = np.where(np.abs(transform_verts[:,2]) < 6e-7)[0]

		for idx in vert_idx:
			if np.linalg.norm(pt_normal[idx]) == 0:
				p4 = transform_verts[idx][:2].reshape(1,2)
				pt_4 = np.append(face_vertex_2d, p4, axis=0)  
				hull = ConvexHull(pt_4)
				if len(hull.vertices) == 3:
					pt_normal[idx] = f_n * (-1)
	
	return np.hstack((pt_position, pt_normal))

if __name__ == '__main__':
    vert_path = '1a0bc9ab92c915167ae33d942430658c/model.xyz'
    mesh_path = '1a0bc9ab92c915167ae33d942430658c/model.obj'
    
    face_pts, face_axis = readFaceInfo(mesh_path)
    vert = np.loadtxt(vert_path)
    vert_with_normal = generate_normal(vert, face_pts, face_axis)
    np.savetxt(vert_path.replace('.xyz', '_normal.xyz'), vert_with_normal)
