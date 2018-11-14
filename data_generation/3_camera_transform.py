# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:55:31 2018

@author: wnylol
"""

import numpy as np
import cPickle as pickle
import cv2
import os

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    return cam_mat, cam_pos

if __name__ == '__main__':
    
    vert_path = '1a0bc9ab92c915167ae33d942430658c/model_normal.xyz'
    vert = np.loadtxt(vert_path)
    position = vert[:, : 3] * 0.57
    normal = vert[:, 3:]

    view_path = '1a0bc9ab92c915167ae33d942430658c/rendering/rendering_metadata.txt'
    cam_params = np.loadtxt(view_path)
    for index, param in enumerate(cam_params):
        # camera tranform
        cam_mat, cam_pos = camera_info(param)

        pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
        nom_trans = np.dot(normal, cam_mat.transpose())
        train_data = np.hstack((pt_trans, nom_trans))
        
        #### project for sure
        img_path = os.path.join(os.path.split(view_path)[0], '%02d.png'%index)
        np.savetxt(img_path.replace('png','xyz'), train_data)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        
        X,Y,Z = pt_trans.T
        F = 248
        h = (-Y)/(-Z)*F + 224/2.0
        w = X/(-Z)*F + 224/2.0
        h = np.minimum(np.maximum(h, 0), 223)
        w = np.minimum(np.maximum(w, 0), 223)
        img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
        img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
        cv2.imwrite(img_path.replace('.png','_prj.png'), img)
