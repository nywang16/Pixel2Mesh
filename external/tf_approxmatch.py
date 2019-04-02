import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

approxmatch_module = tf.load_op_library(osp.join(base_dir, 'tf_approxmatch_so.so'))


def approx_match(xyz1,xyz2):
	'''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
returns:
	match : batch_size * #query_points * #dataset_points
	'''
	xyz1 = tf.expand_dims(xyz1, 0)
	xyz2 = tf.expand_dims(xyz2, 0)
	return approxmatch_module.approx_match(xyz1,xyz2)
ops.NoGradient('ApproxMatch')
#@tf.RegisterShape('ApproxMatch')
@ops.RegisterShape('ApproxMatch')
def _approx_match_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[1]])]

def match_cost(xyz1,xyz2,match):
	'''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
	match : batch_size * #query_points * #dataset_points
returns:
	cost : batch_size
	'''
	xyz1 = tf.expand_dims(xyz1, 0)
	xyz2 = tf.expand_dims(xyz2, 0)
	return approxmatch_module.match_cost(xyz1,xyz2,match)
#@tf.RegisterShape('MatchCost')
@ops.RegisterShape('MatchCost')
def _match_cost_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	shape3=op.inputs[2].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0]])]
@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op,grad_cost):
	xyz1=op.inputs[0]
	xyz2=op.inputs[1]
	match=op.inputs[2]
	grad_1,grad_2=approxmatch_module.match_cost_grad(xyz1,xyz2,match)
	return [grad_1*tf.expand_dims(tf.expand_dims(grad_cost,1),2),grad_2*tf.expand_dims(tf.expand_dims(grad_cost,1),2),None]

if __name__=='__main__':
	alpha=0.5
	beta=2.0
	import bestmatch
	import numpy as np
	import math
	import random
	import cv2

	import tf_nndistance

	npoint=100

	with tf.device('/gpu:2'):
		pt_in=tf.placeholder(tf.float32,shape=(1,npoint*4,3))
		mypoints=tf.Variable(np.random.randn(1,npoint,3).astype('float32'))
		match=approx_match(pt_in,mypoints)
		loss=tf.reduce_sum(match_cost(pt_in,mypoints,match))
		#match=approx_match(mypoints,pt_in)
		#loss=tf.reduce_sum(match_cost(mypoints,pt_in,match))
		#distf,_,distb,_=tf_nndistance.nn_distance(pt_in,mypoints)
		#loss=tf.reduce_sum((distf+1e-9)**0.5)*0.5+tf.reduce_sum((distb+1e-9)**0.5)*0.5
		#loss=tf.reduce_max((distf+1e-9)**0.5)*0.5*npoint+tf.reduce_max((distb+1e-9)**0.5)*0.5*npoint

		optimizer=tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
	with tf.Session('') as sess:
		sess.run(tf.initialize_all_variables())
		while True:
			meanloss=0
			meantrueloss=0
			for i in xrange(1001):
				#phi=np.random.rand(4*npoint)*math.pi*2
				#tpoints=(np.hstack([np.cos(phi)[:,None],np.sin(phi)[:,None],(phi*0)[:,None]])*random.random())[None,:,:]
				#tpoints=((np.random.rand(400)-0.5)[:,None]*[0,2,0]+[(random.random()-0.5)*2,0,0]).astype('float32')[None,:,:]
				tpoints=np.hstack([np.linspace(-1,1,400)[:,None],(random.random()*2*np.linspace(1,0,400)**2)[:,None],np.zeros((400,1))])[None,:,:]
				trainloss,_=sess.run([loss,optimizer],feed_dict={pt_in:tpoints.astype('float32')})
			trainloss,trainmatch=sess.run([loss,match],feed_dict={pt_in:tpoints.astype('float32')})
			#trainmatch=trainmatch.transpose((0,2,1))
			show=np.zeros((400,400,3),dtype='uint8')^255
			trainmypoints=sess.run(mypoints)
			for i in xrange(len(tpoints[0])):
				u=np.random.choice(range(len(trainmypoints[0])),p=trainmatch[0].T[i])
				cv2.line(show,
					(int(tpoints[0][i,1]*100+200),int(tpoints[0][i,0]*100+200)),
					(int(trainmypoints[0][u,1]*100+200),int(trainmypoints[0][u,0]*100+200)),
					cv2.cv.CV_RGB(0,255,0))
			for x,y,z in tpoints[0]:
				cv2.circle(show,(int(y*100+200),int(x*100+200)),2,cv2.cv.CV_RGB(255,0,0))
			for x,y,z in trainmypoints[0]:
				cv2.circle(show,(int(y*100+200),int(x*100+200)),3,cv2.cv.CV_RGB(0,0,255))
			cost=((tpoints[0][:,None,:]-np.repeat(trainmypoints[0][None,:,:],4,axis=1))**2).sum(axis=2)**0.5
			#trueloss=bestmatch.bestmatch(cost)[0]
			print trainloss#,trueloss
			cv2.imshow('show',show)
			cmd=cv2.waitKey(10)%256
			if cmd==ord('q'):
				break
