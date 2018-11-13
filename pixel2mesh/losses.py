import tensorflow as tf
from pixel2mesh.cd_dist import *

def laplace_coord(pred, placeholders, block_id):
	vertex = tf.concat([pred, tf.zeros([1,3])], 0)
	indices = placeholders['lape_idx'][block_id-1][:, :8]
	weights = tf.cast(placeholders['lape_idx'][block_id-1][:,-1], tf.float32)

	weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1,1]), [1,3])
	laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
	laplace = tf.subtract(pred, tf.multiply(laplace, weights))
	return laplace

def laplace_loss(pred1, pred2, placeholders, block_id):
	# laplace term
	lap1 = laplace_coord(pred1, placeholders, block_id)
	lap2 = laplace_coord(pred2, placeholders, block_id)
	laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1,lap2)), 1)) * 1500

	move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
	move_loss = tf.cond(tf.equal(block_id,1), lambda:0., lambda:move_loss)
	return laplace_loss + move_loss
	
def unit(tensor):
	return tf.nn.l2_normalize(tensor, dim=1)

def mesh_loss(pred, placeholders, block_id):
	gt_pt = placeholders['labels'][:, :3] # gt points
	gt_nm = placeholders['labels'][:, 3:] # gt normals

	# edge in graph
	nod1 = tf.gather(pred, placeholders['edges'][block_id-1][:,0])
	nod2 = tf.gather(pred, placeholders['edges'][block_id-1][:,1])
	edge = tf.subtract(nod1, nod2)

	# edge length loss
	edge_length = tf.reduce_sum(tf.square(edge), 1)
	edge_loss = tf.reduce_mean(edge_length) * 300

	# chamer distance
	dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
	point_loss = (tf.reduce_mean(dist1) + 0.55*tf.reduce_mean(dist2)) * 3000

	# normal cosine loss
	normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
	normal = tf.gather(normal, placeholders['edges'][block_id-1][:,0])
	cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
	# cosine = tf.where(tf.greater(cosine,0.866), tf.zeros_like(cosine), cosine) # truncated
	normal_loss = tf.reduce_mean(cosine) * 0.5

	total_loss = point_loss + edge_loss + normal_loss
	return total_loss
