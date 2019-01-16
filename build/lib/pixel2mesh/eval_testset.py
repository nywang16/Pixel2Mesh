import os, sys
import tensorflow as tf
from pixel2mesh.models import GCN
from pixel2mesh.fetcher import *
from pixel2mesh.cd_dist import nn_distance
sys.path.append('external')
from tf_approxmatch import approx_match, match_cost

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', 'utils/test_list.txt', 'Data list path.')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('hidden', 192, 'Number of units in  hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)), # initial 3D coordinates
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)), # input image to network
    'labels': tf.placeholder(tf.float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the first block
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the second block
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the third block
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)], # helper for face loss (not used)
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], # helper for normal loss
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], # helper for laplacian regularization
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
}
model = GCN(placeholders, logging=True)

# Construct feed dictionary
def construct_feed_dict(pkl, placeholders):
	coord = pkl[0]
	pool_idx = pkl[4]
	faces = pkl[5]
	lape_idx = pkl[7]
	edges = []
	for i in range(1,4):
		adj = pkl[i][1]
		edges.append(adj[0])

	feed_dict = dict()
	feed_dict.update({placeholders['features']: coord})
	feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
	feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
	feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
	feed_dict.update({placeholders['support1'][i]: pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({placeholders['support2'][i]: pkl[2][i] for i in range(len(pkl[2]))})
	feed_dict.update({placeholders['support3'][i]: pkl[3][i] for i in range(len(pkl[3]))})
	return feed_dict

def f_score(label, predict, dist_label, dist_pred, threshold):
	num_label = label.shape[0]
	num_predict = predict.shape[0]

	f_scores = []
	for i in range(len(threshold)):
		num = len(np.where(dist_label <= threshold[i])[0])
		recall = 100.0 * num / num_label
		num = len(np.where(dist_pred <= threshold[i])[0])
		precision = 100.0 * num / num_predict

		f_scores.append((2*precision*recall)/(precision+recall+1e-8))
	return np.array(f_scores)

# Load data
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) ####
data.start()
train_number = data.number

# Initialize session
# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
# chamfer distance
dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)
# earth mover distance, notice that emd_dist return the sum of all distance
match = approx_match(xyz1, xyz2)
emd_dist = match_cost(xyz1, xyz2, match)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)

# Construct feed dictionary
pkl = pickle.load(open('utils/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

###
class_name = {'02828884':'bench','03001627':'chair','03636649':'lamp','03691459':'speaker','04090263':'firearm','04379243':'table','04530566':'watercraft','02691156':'plane','02933112':'cabinet','02958343':'car','03211117':'monitor','04256520':'couch','04401088':'cellphone'}
model_number = {i:0 for i in class_name}
sum_f = {i:0 for i in class_name}
sum_cd = {i:0 for i in class_name}
sum_emd = {i:0 for i in class_name}

for iters in range(train_number):
	# Fetch training data
	img_inp, label, model_id = data.fetch()
	feed_dict.update({placeholders['img_inp']: img_inp})
	feed_dict.update({placeholders['labels']: label})
	# Training step
	predict = sess.run(model.output3, feed_dict=feed_dict)

	label = label[:, :3]
	d1,i1,d2,i2,emd = sess.run([dist1,idx1,dist2,idx2, emd_dist], feed_dict={xyz1:label,xyz2:predict})
	cd = np.mean(d1) + np.mean(d2)

	class_id = model_id.split('_')[0]
	model_number[class_id] += 1.0

	sum_f[class_id] += f_score(label,predict,d1,d2,[0.0001, 0.0002])
	sum_cd[class_id] += cd # cd is the mean of all distance
	sum_emd[class_id] += emd[0] # emd is the sum of all distance
	print 'processed number', iters

log = open('record_evaluation.txt', 'a')
for item in model_number:
	number = model_number[item] + 1e-8
	f = sum_f[item] / number
	cd = (sum_cd[item] / number) * 1000 #cd is the mean of all distance, cd is L2
	emd = (sum_emd[item] / number) * 0.01 #emd is the sum of all distance, emd is L1
	print class_name[item], int(number), f, cd, emd
	print >> log, class_name[item], int(number), f, cd, emd
log.close()
sess.close()
data.shutdown()
print 'Testing Finished!'
