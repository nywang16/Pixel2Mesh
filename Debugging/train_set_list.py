import cPickle as pickle
import tensorflow as tf

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


# dataset_root = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/'
# dataset_lib = '04530566_ffffe224db39febe288b05b36358465d_23.dat'
# dataset_dir = dataset_root + dataset_lib
dataset_dir = '/home/eralien/storage/Pixel2Mesh/pixel2mesh/utils/ellipsoid/info_ellipsoid.dat'
print(dataset_dir)
data = pickle.load(open(dataset_dir, 'rb'))
coo     = data[0]
data1   = data[1]
data2   = data[2]
data3   = data[3]
pool_idx= data[4]
faces   = data[5]
data6   = data[6]
lapn    = data[7]

feed_dict = construct_feed_dict(data, placeholders)
pass