import tensorflow as tf
import cv2
from pixel2mesh.models import GCN
from pixel2mesh.fetcher import *

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('image', 'utils/examples/chair.png', 'Testing image.')
flags.DEFINE_float('learning_rate', 0., 'Initial learning rate.')
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

def load_image(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	if img.shape[2] == 4:
		img[np.where(img[:,:,3]==0)] = 255
	img = cv2.resize(img, (224,224))
	img = img.astype('float32')/255.0
	return img[:,:,:3]

# Load data, initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)

# Runing the demo
pkl = pickle.load(open('utils/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

img_inp = load_image(FLAGS.image)
feed_dict.update({placeholders['img_inp']: img_inp})
feed_dict.update({placeholders['labels']: np.zeros([10,6])})

vert = sess.run(model.output3, feed_dict=feed_dict)
vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
face = np.loadtxt('utils/ellipsoid/face3.obj', dtype='|S32')
mesh = np.vstack((vert, face))
pred_path = FLAGS.image.replace('.png', '.obj')
np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

print 'Saved to', pred_path
