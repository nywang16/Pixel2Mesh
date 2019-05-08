#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tensorflow as tf
import cPickle as pickle
from skimage import io,transform
from p2m.api import GCN
from p2m.utils import *

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('image', 'Data/examples/plane.png', 'Testing image.')
flags.DEFINE_float('learning_rate', 0., 'Initial learning rate.')
flags.DEFINE_integer('hidden', 256, 'Number of units in  hidden layer.')
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

def load_image(img_path):
	img = io.imread(img_path)
	if img.shape[2] == 4:
		img[np.where(img[:,:,3]==0)] = 255
	img = transform.resize(img, (224,224))
	img = img[:,:,:3].astype('float32')

	return img

# Load data, initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)

# Runing the demo
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

img_inp = load_image(FLAGS.image)
feed_dict.update({placeholders['img_inp']: img_inp})
feed_dict.update({placeholders['labels']: np.zeros([10,6])})

vert = sess.run(model.output3, feed_dict=feed_dict)
vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
face = np.loadtxt('Data/ellipsoid/face3.obj', dtype='|S32')
mesh = np.vstack((vert, face))
pred_path = FLAGS.image.replace('.png', '.obj')
np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

print 'Saved to', pred_path
