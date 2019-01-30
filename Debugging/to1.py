import cPickle as pickle


dataset_root = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/'
dataset_lib = '04530566_ffffe224db39febe288b05b36358465d_23.dat'
dataset_dir = dataset_root + dataset_lib
print(dataset_dir)
data = pickle.load(open(dataset_dir, 'rb'))
pass