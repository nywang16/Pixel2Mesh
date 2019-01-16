import numpy as np
import cPickle as pickle
import threading
import Queue
#np.random.seed(123)

class DataFetcher(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.pkl_list = []
		with open(file_list, 'r') as f:
			while(True):
				line = f.readline().strip()
				if not line:
					break
				self.pkl_list.append(line)
		self.index = 0
		self.number = len(self.pkl_list)
		np.random.shuffle(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		pkl = pickle.load(open(pkl_path, 'rb'))
		img = pkl[0].astype('float32')/255.0
		label = pkl[1]

		return img, label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

if __name__ == '__main__':
	file_list = sys.argv[1]
	data = DataFetcher(file_list)
	data.start()

	image,point,_ = data.fetch()
	print image.shape
	print point.shape
	print normal.shape
	data.stopped = True
