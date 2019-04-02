import numpy as np

def construct_feed_dict(pkl, placeholders):
	"""Construct feed dictionary."""
	coord = pkl[0]
	pool_idx = pkl[4]
	faces = pkl[5]
	# laplace = pkl[6]
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
