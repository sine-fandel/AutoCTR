# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

One step of preprocessing the data

"""


from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

# from .layers.sequence import SequencePoolingLayer
# from .layers.utils import concat_fun

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
							['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
							'group_name'])):
	__slots__ = ()

	def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
				group_name=DEFAULT_GROUP_NAME):
		if embedding_name is None:
			embedding_name = name
		if embedding_dim == "auto":
			embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
		if use_hash:
			print(
				"Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
		return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
											embedding_name, group_name)

	def __hash__(self):
		return self.name.__hash__()

class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
								['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
	__slots__ = ()

	def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
		return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

	@property
	def name(self):
		return self.sparsefeat.name

	@property
	def vocabulary_size(self):
		return self.sparsefeat.vocabulary_size

	@property
	def embedding_dim(self):
		return self.sparsefeat.embedding_dim

	@property
	def use_hash(self):
		return self.sparsefeat.use_hash

	@property
	def dtype(self):
		return self.sparsefeat.dtype

	@property
	def embedding_name(self):
		return self.sparsefeat.embedding_name

	@property
	def group_name(self):
		return self.sparsefeat.group_name

	def __hash__(self):
		return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
	__slots__ = ()

	def __new__(cls, name, dimension=1, dtype="float32"):
		return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

	def __hash__(self):
		return self.name.__hash__()