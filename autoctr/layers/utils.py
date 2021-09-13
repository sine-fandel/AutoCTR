# -*- coding:utf-8 -*-
"""

Author:
	Zhengxin Fang, 358746595@qq.com

Several "tools" for model

"""
import numpy as np
import torch

def concat_fun (inputs, axis=-1) :
	if len (inputs) == 1 :
		return inputs [0]
	else :
		return torch.cat (inputs, dim=axis)

def slice_arrays (arrays, start=None, stop=None) :
	"""Slice array

	Arguments: 
		arrays: an array that need to be sliced
		start: an integer index
		stop: stop index

	Returns:
		A slice of the array

	"""
	if arrays is None :
		return [None]

	if isinstance (arrays, np.ndarray) :
		arrays = [arrays]

	if isinstance (arrays, list) and stop is not None :
		raise ValueError ('The stop argument has to be None if the value of start is a list')

	elif isinstance (arrays, list) :
		if hasattr (start, '__len__') :
			if hasattr (start, 'shape') :
				start = start.tolist
			
			return [None if x is None else x[start] for x in arrays]

		else :
			if len (arrays) == 1 :
				return arrays[0][start : stop]

			return [None if x is None else x[start : stop] for x in arrays]

	else :
		if hasattr (start, '__len__') :
			if hasattr (start, 'shape') :
				start = start.tolist ()
			return arrays[start]
		elif hasattr (start, '__getitem__') :
			return arrays[start : stop]
		else :
			return [None]

