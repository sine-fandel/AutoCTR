# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

ML schema inference by input data

"""
import pandas as pd
import numpy as np
import pickle
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class ML_schema (object) :
	def __init__ (self, data) :
		self.df = data

	def onehotword (self, word) :
		word = re.sub(r"[^a-zA-Z0-9]", "", word)
		word = word.lower()
		aword = np.zeros ((1, 26))
		order = 1
		for c in word :
			if c.isalpha () :
				index = ord (c) - 97
				aword[0][index] = order
				order += 1
		
		return aword

	def inference (self) :
		"""Inference the ML schema
			1. Get the feature semantic of data features.
		
		"""
		with open ('/Users/apple/AutoCTR project/AutoCTR/autoctr/preprocessor/pkl/feature_type_model.pkl', 'rb') as file :
			model = pickle.load (file)
		
		types_list = ['categorical', 'numerical', 'time', 'others']
		schema = {}
		column_name = self.df.columns
		for column in column_name :
			schema[column] = types_list[int (model.predict (self.onehotword (column))[0])]
		
		return schema

# # data = pd.read_csv ('/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data', sep='\\s+')
# mlschema = ML_schema ('/Users/apple/AutoCTR project/dataset/criteo_2m.csv')
# print (mlschema.inference ())
