# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

ML schema inference by input data

"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class ML_schema (object) :
	def __init__ (self, df) :
		self.df = data

	def data_info (self) :
		"""Get the data info for preparation of ML schema
		"""
		df_col = self.df.columns
		df_col
		
		df_des = self.df.describe ()
		df_des

		cate_list = [c for c in df_col if c not in df_des.columns]
		for c in cate_list :
			df_des.insert (df_des.shape[1], c, 0)

		df_des = df_des.T
	#     df_des.insert (df_des.shape[1], 'column_name', df_col)

		nan_column = self.df.isnull ().sum (axis=0)
		df_des.insert (df_des.shape[1], 'num_of_nan', nan_column.values)
		df_des.insert (df_des.shape[1], 'Attribute_name', df_col)
		df_des = df_des.drop (['25%', '50%', '75%'], axis=1)
		
		return df_des

	def inference (self) :
		"""Inference the ML schema
		"""
		df_info = self.data_info ()
		feature_types_data = pd.read_csv ('/Users/apple/AutoCTR project/AutoCTR/feature type inference/test/feature_types.csv', sep=',')
		len_data_info = len (df_info)
		len_feature_types_data = len (feature_types_data)

		train_data = feature_types_data.drop (['y_act'], axis=1)
		x_train, x_test,  y_train, y_test = train_test_split(train_data, feature_types_data['y_act'], test_size = 0.2, random_state = 7)

		x_train = pd.concat ([df_info, x_train], axis=0)
		label = preprocessing.LabelEncoder ()
		for i in x_train.columns :
			if x_train[i].dtype == 'object' :
				x_train[i] = label.fit_transform (x_train[i])
				
		x_train = x_train[len_data_info : ]
		df_info = x_train[0 : len_data_info]

		rfc = RandomForestClassifier(n_estimators= 60,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10)
		rfc = rfc.fit (x_train, y_train)

		print (rfc.predict (df_info))


data = pd.read_csv ('/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data', sep='\\s+')
mlschema = ML_schema (data)
mlschema.inference ()
