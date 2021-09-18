# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

The extrance of AutoCTR package.

"""
from .preprocessor.profile import Profiling
from .preprocessor.cleaning import Impute
from .preprocessor.feature_column import SparseFeat, DenseFeat
from .models import *

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import os
import pandas as pd
import numpy as np

class AutoCTR :
	"""The whole process of recommender
	
	:param data_path: The path of dataset
	:param target: The training / testing target of all model
	:param sep: The seq of the dataset
	"""
	def __init__ (self, data_path, target, sep=",") :
		self.data = pd.read_csv (data_path, sep=sep)
		self.target = target
		self.model_list = [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, DCNMix, ONN, WDL]
		self.sep = sep
		self.input_list = []
		self.data_profile = None
		self.types_dict = {}
		self.tag = 0

	def profiling (self, test_size=0.2, outlier="z_score", correlation="pearson") :
		"""Get the data summary of the dataset.
		"""
		self.data_profile = Profiling (self.data, outlier=outlier, correlation=correlation).summary ()
		self.types_dict = self.data_profile.loc['types'].to_dict ()
		self.tag = 1

	def preprocessing (self, impute_method="knn", test_size=0.2) :
		"""Deal with the missing values and convert data for training
		"""
		try :
			if not self.tag :
				raise Exception ("[Error]: Do not get dataset yet...")

			types_dict = self.data_profile.loc['types'].to_dict ()
			count_missing = self.data_profile.loc['missing'].to_dict ()
			if_missing = 0
			for key, value in count_missing.items () :
				if value != 0 :
					if_missing = 1

			if if_missing :
				impute = Impute (self.data)
				if impute_method == 'knn' :
					self.data = impute.KnnImputation (n_neighbors=2)
				elif impute_method == 'simple' :
					self.data = impute.SimpleImputation ()
				elif impute_method == 'iterative' :
					self.data = impute.IterativeImputation ()
				elif impute_method == 'forest' :
					self.data = impute.RandomforestImputation ()
				elif impute_method == 'mf' :
					self.data = impute.MatrixFactorization ()

				print ("Finish imputation by ", impute_method)
			else :
				print ("No missing value. Nothing to do...")

			feature_names = self.data_profile.columns.values
			feature_names_temp = list (feature_names)
			feature_names = np.delete (feature_names, feature_names_temp.index (self.target))

			fixlen_feature_columns = []

			for key, value in types_dict.items () :
				if key != self.target :
					if value == "categorical" :
						lbe = LabelEncoder ()
						self.data[key] = lbe.fit_transform (self.data[key])
						fixlen_feature_columns.append (SparseFeat (key, self.data[key].nunique ()))
						
					elif value == "numeric" :
						fixlen_feature_columns.append (DenseFeat (key, 1, ))
			
			train, test = train_test_split (self.data, test_size=test_size, random_state=2021)
			self.input_list.append (train)
			self.input_list.append (test)
			self.input_list.append ({name: train[name] for name in feature_names})
			self.input_list.append ({name: test[name] for name in feature_names})
			self.input_list.append (fixlen_feature_columns)
			self.input_list.append (fixlen_feature_columns)

			print ("Have converted data to training format...")

		except Exception as e :
			print (e)
			

	def run (self, batch_size=32, epochs=100, verbose=2, save_path="./PKL/", earl_stop_patience=0) :
		"""Train and Test
		"""
		
		if not os.path.exists (save_path) :
			os.makedirs (save_path)

		train = self.input_list[0]
		test = self.input_list[1]
		train_model_input = self.input_list[2]
		test_model_input = self.input_list[3]
		linear_feature_columns = self.input_list[4]
		dnn_feature_columns = self.input_list[5]

		if train[self.target].nunique () > 2:
			metrics = 0			# MSE
		else :
			metrics = 1			# AUC

		use_cuda = True
		if use_cuda and torch.cuda.is_available () :
			print ('cuda ready...')
			device = 'cuda:0'
		else :
			print ('using cpu...')
			device = 'cpu'

		for Model in self.model_list :
			print("Train on {0} samples, validate on {1} samples".format (len(train), len(test)))
			if Model.__name__ != "PNN" :
				model = Model (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=device)
				if metrics == 1 :
					model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
				else :
					model.compile ("adam", "mse", metrics=["mse"], )
				model.fit (train_model_input, train[self.target].values, batch_size=batch_size, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience)
				pred_ans = model.predict (test_model_input, 256)

				if metrics == 1 :
					print ("Validation Accuracy: ", round (roc_auc_score (test[self.target].values, pred_ans), 4))
				else :
					print ("Validation MSE: ", round (mean_squared_error (test[self.target].values, pred_ans), 4))

				torch.save (model.state_dict (), save_path + Model.__name__ + "_epoach:" + str (epochs) + ".pkl") 

			
			else :
				model = Model (dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=device)
				if metrics == 1 :
					model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
				else :
					model.compile ("adam", "mse", metrics=["mse"], )
				model.fit (train_model_input, train[self.target].values, batch_size=batch_size, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience)
				pred_ans = model.predict (test_model_input, 256)

				if metrics == 1 :
					print ("Validation Accuracy: ", round (roc_auc_score (test[self.target].values, pred_ans), 4))
				else :
					print ("Validation MSE: ", round (mean_squared_error (test[self.target].values, pred_ans), 4))

				torch.save (model.state_dict (), save_path + Model.__name__ + "_epoach:" + str (epochs) + ".pkl") 
	