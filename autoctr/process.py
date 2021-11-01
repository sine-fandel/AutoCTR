# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

The extrance of AutoCTR package.

"""
from .preprocessor.profile import Profiling
from .preprocessor.cleaning import Impute
from .preprocessor.feature_column import SparseFeat, DenseFeat
from .preprocessor.quality import QoD
from .preprocessor.ml_schema import ML_schema

from .models import *
from .optimizer import RandomSearch, BayesianOptimization

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from alive_progress import alive_bar
from imblearn.over_sampling import SMOTE

import torch
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

class AutoCTR :
	"""The whole process of recommender
	
	:param data_path: The path of dataset
	:param target: The training / testing target of all model
	:param sep: The seq of the dataset
	"""
	def __init__ (self, data_path, target, sep=",") :
		self.data = pd.read_csv (data_path, sep=sep)
		self.target = target
		self.model_list = ['DeepFM', 'xDeepFM', 'AFN', 'NFM', 'IFM', 'DIFM', 'AutoInt', 'PNN', 'DCN', 'ONN', 'WDL']
		self.sep = sep
		self.input_list = []
		self.data_profile = None
		# self.types_dict = self.get_types ()
		self.tag = 0
		self.parity_score = {}

	def get_types (self) :
		"""Get the column types
		"""
		types_dict = {}
		for c in self.data.columns :
			if c != self.target :
				types_dict[c] = self.data[c].dtypes

		return types_dict

	def quality_checking (self) :
		"""Checking the data quality
		"""
		self.tag = 1
		qod = QoD (self.data, target=self.target)
		qod._get_outlier ()
		qod._get_completeness ()
		qod._get_duplicated ()
		self.parity_score = qod._get_class_parity ()
		qod._get_correlations ()

	def data_cleaning (self, impute_method="knn") :
		"""clean the data
		"""
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

		print ("Finished imputation by ", impute_method)

		if self.parity_score['Class Parity'] <= 0.5 :
			sm = SMOTE (random_state=666)
			dataset = self.data
			dataset[self.target] = dataset[self.target].astype (int)
			y = dataset[self.target].values
			X = dataset.drop (labels=self.target, axis=1).values
			X_res, y_res = sm.fit_resample(X, y)
			label_name = self.target
			column_name = self.types_dict
			column_name.pop (self.target)
			column_list = column_name.keys ()

			ds = pd.DataFrame (X_res, columns=column_list)
			ds.insert (ds.shape[1], label_name, y_res)

			self.data = ds

			print ("Finished SMOTE to balance the imbalance data")
		else :
			print ("Data is balanced")

	# def profiling (self, test_size=0.2, outlier="z_score", correlation="pearson") :
	# 	"""Get the data summary of the dataset.
	# 	"""
	# 	self.data_profile = Profiling (self.data, outlier=outlier, correlation=correlation, target=self.target).summary ()
	# 	self.types_dict = self.data_profile.loc['types'].to_dict ()
	# 	self.tag = 1

	def preprocessing (self, impute_method="knn", test_size=0.2) :
		"""Deal with the missing values and convert data for training
		"""
		try :
			if not self.tag :
				raise Exception ("[Error]: Do not get dataset yet...")

			# get data schema
			mlschema = ML_schema (self.data)
			data_schema = mlschema.inference ()

			feature_names = self.data.columns.values
			feature_names_temp = list (feature_names)
			feature_names = np.delete (feature_names, feature_names_temp.index (self.target))
			# feature_names = np.append (feature_names, 'TE_user')
			feature_names = np.append (feature_names, 'TE_item_time')
			print (feature_names)
			fixlen_feature_columns = []

			cat = 'user'
			te = self.data[[cat, 'ratings']].groupby (cat).mean ()
			te = te.reset_index ()
			te.columns = [cat, 'TE_' + cat]
			self.data = self.data.merge (te, how='left', on=cat)
			data_schema['TE_user'] = 'numerical'

			# cat1 = 'item'
			# te1 = self.data[[cat1, 'ratings']].groupby (cat1).mean ()
			# te1 = te1.reset_index ()
			# te1.columns = [cat1, 'TE_' + cat1]
			# self.data = self.data.merge (te1, how='left', on=cat1)
			# data_schema['TE_item'] = 'numerical'

			# te = self.data[['user', 'time', 'ratings']].groupby (['user', 'time']).mean ()
			# te = te.reset_index()
			# te.columns = ['user', 'time', 'TE_user_time']
			# self.data = self.data.merge (te, how='left', left_on=['user', 'time'], right_on=['user', 'time'])
			# data_schema['TE_user_time'] = 'numerical'

			# te = self.data[['item', 'time', 'ratings']].groupby (['item', 'time']).mean ()
			# print (te)
			# te = te.reset_index()
			# te.columns = ['item', 'time', 'TE_item_time']
			# self.data = self.data.merge (te, how='left', left_on=['item', 'time'], right_on=['item', 'time'])
			# data_schema['TE_item_time'] = 'numerical'

			for key, value in data_schema.items () :
				if key != self.target :
					if value != 'numerical' :
						lbe = LabelEncoder ()
						self.data[key] = lbe.fit_transform (self.data[key])
						fixlen_feature_columns.append (SparseFeat (key, self.data[key].nunique ()))
					
					else :
						fixlen_feature_columns.append (DenseFeat (key, 1, ))
			

			train, test = train_test_split (self.data, test_size=test_size, random_state=2021)
			self.input_list.append (train)
			self.input_list.append (test)
			self.input_list.append ({name: train[name] for name in feature_names})
			self.input_list.append ({name: test[name] for name in feature_names})
			self.input_list.append (fixlen_feature_columns)
			self.input_list.append (fixlen_feature_columns)

			print ("Have converted data to training format...")

			return self.input_list

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
			task = "regression"
			print ("TASK: ", task)
		else :
			metrics = 1			# AUC
			task = "binary"
			print ("TASK: ", task)

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
				model = Model (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task=task, l2_reg_embedding=1e-5, device=device)
				if metrics == 1 :
					model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"], )
				else :
					model.compile ("adam", "mse", metrics=["mse"], )
				print (train)
				model.fit (train_model_input, train[self.target].values, batch_size=batch_size, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience)
				pred_ans = model.predict (test_model_input, 256)

				if metrics == 1 :
					print ("Validation Accuracy: ", round (roc_auc_score (test[self.target].values, pred_ans), 4))
				else :
					print ("Validation MSE: ", round (mean_squared_error (test[self.target].values, pred_ans), 4))

				torch.save (model.state_dict (), save_path + Model.__name__ + "_epoach:" + str (epochs) + ".pkl") 

			
			else :
				model = Model (dnn_feature_columns=dnn_feature_columns, task=task, l2_reg_embedding=1e-5, device=device)
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
	
	def search (self, model_list=["DeepFM"], tuner="bayesian", save_path="./PKL/", hp_path="./HP/", max_evals=10, epochs=100) :
		"""Search the best hyperparameters for model
		"""
		save_path = save_path + tuner + "/"
		hp_path = hp_path + tuner + "/"

		train = self.input_list[0]
		if train[self.target].nunique () > 2:
			metrics = 0			# MSE
			task = "regression"
			print ("TASK: ", task)
		else :
			metrics = 1			# AUC
			task = "binary"
			print ("TASK: ", task)

		use_cuda = True
		if use_cuda and torch.cuda.is_available () :
			print ('cuda ready...')
			device = 'cuda:0'
		else :
			print ('using cpu...')
			device = 'cpu'
		
		if not os.path.exists (save_path) :
			os.makedirs (save_path)
		if not os.path.exists (hp_path) :
			os.makedirs (hp_path)
		for model_tune in model_list :
			if tuner == "random" :
				print ("Tuning the %s model by %s..." % (model_tune, tuner))
				random_search = RandomSearch (model_name=model_tune, linear_feature_columns=self.input_list[4],
											dnn_feature_columns=self.input_list[5], task="binary", 
											device="cpu", max_evals=max_evals, save_path=save_path)

				best_param = random_search.search (self.input_list[2], self.input_list[0][self.target].values, self.input_list[3], 
													self.input_list[1][self.target].values, epochs=epochs)
										
				with open (hp_path + model_tune + ".json", "w") as f :
					f.write (json.dumps (best_param, ensure_ascii=False, indent=4, separators=(',', ':')))


			if tuner == "bayesian" :
				print ("Tuning the %s model by %s..." % (model_tune, tuner))
				bayesian_search = BayesianOptimization (inputs=self.input_list, random_state=None, verbose=2, bounds_transformer=None, device=device,
														model_name=model_tune, epochs=epochs, max_evals=max_evals, target=self.target, metrics=metrics,
														task=task)	
				best_param = bayesian_search.maximize ()

				with open (hp_path + model_tune + ".json", "w") as f :
					f.write (json.dumps (best_param, ensure_ascii=False, indent=4, separators=(',', ':')))


	def _get_model (self, models=[]) :
		"""Get models

		Models list: [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, DCNMix, ONN, WDL]
		"""
		model_list = []
		for model in models :
			if model == "DeepFM" :
				model_list.append (DeepFM)
			elif model == "xDeepFM" :
				model_list.append (xDeepFM)
			elif model == "AFN" :
				model_list.append (AFN)
			elif model == "NFM" :
				model_list.append (NFM)
			elif model == "IFM" :
				model_list.append (IFM)
			elif model == "DIFM" :
				model_list.append (DIFM)
			elif model == "AutoInt" :
				model_list.append (AutoInt)
			elif model == "PNN" :
				model_list.append (PNN)
			elif model == "DCN" :
				model_list.append (DCN)
			elif model == "DCNMix" :
				model_list.append (DCNMix)
			elif model == "ONN" :
				model_list.append (ONN)
			elif model == "WDL" :
				model_list.append (WDL)

		return model_list