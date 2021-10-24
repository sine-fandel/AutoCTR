# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

AutoML for Recommender to find the best recommender pipeline

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

class Recommender (object) :
	"""To find the best recommender pipeline by AutoML
	
	:param data_path: The path of dataset
	:param target: The training / testing target of all model
	:param sep: The seq of the dataset
	"""
	def __init__ (self, data_path, target, sep=",", frac=1.0) :
			self.data = pd.read_csv (data_path, sep=sep, encoding='unicode_escape').sample (frac=frac)
			print (self.data)
			self.target = target
			self.model_list = [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, ONN, WDL]
			self.sep = sep
			self.input_list = []
			self.tag = 0
			self.parity_score = {}
			self.score = []
			self.task = 0
			self.metrics = 0
			self.data_schema = {}

	def get_pipeline (self, frac=0.1, impute_method="simple", batch_size=256, pre_train=1, pre_train_epoch=10, epochs=10) :
		"""Get the best recommender pipeline for specify dataset
		:param frac: The frac of pre-train dataset
		:param impute_method: The method for imputing missing value
		:param batch_size: Batch size of training
		:param pre_train: whether to pretrain
		:param pre_train_epoch / epochs: the epochs of pre train / train
		"""
		################################################################
		#	Data quality checking and data cleaning					   #
		################################################################
		self.quality_checking ()
		self.data_cleaning (impute_method=impute_method)

		################################################################
		#	Get the data schema 									   #
		################################################################
		mlschema = ML_schema (self.data)
		self.data_schema = mlschema.inference ()
		print ("Data Schema: ", self.data_schema)

		################################################################
		#	Pre-train the model to find the best combination		   #
		################################################################
		sample_data = self.data.sample (frac=frac)
		temp = self.data
		column_list = self.data.columns.values.tolist ()
		column_list.remove (self.target)
		if self.data[self.target].nunique () > 2:
			self.metrics = 0			# MSE
			self.task = "regression"
			best_score = 9999
 
		else :
			self.metrics = 1			# AUC
			self.task = "binary"
			best_score = 0

		if pre_train :
			print ("###Pre-training the Model to find the best feature comibinatin")
			print ("###CONFIGURE")
			print ('###pre-train-epochs: ', pre_train_epoch)
			print ('###training-epochs: ', epochs)
			print ('###batch_size: ', batch_size)
			vis_list = []
			score_dict = {}
			print ("ORIGINAL...")
			self.data = sample_data
			self.feature_engineering ()
			self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch)
			self.input_list = []
			for c1 in column_list :
				vis_list.append (c1)
				for c2 in column_list :
					if self.data_schema[c1] != 'numerical' and self.data_schema[c2] != 'numerical' :
						if c1 == c2 :
							print ("Combination: ", c1, " ...")
							self.data = sample_data
							self.feature_engineering (col_list=[c1])
							self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch)
							self.input_list = []
							score_dict[self.score[-1]] = [c1]

						elif c1 != c2 and c2 not in vis_list :
							print ("Combination: ", c1, " ", c2, " ...")
							self.data = sample_data
							self.feature_engineering (col_list=[c1, c2])
							self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch)
							self.input_list = []
							score_dict[self.score[-1]] = [c1, c2]
			
			best_com = []
			for key, value in score_dict.items () :
				if self.metrics == 0 :
					if best_score > key :
						best_score = key
						best_com = value

				else :
					if best_score < key :
						best_score = key
						best_com = value

			print ("The best combination of feature is: ", best_com, "	With the score: ", best_score)
		################################################################
		#	Train the model by the best combination feature			   #
		################################################################
		self.data = temp
		self.input_list = []
		if not pre_train :
			best_com = None
		self.feature_engineering (col_list=best_com)
		self.run (batch_size=batch_size, epochs=epochs)

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
		print ("Quality Checking ......")
		self.tag = 1
		qod = QoD (self.data, target=self.target)
		qod._get_outlier ()
		qod._get_completeness ()
		qod._get_duplicated ()
		self.parity_score = qod._get_class_parity ()
		qod._get_correlations ()
		print ("...done!")

	def data_cleaning (self, impute_method="simple") :
		"""clean the data
		"""
		print ("Quality Cleaning ......")
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

		print ("...done!")

	def feature_engineering (self, test_size=0.2, col_list=None) :
		"""Deal with the missing values and convert data for training
		"""
		try :
			if not self.tag :
				raise Exception ("[Error]: Do not get dataset yet...")

			feature_names = self.data.columns.values
			feature_names_temp = list (feature_names)
			feature_names = np.delete (feature_names, feature_names_temp.index (self.target))
			fixlen_feature_columns = []


			# feature combination by target encoding
			if col_list != None :
				col_list1 = col_list.copy ()
				col_list1.append (self.target)
				te = self.data[col_list1].groupby (col_list).mean ()
				te = te.reset_index()
				new_col_name = 'TE'
				for c in col_list :
					new_col_name = new_col_name + '_' + str (c)
				col_list2 = col_list.copy ()
				col_list2.append (new_col_name) 
				te.columns = col_list2
				self.data = self.data.merge (te, how='left', on=col_list)
				self.data_schema[new_col_name] = 'numerical'
				feature_names = np.append (feature_names, new_col_name)
			
			for key, value in self.data_schema.items () :
				if key != self.target :
					if value != 'numerical' :
						lbe = LabelEncoder ()
						self.data[key] = lbe.fit_transform (self.data[key])
						fixlen_feature_columns.append (SparseFeat (key, self.data[key].nunique ()))
					
					else :
						fixlen_feature_columns.append (DenseFeat (key, 1, ))
			
			if col_list != None :
				self.data_schema.pop (new_col_name)

			train, test = train_test_split (self.data, test_size=test_size, random_state=2021)
			self.input_list.append (train)
			self.input_list.append (test)
			self.input_list.append ({name: train[name] for name in feature_names})
			self.input_list.append ({name: test[name] for name in feature_names})
			self.input_list.append (fixlen_feature_columns)
			self.input_list.append (fixlen_feature_columns)
		
			return self.input_list

		except Exception as e :
			print (e)

	def run (self, batch_size=256, epochs=100, verbose=2, save_path="./PKL/", earl_stop_patience=0, if_tune=0, Model=DeepFM) :
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

		use_cuda = True
		if use_cuda and torch.cuda.is_available () :
			if if_tune == 0 :
				print ('cuda ready...')
				print ("Train on {0} samples, validate on {1} samples".format (len(train), len(test)))
			device = 'cuda:0'
		else :
			if if_tune == 0 :
				print ('using cpu...')
				print ("Train on {0} samples, validate on {1} samples".format (len(train), len(test)))
			device = 'cpu'


		if Model.__name__ != "PNN" :
			model = Model (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task=self.task, l2_reg_embedding=1e-5, device=device)
			if self.metrics == 1 :
				model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"], )
			else :
				model.compile ("adam", "mse", metrics=["mse"], )

			model.fit (train_model_input, train[self.target].values, batch_size=batch_size, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience, if_tune=if_tune)
			pred_ans = model.predict (test_model_input, 256)

			if self.metrics == 1 :
				print ("Validation Accuracy: ", round (roc_auc_score (test[self.target].values, pred_ans), 4))
				self.score.append (round (roc_auc_score (test[self.target].values, pred_ans), 4))
			else :
				print ("Validation MSE: ", round (mean_squared_error (test[self.target].values, pred_ans), 4))
				self.score.append (round (mean_squared_error (test[self.target].values, pred_ans), 4))
		
		else :
			model = Model (dnn_feature_columns=dnn_feature_columns, task=self.task, l2_reg_embedding=1e-5, device=device)
			if self.metrics == 1 :
				model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
			else :
				model.compile ("adam", "mse", metrics=["mse"], )
			model.fit (train_model_input, train[self.target].values, batch_size=batch_size, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience, if_tune=if_tune)
			pred_ans = model.predict (test_model_input, 256)

			if self.metrics == 1 :
				print ("Validation Accuracy: ", round (roc_auc_score (test[self.target].values, pred_ans), 4))
				self.score.append (round (roc_auc_score (test[self.target].values, pred_ans), 4))
			else :
				print ("Validation MSE: ", round (mean_squared_error (test[self.target].values, pred_ans), 4))
				self.score.append (round (mean_squared_error (test[self.target].values, pred_ans), 4))


