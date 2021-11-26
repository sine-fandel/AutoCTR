# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

AutoML for Recommender to find the best recommender pipeline

"""
from .optimizer import geneticsearch
from .optimizer.core.param import CategoricalParam, ContinuousParam
from .preprocessor.cleaning import Impute
from .preprocessor.feature_column import SparseFeat, DenseFeat
from .preprocessor.quality import QoD
from .preprocessor.ml_schema import ML_schema

from .models import *
from .optimizer import RandomSearch, BayesianOptimization, GeneticHyperopt

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
import time

from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

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
			# self.model_list = [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, ONN, WDL]
			self.model_list = [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, DCN, ONN, WDL]
			self.sep = sep
			self.input_list = []
			self.tag = 0
			self.parity_score = {}
			self.score = []
			self.task = 0
			self.metrics = 0
			self.data_schema = {}
			self.quality_list = []
			self.best_com = []
			self.pre_train = 0
			self.maximize = False
	
	def generate_pdf (self, report_path) :
		"""Generate the pdf report of data
		"""
		doc = SimpleDocTemplate (report_path)
		styles = getSampleStyleSheet ()
		title_style = styles['Title']
		heading2_style = styles['Heading2']
		code_style = styles['Code']
		def_style = styles['Definition']
		body_stype = styles['BodyText']

		story = []
		story.append (Paragraph ("Data Profile", title_style))

		# show the first five row of dataset
		story.append (Paragraph ("The shape of dataset: " + str (self.data.shape), body_stype))

		# data schema part 
		story.append (Paragraph ("1 DATA SCHEMA", heading2_style))
		story.append (Spacer (1, .07 * inch))
		story.append (Paragraph ("Note: Data schema is shown as the following table to present what data belongs to what types.", def_style))
		story.append (Spacer (1, .2 * inch))
		data_schema = [['COLUMN NAME', 'FEATURE TYPE']]
		for key, value in self.data_schema.items () :
			data_schema.append ([key, value])

		t = Table (data_schema, splitByRow=1)
		t.setStyle (TableStyle (
			[('BOX', (0, 0), (-1, -1), 1, colors.black),
			('BACKGROUND', (0, 0), (1, 0), colors.lavender),
			("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
			('ALIGN', (0, 0), (-1, -1), 'CENTER')]
			))
		story.append (t)

		# data quality
		story.append (Spacer (1, .2 * inch))
		story.append (Paragraph ("2 DATA QUALITY", heading2_style))
		story.append (Spacer (1, .07 * inch))
		story.append (Paragraph ("Note: Data quality is shown as the following table, which includes the quality metric, the description of this type and the quality score", def_style))
		story.append (Spacer (1, .2 * inch))
		data_quality = [['QUALITY METRIC', 'DESCRIPTION', 'SCORE']]
		for q in self.quality_list :
			key, value = list (q.keys ())[0], list (q.values ())[0]
			description = ''
			if key == 'Outlier Detection' :
				description = Paragraph ('The outlier in range of [0, 1], wish higher score = 1 - #outlier / #data')
			elif key == 'Data Completeness' :
				description = Paragraph ('The accounts of missing value score = 1 - #missing / #data')
			elif key == 'Data Duplicated' :
				description = Paragraph ('The accounts of redundant data score = 1 - #redundant / # data')
			elif key == 'Class Parity' :
				description = Paragraph ('Get class parity, includes:    class imbalance: Shannon Entropy    class purity: parity score = (class imbalance + class purity) / 2')
			elif key == 'Correlation Dection' :
				description = Paragraph ('Get correlation between the data by pearson method')
			
			data_quality.append ([key, description, value])

		t = Table (data_quality, splitByRow=1, colWidths=(1.5*inch, 3.*inch, .6*inch))
		t.setStyle (TableStyle (
			[('BOX', (0, 0), (-1, -1), 1, colors.black),
			('BACKGROUND', (0, 0), (2, 0), colors.lavender),
			('LINEABOVE', (0, 1), (-1, 1), 0.7, colors.grey),
			('ALIGN', (0, 0), (-1, 0), 'CENTER'),
			('ALIGN', (0, 1), (0, -1), 'CENTER'),
			('VALIGN', (0, 1), (0, -1), 'MIDDLE'),
			('ALIGN', (2, 1), (2, -1), 'CENTER'),
			('VALIGN', (2, 1), (2, -1), 'MIDDLE'),]
			))

		story.append (t)

		doc.build (story)

	def get_pipeline (self, max_evals=10, frac=0.1, pop_size=40, impute_method="simple", tuner="bayesian", batch_size=256, pre_train=1, pre_train_epoch=10, epochs=10, report_path="./data_profile.pdf") :
		"""Get the best recommender pipeline for specify dataset
		:param frac: The frac of pre-train dataset
		:param impute_method: The method for imputing missing value
		:param batch_size: Batch size of training
		:param pre_train: whether to pretrain
		:param pre_train_epoch / epochs: the epochs of pre train / train
		:param report_path: the path of saving data report
		:param max_evals: max epoch of search
		"""
		self.pre_train = pre_train
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

		self.generate_pdf (report_path=report_path)

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
			best_model_score = 9999
		else :
			self.metrics = 1			# AUC
			self.task = "binary"
			best_score = 0
			self.maximize = True
			best_model_score = 0

		for model in self.model_list :
			if pre_train :
				print ("###Pre-training the ", model.__name__, " to find the best feature comibinatin")
				print ("###CONFIGURE")
				print ('###pre-train-epochs: ', pre_train_epoch)
				print ('###training-epochs: ', epochs)
				print ('###batch_size: ', batch_size)
				vis_list = []
				score_dict = {}
				print ("ORIGINAL...")
				self.data = sample_data
				self.feature_engineering ()
				self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch, Model=model)
				self.input_list = []
                
				for c1 in column_list :
					vis_list.append (c1)
					for c2 in column_list :
						if self.data_schema[c1] != 'numerical' and self.data_schema[c2] != 'numerical' :
							if c1 == c2 :
								print ("Combination: ", c1, " ...")
								self.data = sample_data
								self.feature_engineering (col_list=[c1])
								self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch, Model=model)
								self.input_list = []
								score_dict[self.score[-1]] = [c1]

							elif c1 != c2 and c2 not in vis_list :
								print ("Combination: ", c1, " ", c2, " ...")
								self.data = sample_data
								self.feature_engineering (col_list=[c1, c2])
								self.run (if_tune=1, batch_size=batch_size, epochs=pre_train_epoch, Model=model)
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
			self.best_com = best_com
			self.feature_engineering (col_list=best_com)
			self.run (batch_size=batch_size, epochs=epochs, Model=model)
			s_param, s_score = self.search (batch_size=batch_size, max_evals=max_evals, epochs=epochs, Model=model, tuner=tuner, pop_size=pop_size)
			if self.metrics == 0 :
				if best_model_score > s_score :
					best_model = model
					best_model_score = s_score
					best_param = s_param
			else :
				if best_model_score < s_score :
					best_model = model
					best_model_score = s_score
					best_param = s_param
		
		return best_model, best_model_score, best_param

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
		self.quality_list.append (qod._get_outlier ())
		self.quality_list.append (qod._get_completeness ())
		self.quality_list.append (qod._get_duplicated ())
		self.parity_score = qod._get_class_parity ()
		self.quality_list.append (self.parity_score)
		self.quality_list.append (qod._get_correlations ())
		print ("...done!")

	def data_cleaning (self, impute_method="simple") :
		"""clean the data
		"""
		print ("Data Cleaning ......")
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

			train, test = train_test_split (self.data, test_size=test_size, random_state=2021)
			# feature combination by target encoding
			if col_list != None :
				col_list1 = col_list.copy ()
				col_list1.append (self.target)
				te_train = train[col_list1].groupby (col_list).mean ()
				te_test = test[col_list1].groupby (col_list).mean ()
				te_train = te_train.reset_index ()
				te_test = te_test.reset_index ()
				new_col_name = 'TE'
				for c in col_list :
					new_col_name = new_col_name + '_' + str (c)
				col_list2 = col_list.copy ()
				col_list2.append (new_col_name) 
				te_train.columns = col_list2
				te_test.columns = col_list2
				train = train.merge (te_train, how='left', on=col_list)
				test = test.merge (te_test, how='left', on=col_list)
				self.data_schema[new_col_name] = 'numerical'
				feature_names = np.append (feature_names, new_col_name)

			for key, value in self.data_schema.items () :
				if key != self.target :
					if value != 'numerical' :
						lbe = LabelEncoder ()
						train[key] = lbe.fit_transform (train[key])
						test[key] = lbe.fit_transform (test[key])
						fixlen_feature_columns.append (SparseFeat (key, self.data[key].nunique ()))
					
					else :
						fixlen_feature_columns.append (DenseFeat (key, 1, ))
			
			if col_list != None :
				self.data_schema.pop (new_col_name)

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

# 			if self.metrics == 1 :
# 				self.score.append (round (roc_auc_score (test[self.target].values, pred_ans), 4))
# 			else :
# 				self.score.append (round (mean_squared_error (test[self.target].values, pred_ans), 4))
		
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
            
# 			if self.metrics == 1 :
# 				self.score.append (round (roc_auc_score (test[self.target].values, pred_ans), 4))
# 			else :
# 				self.score.append (round (mean_squared_error (test[self.target].values, pred_ans), 4))

	def search (self, batch_size=256, Model=DeepFM, tuner="bayesian", save_path="./PKL/", hp_path="./HP/", max_evals=10, epochs=100, pop_size=40) :
		"""Search the best hyperparameters for model
		"""
		if max_evals == 0 :
			return
        
		save_path = save_path + tuner + "/"
		hp_path = hp_path + tuner + "/"

		train = self.input_list[0]
		# if train[self.target].nunique () > 2:
		# 	metrics = 0			# MSE
		# 	task = "regression"
		# 	print ("TASK: ", task)
		# else :
		# 	metrics = 1			# AUC
		# 	task = "binary"
		# 	print ("TASK: ", task)

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

		if tuner == "random" :
			print ("Tuning the %s model by %s..." % (Model.__name__, tuner))
			random_search = RandomSearch (model_name=Model.__name__, linear_feature_columns=self.input_list[4],
										dnn_feature_columns=self.input_list[5], task=self.task, metrics=self.metrics,
										device="cpu", max_evals=max_evals, save_path=save_path, batch_size=batch_size)

			best_param, best_score = random_search.search (self.input_list[2], self.input_list[0][self.target].values, self.input_list[3], 
												self.input_list[1][self.target].values, epochs=epochs)
									
			with open (hp_path + Model.__name__ + "_" + str (1) + ".json", "w") as f :
				f.write (json.dumps (best_param, ensure_ascii=False, indent=4, separators=(',', ':')))


		if tuner == "bayesian" :
			print ("Tuning the %s model by %s..." % (Model.__name__, tuner))
			bayesian_search = BayesianOptimization (inputs=self.input_list, random_state=None, verbose=2, bounds_transformer=None, device=device,
													model_name=Model.__name__, epochs=epochs, max_evals=max_evals, target=self.target, metrics=self.metrics,
													task=self.task, batch_size=batch_size, save_path=save_path)	
			best_param, best_score = bayesian_search.maximize ()

			with open (hp_path + Model.__name__ + "_" + str (1) + ".json", "w") as f :
				f.write (json.dumps (best_param, ensure_ascii=False, indent=4, separators=(',', ':')))
			if self.pre_train : 
				# if pre_train == 1 then save the best combination
				with open (hp_path + Model.__name__ + "_" + str (1) + "_com.txt", "w") as file :
					for bc in self.best_com :
						file.write (bc)
						file.write ('\n')

		elif tuner == "genetic" :
			print ("Tuning the %s model by %s..." % (Model.__name__, tuner))
			geneticsearch = GeneticHyperopt (train_model_input=self.input_list[2], train_y=self.input_list[0][self.target].values, test_model_input=self.input_list[3], test_y=self.input_list[1][self.target].values,
											model_name=Model.__name__, linear_feature_columns=self.input_list[4], maximize=self.maximize, metrics=self.metrics, dnn_feature_columns=self.input_list[5], task=self.task, device="cpu",
											batch_size=batch_size, num_gen=max_evals, epochs=epochs, pop_size=pop_size)
			if Model.__name__ == "DeepFM" :
				l2_reg_linear_param = ContinuousParam ("l2_reg_linear", 0.5, 0.1, min_limit=0, max_limit=1, is_int=False)
				l2_reg_embedding_param = ContinuousParam ("l2_reg_embedding", 0.5, 0.1, min_limit=0, max_limit=1, is_int=False)
				l2_reg_dnn_param = ContinuousParam ("l2_reg_dnn", 0.5, 0.1, min_limit=0, max_limit=1, is_int=False)
				init_std_param = ContinuousParam ("init_std", 0.5, 0.1, min_limit=0, max_limit=1, is_int=False)
				# dnn_dropout_param = ContinuousParam ("dnn_dropout", 0.5, 0.01, min_limit=0, max_limit=1, is_int=False)
				
				geneticsearch.add_param (l2_reg_linear_param).add_param (l2_reg_embedding_param).add_param (l2_reg_dnn_param).add_param (init_std_param)

			best_params, best_score = geneticsearch.evolve()

		return best_param, best_score
	# def _get_model (self, models=[]) :
	# 	"""Get models

	# 	Models list: [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, DCNMix, ONN, WDL]
	# 	"""
	# 	model_list = []
	# 	for model in models :
	# 		if model == "DeepFM" :
	# 			model_list.append (DeepFM)
	# 		elif model == "xDeepFM" :
	# 			model_list.append (xDeepFM)
	# 		elif model == "AFN" :
	# 			model_list.append (AFN)
	# 		elif model == "NFM" :
	# 			model_list.append (NFM)
	# 		elif model == "IFM" :
	# 			model_list.append (IFM)
	# 		elif model == "DIFM" :
	# 			model_list.append (DIFM)
	# 		elif model == "AutoInt" :
	# 			model_list.append (AutoInt)
	# 		elif model == "PNN" :
	# 			model_list.append (PNN)
	# 		elif model == "DCN" :
	# 			model_list.append (DCN)
	# 		elif model == "DCNMix" :
	# 			model_list.append (DCNMix)
	# 		elif model == "ONN" :
	# 			model_list.append (ONN)
	# 		elif model == "WDL" :
	# 			model_list.append (WDL)

	# 	return model_list


