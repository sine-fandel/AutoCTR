from .preprocessor.inputs import Input
import pandas as pd
from .models import *
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
import torch
import os

class AutoCTR :
	"""The whole process of recommender
	
	:param data_path: The path of dataset
	:param target: The training / testing target of all model
	:param sep: The seq of the dataset
	"""
	def __init__ (self, data_path, target, sep=",") :
		self.data_path = data_path
		self.target = target
		self.model_list = [DeepFM, xDeepFM, AFN, NFM, IFM, DIFM, AutoInt, PNN, DCN, DCNMix, ONN, WDL]
		self.sep = sep
		self.input_list = []

	def preprocessing (self, test_size=0.2) :
		profiling = Input (data_path=self.data_path, sep=self.sep, target=self.target, test_size=test_size)
		self.input_list = profiling.preprocessing (impute_method='iterative')

	def run (self, batch_size=32, epochs=100, verbose=2, save_path="./PKL/", earl_stop_patience=0) :
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
	