"""

author:
	Zhengxin Fang, 358746595@qq.com

Random Search for hyper-parameters tuning.

"""
import os
import torch
import random
import numpy as np
from alive_progress import alive_bar

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

from ..models import *


class RandomSearch () :
	"""Random search for hp tuning

	:param models: Model list of tuning
	:param max_evals: The max rounds of tuning
	"""
	def __init__ (self, model_name, hp_grid, linear_feature_columns, dnn_feature_columns, task="binary", device="cpu", max_evals=10) :
		self.model_name = model_name
		self.max_evals = max_evals
		self.linear_feature_columns = linear_feature_columns
		self.dnn_feature_columns = dnn_feature_columns
		self.task = task
		self.device = device
		self.hp_grid = hp_grid

	def search (self, train_model_input, train_y, test_model_input, test_y, epochs=100, verbose=2, earl_stop_patience=0) :
		best_Score = 0
		best_param = {}
		best_round = 0
		with alive_bar (self.max_evals) as bar :
			for i in range (self.max_evals) :
				random_params = {k: random.sample (v.tolist (), 1)[0] if not isinstance (v, tuple) else (random.sample (v[0].tolist (), 1)[0], random.sample (v[1].tolist (), 1)[0]) for k, v in self.hp_grid.items ()}
				
				if self.model_name == "DeepFM" :
					model = DeepFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "xDeepFM" :
					model = xDeepFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, cin_layer_size=random_params['cin_layer_size'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'], l2_reg_cin=random_params['l2_reg_cin'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "IFM" :
					model = IFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "DIFM" :
					model = DIFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "NFM" :
					model = NFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'], bi_dropout=random_params['bi_dropout'])
				elif self.model_name == "ONN" :
					model = ONN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "PNN" :
					model = PNN (dnn_feature_columns=self.dnn_feature_columns, task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_embedding=random_params['l2_reg_embedding'], l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "DCN" :
					model = DCN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'], l2_reg_cross=random_params['l2_reg_cross'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "DCNMix" :
					model = DCNMix (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'], l2_reg_cross=random_params['l2_reg_cross'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'],
										low_rank=random_params['low_rank'], num_experts=random_params['num_experts'])
				elif self.model_name == "AFN" :
					model = AFN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, ltl_hidden_size=random_params['ltl_hidden_size'], afn_dnn_hidden_units=random_params['afn_dnn_hidden_units'],
										l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])
				elif self.model_name == "AutoInt" :
					model = AutoInt (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										att_layer_num=random_params['att_layer_num'], att_head_num=random_params['att_head_num'],
										task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'], l2_reg_embedding=random_params['l2_reg_embedding'],
										l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])

				model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
				model.fit (train_model_input, train_y, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience, if_tune=1)
				pred_ans = model.predict (test_model_input, 256)

				cur_score = roc_auc_score (test_y, pred_ans)
				if cur_score > best_Score :
					best_round = i
					best_Score =cur_score
					best_param = random_params

				bar ()
				bar.text ("#%d  Accuracy: %.4f" % (i + 1, cur_score))

		print ("Best Accuracy: %.4f in %d" % (best_Score, best_round))
		print ("Best Hyperparameters: ", (best_param))




