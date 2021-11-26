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

def _get_hp_grid (model) :
	"""Get hyperparameters of each models
	"""
	hp_grid = {}
	if model == "DeepFM" or model == "IFM" or model == "DIFM" or model == "ONN":
		hp_grid = {
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "xDeepFM" :
		hp_grid = {
			"cin_layer_size": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"l2_reg_cin": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "NFM" :
		hp_grid = {
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
			"bi_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "PNN" :
		hp_grid = {
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "DCN" :
		hp_grid = {
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_cross": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "DCNMix" :
		hp_grid = {
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_cross": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
			"low_rank": np.array ([4, 16, 32, 64, 128]),
			"num_experts": np.array ([4, 8, 16, 32])
		}
	elif model == "AFN" :
		hp_grid = {
			"ltl_hidden_size": np.array ([4, 16, 32, 64, 128, 256, 1024, 2048]),
			"afn_dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_linear": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}
	elif model == "AutoInt" :
		hp_grid = {
			"att_layer_num": np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
			"att_head_num": np.array ([2]),
			"dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
			"l2_reg_embedding": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"l2_reg_dnn": np.array ([0.0, 0.01, 0.1]),
			"init_std": np.array ([0.00001, 0.0001, 0.001, 0.01, 0.1]),
			"dnn_dropout": np.arange (0, 1, 0.1),
		}


	return hp_grid

class RandomSearch () :
	"""Random search for hp tuning

	:param models: Model list of tuning
	:param max_evals: The max rounds of tuning
	"""
	def __init__ (self, model_name, linear_feature_columns, metrics, batch_size, dnn_feature_columns, save_path, task="binary", device="cpu", max_evals=10) :
		self.model_name = model_name
		self.max_evals = max_evals
		self.linear_feature_columns = linear_feature_columns
		self.dnn_feature_columns = dnn_feature_columns
		self.task = task
		self.device = device
		self.hp_grid = _get_hp_grid (model_name)
		self.save_path = save_path
		self.batch_size = batch_size
		self.metrics = metrics

	def search (self, train_model_input, train_y, test_model_input, test_y, epochs=100, verbose=2, earl_stop_patience=0) :
		best_Score = 0
		best_param = {}
		best_round = 0
		if self.metrics == 1 :
			best_score = 0
		elif self.metrics == 0 :
			best_score = 9999
		with alive_bar (self.max_evals) as bar :
			for i in range (self.max_evals) :
				random_params = {k: random.sample (v.tolist (), 1)[0] if not isinstance (v, tuple) else (random.sample (v[0].tolist (), 1)[0], random.sample (v[1].tolist (), 1)[0]) for k, v in self.hp_grid.items ()}
				
				if self.model_name == "DeepFM" :
					model = DeepFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "xDeepFM" :
					model = xDeepFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "IFM" :
					model = IFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "DIFM" :
					model = DIFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "NFM" :
					model = NFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "ONN" :
					model = ONN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "PNN" :
					model = PNN (dnn_feature_columns=self.dnn_feature_columns, task=self.task, device=self.device, **random_params)
				elif self.model_name == "DCN" :
					model = DCN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "DCNMix" :
					model = DCNMix (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "AFN" :
					model = AFN (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
										task=self.task, device=self.device, **random_params)
				elif self.model_name == "AutoInt" :
					model = AutoInt (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, **random_params)
				elif self.model_name == "WDL" :
					model = WDL (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, **random_params)

				if self.metrics == 1 :
					model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"], )
				elif self.metrics == 0 :
					model.compile ("adam", "mse", metrics=["mse"], )

				model.fit (train_model_input, train_y, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience, if_tune=1, batch_size=self.batch_size)
				pred_ans = model.predict (test_model_input, self.batch_size)

				if self.metrics == 1 :
					cur_score = roc_auc_score (test_y, pred_ans)
				elif self.metrics == 0 :
					cur_score = mean_squared_error (test_y, pred_ans)

				if self.metrics == 1 and cur_score > best_score :
					best_round = i
					best_score = cur_score
					best_param = random_params
				elif self.metrics == 0 and cur_score < best_score :
					best_round = i
					best_score = cur_score
					best_param = random_params

				bar ()
				bar.text ("#%d  score: %.4f		Best score currently: %.4f" % (i + 1, cur_score, best_score))
				torch.save (model.state_dict (), self.save_path + self.model_name + "_epoach:" + str (epochs) + ".pkl")

		print ("Best Score: %.4f in %d" % (best_score, best_round))
		# print ("Best Hyperparameters: ", (best_param))
		return best_param, best_score




