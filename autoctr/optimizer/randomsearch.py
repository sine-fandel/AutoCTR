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


class RandomSearch () :
	"""Random search for hp tuning

	:param models: Model list of tuning
	:param max_evals: The max rounds of tuning
	"""
	def __init__ (self, Model, hp_grid, linear_feature_columns, dnn_feature_columns, task="binary", device="cpu", max_evals=10) :
		self.Model = Model
		self.max_evals = max_evals
		self.linear_feature_columns = linear_feature_columns
		self.dnn_feature_columns = dnn_feature_columns
		self.task = task
		self.device = device
		self.hp_grid = hp_grid

	def search (self, train_model_input, train_y, test_model_input, test_y, epochs=100, verbose=2, earl_stop_patience=0) :
		best_Score = 0
		best_param = {}
		with alive_bar (self.max_evals) as bar :
			for i in range (self.max_evals) :
				random_params = {k: random.sample (v.tolist (), 1)[0] if not isinstance (v, tuple) else (random.sample (v[0].tolist (), 1)[0], random.sample (v[1].tolist (), 1)[0]) for k, v in self.hp_grid.items ()}

				model = self.Model (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns, 
									task=self.task, device=self.device, dnn_hidden_units=random_params['dnn_hidden_units'],
									l2_reg_linear=random_params['l2_reg_linear'], l2_reg_embedding=random_params['l2_reg_embedding'],
									l2_reg_dnn=random_params['l2_reg_dnn'], init_std=random_params['init_std'], dnn_dropout=random_params['dnn_dropout'])

				model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
				model.fit (train_model_input, train_y, epochs=epochs, verbose=verbose, earl_stop_patience=earl_stop_patience, if_tune=1)
				pred_ans = model.predict (test_model_input, 256)

				cur_score = roc_auc_score (test_y, pred_ans)
				if cur_score > best_Score :
					best_Score =cur_score
					best_param = random_params

				bar ()
				bar.text ("#%d  Accuracy: %.4f" % (i + 1, cur_score))

		print (best_Score)
		print (best_param)




