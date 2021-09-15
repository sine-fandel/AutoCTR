from .preprocessor.inputs import Input
import pandas as pd
from .models import *
from sklearn.metrics import log_loss, roc_auc_score

class AutoCTR :
	"""The whole process of recommender
	
	:param data_path: The path of dataset
	:param target: The training / testing target of all model
	:param sep: The seq of the dataset
	"""
	def __init__ (self, data_path, target, seq=",") :
		self.data_path = data_path
		self.target = target
		self.model_list = [DeepFM, AutoInt]
		self.seq = seq
		self.input_list = []

	def preprocessing (self) :
		profiling = Input (data_path=self.data_path, sep=self.seq)
		self.input_list = profiling.preprocessing (impute_method='iterative')

	def run (self) :
		train = self.input_list[0]
		test = self.input_list[1]
		train_model_input = self.input_list[2]
		test_model_input = self.input_list[3]
		linear_feature_columns = self.input_list[4]
		dnn_feature_columns = self.input_list[5]

		for Model in self.model_list :
			model = Model (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device='cpu')
			model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
			model.fit (train_model_input, train[self.target].values, batch_size=32, epochs=100, verbose=2, validation_split=0.8)
			pred_ans = model.predict (test_model_input, 256)

			print ("test AUC:", round (roc_auc_score(test[self.target].values, pred_ans), 4))


	