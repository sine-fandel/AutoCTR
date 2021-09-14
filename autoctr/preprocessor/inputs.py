# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

show the data profile, clean the data 
and convert the input data into suitable version

"""
import pandas as pd
import numpy as np
from .cleaning import Impute
from .profile import Profiling
from .feature_column import SparseFeat, DenseFeat
from sklearn.model_selection import train_test_split


class Input (object) :
	def __init__ (self, data_path, sep=",", target="label", test_size=0.2) :
		self.data = pd.read_csv (data_path, sep=sep)
		self.target = target

	def preprocessing (self, outlier="z_score", correlation="pearson", impute_method="knn") :
		profile = Profiling (self.data, outlier=outlier, correlation=correlation)
		impute = Impute (self.data)

		print ("************************************ The Profile of the Dataset ************************************")
		types_dict = profile.summary ().loc['types'].to_dict ()
		print (profile.summary ())
		
		feature_names = profile.summary ().columns.values
		feature_names_temp = list (feature_names)
		feature_names = np.delete (feature_names, feature_names_temp.index (self.target))

		if impute_method == 'knn' :
			self.data = impute.KnnImputation (n_neighbors=2)
		elif impute_method == 'simple' :
			self.data = impute,SimpleImputation ()
		elif impute_method == 'iterative' :
			self.data = impute.IterativeImputation ()
		elif impute_method == 'forest' :
			self.data = impute.RandomforestImputation ()
		elif impute_method == 'mf' :
			self.data = impute.MatrixFactorization ()

		print ("Finished imputation by ", impute_method)
		
		fixlen_feature_columns = []
		for key, value in types_dict.items () :
			if key != self.target :
				if value == "categorical" :
					fixlen_feature_columns.append (SparseFeat (key, self.data[key].nunique ()))
					
				elif value == "numeric" :
					fixlen_feature_columns.append (DenseFeat (key, 1, ))
		train, test = train_test_split (self.data, test_size=0.2, random_state=2021)
		train_model_input = {name: train[name] for name in feature_names}
		test_model_input = {name: test[name] for name in feature_names}
		dnn_feature_columns = fixlen_feature_columns
		linear_feature_columns = fixlen_feature_columns

		return train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns

		




# def run (data_path, outlier='z_score', correlation='pearson'):
# 	"""Entry point for console_scripts
# 	"""

# 	profiling = Input (data_path=data_path, sep=",")

# 	profiling.preprocessing ()


# if __name__ == "__main__" :
# 	# data = pd.read_csv('/Users/apple/Downloads/tf-experiment/criteo_sample.txt')

# 	run (data_path='/Users/apple/Downloads/tf-experiment/criteo_sample.txt')