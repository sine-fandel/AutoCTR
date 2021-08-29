# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

show the data profile, clean the data 
and convert the input data into suitable version

"""
import pandas as pd
from .cleaning import Impute
from .profile import Profiling
from sklearn.model_selection import train_test_split

class Input (object) :
	def __init__ (self, data_path, sep=",", target="rating", outlier="z_score", correlation="pearson", impute_method="knn", test_size=0.2) :
		self.data = pd.read_csv (data_path, sep=sep)
		self.outlier = outlier
		self.correlation = correlation
		self.impute_method = impute_method

	def preprocessing (self) :
		profile = Profiling (self.data, outlier=self.outlier, correlation=self.correlation)
		impute = Impute (self.data)
		print ("************************************ The Profile of the Dataset ************************************")
		print (profile.summary ())
		feature_names = profile.summary ().columns.values
		summary = profile.summary ().values

		if self.impute_method == 'knn' :
			impute.KnnImputation (n_neighbors=2)
		elif self.impute_method == 'simple' :
			impute,SimpleImputation ()
		elif self.impute_method == 'iterative' :
			impute.IterativeImputation ()
		elif self.impute_method == 'forest' :
			impute.RandomforestImputation ()
		elif self.impute_method == 'mf' :
			impute.MatrixFactorization ()

		print ("Finished imputation by ", self.impute_method)

		train, test = train_test_split (self.data, test_size=0.2, random_state=2021)
		train_model_input = {name: train[name] for name in feature_names}
		test_model_input = {name: test[name] for name in feature_names}

		return train_model_input, test_model_input

		




# def run (data_path, outlier='z_score', correlation='pearson'):
# 	"""Entry point for console_scripts
# 	"""

# 	profiling = Input (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-1m/ratings.dat', sep="::")

# 	print (profiling.preprocessing ())


# if __name__ == "__main__" :
# 	run (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-1m/ratings.dat')    
# 	# run (data_path='/Users/apple/project/AI + elearning project/data_profiling/test/west.csv')