# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

show the data profile, clean the data 
and convert the input data into suitable version

"""

from .cleaning import Impute
from .profile import Profiling

class Input (object) :
	def __init__ (self, data, outlier='z_score', correlation='pearson', impute_method="knn") :
		self.data = data
		self.outlier = outlier
		self.correlation = correlation
		self.impute_method = impute_method

	def preprocessing (self) :
		profile = Profiling (self.data, outlier=self.outlier, correlation=self.correlation)
		impute = Impute (self.data)
		print (profile.summary ())
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

		

# def run (data_path, outlier='z_score', correlation='pearson'):
# 	"""Entry point for console_scripts
# 	"""
# 	data = pd.read_csv (data_path, sep='::')
# 	profiling = Input (data)

# 	print (profiling.preprocessing ())


# if __name__ == "__main__" :
# 	run (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-1m/ratings.dat')    
	# run (data_path='/Users/apple/project/AI + elearning project/data_profiling/test/west.csv')