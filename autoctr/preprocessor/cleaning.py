# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

Clean the data (impute missing values and outliers)

"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import math
from sklearn.datasets import load_boston 

class Impute :
	def __init__ (self, df) :
		self.df = df

	def SimpleImputation (self, strategy='mean') :
		"""
		Impute missing data by some constant value
		strategy: choose the strategy for SingleImputation
				includes: mean, mid...
		"""
		ic_data = self.df
		imp = SimpleImputer (missing_values=np.nan, strategy=strategy)
		c_data = imp.fit_transform (ic_data)
		cp = pd.DataFrame (c_data)
		cp.columns = ic_data.columns.values

		return cp

	def IterativeImputation (self, max_iter=10) :
		"""
		Impute missing data by combining other features
		max_iter: the number of fiting rounds
		"""
		ic_data = self.df
		imp = IterativeImputer(max_iter=max_iter, sample_posterior=True, random_state=0)
		c_data = imp.fit_transform (ic_data)
		cp = pd.DataFrame (c_data)
		cp.columns = ic_data.columns.values

		return cp

	def RandomforestImputation (self, n_estimators=100) :
		"""
		Impute missing data by using Regression Tree
		n_estimators: the number of TREE
		"""
		ic_data = self.df.copy ()
		sortindex = np.argsort (ic_data.isnull ().sum (axis=0)).values
		for i in sortindex :
			df = ic_data
			fillc = df.iloc[:, i]
			if fillc.isnull ().any () == False :
				continue
			df = df.iloc[:, df.columns != i]
			df_0 = SimpleImputer (missing_values=np.nan, strategy='constant', fill_value=0).fit_transform (df)

			# divide training and testing set
			ytrain = fillc[fillc.notnull ()]
			ytest = fillc[fillc.isnull ()]
			xtrain = df_0[ytrain.index, :]
			xtest = df_0[ytest.index, :]

			rfc = RandomForestRegressor (n_estimators=n_estimators)
			rfc = rfc.fit (xtrain, ytrain)
			y_predict = rfc.predict (xtest)

			ic_data.loc [ic_data.iloc[:, i].isnull (), i] = y_predict
		
		cp = pd.DataFrame (ic_data)
		cp.columns = ic_data.columns.values

		return cp


	def KnnImputation (self, n_neighbors, weights="uniform") :
		"""
		impute missing data by knn algorithm
		n_neighbors: the number of the nearest neighbors
		weights: the weight of each neighbors
		"""
		ic_data = self.df.copy ()
		knn = KNNImputer (n_neighbors=n_neighbors, weights=weights)
		c_data = knn.fit_transform (ic_data)
		cp = pd.DataFrame (c_data)
		cp.columns = ic_data.columns.values

		return cp

	def MatrixFactorization (self, lr=0.00001, lamda=0.001, epoch=500, factor=20) :
		"""
		Matrixfactor to impute missing data
		"""
		if isinstance (self.df, pd.DataFrame) :
			data = self.df.values
		else :
			data = self.df
		
		# np.seterr (all='raise')
		f = factor
		U = np.random.rand (data.shape[0], f)
		M = np.random.rand (data.shape[1], f)

		epoch = 500
		lr = lr
		lamda = lamda

		for e in range (epoch) :
			rmse = 0
			n = 0
			for i in range (data.shape[0]) :
				for j in range (data.shape[1]) :
					if not np.isnan (data[i][j]) :
						error = data[i][j] - np.dot (U[i], M[j].T)
						rmse += (error ** 2)
						n +=1 
						U[i] = U[i] + lr * (error * M[j] - lamda * U[i])
						M[j] = M[j] + lr * (error * U[i] - lamda * M[j])
		
			trainRmse = math.sqrt (rmse * 1.0 / n)
		
		c_data = np.dot (U, M.T)
		cp = pd.DataFrame (c_data)
		cp.columns = ic_data.columns.values

		return cp
		


# ic_data = pd.read_csv ('/Users/apple/project/AI + elearning project/data_imputation/test/heart.csv')
# Y = pd.read_csv ('/Users/apple/project/AI + elearning project/data_imputation/test/heart.csv')
# for i in range (200) :
# 	import random
# 	x = random.randint (0, 505)
# 	y = random.randint (0, 12)
# 	ic_data[x, y] = np.nan

# X = pd.DataFrame (ic_data)
# Y = pd.DataFrame (Y)
# print (X)
# imp = Impute (X)
# print (pd.DataFrame (imp.KnnImputation (n_neighbors=2)))

