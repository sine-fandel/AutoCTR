# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

Data quality checking

"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn import preprocessing
from collections import Counter

class QoD (object) :
	def __init__ (self, df, target='label') :
		self.df = df
		self.length = len (self.df)
		self.target = target

	def Plot (self, score, title) :
		# just finished the missing value detect
		"""Plot the outlier score by ring pie 
		:param score: the score of ploting
		:param title: the type of score
		"""
		plt.figure (figsize=(2, 2))
		for outlier in score.items () :
			key = outlier[0]
			value = outlier[1]

			if value != 1.0 :
				groups = [value, 1 - value]
			else :
				groups = [value]
			
			plt.pie (groups, wedgeprops=dict (width=0.25, edgecolor='w'), colors=['dodgerblue', 'skyblue'])
			plt.text (x=-0.2, y=-0.08, s=value, fontsize=12)
			plt.title (key)

		# plt.savefig ('./' + title + '_Score.png')

	def _get_outlier (self, method="z_score") :
		"""Get outlier score
			score = (total - outliers_count) / total
		:param method: the method of getting outlier score
		"""
		score = {}
		temp = 0
		counts = 0
		for c in self.df.columns :
			if self.df[c].dtype != 'object' and c != self.target:
				if method == 'z_score' :
					count = 0
					counts += self.df[c].count ()
					z_score = (np.array (self.df[c]) - np.mean (self.df[c])) / np.std (self.df[c])
					for z in np.abs (z_score) :
						# print (z)
						if z > 3 :
							count += 1

					temp += count
					
		score['Outlier Detection'] = round ((counts - temp) / counts, 2)

		self.Plot (score, "Outlier")

		return score

	def _get_completeness (self) :
		# to do list:
		# 	format
		#	typo
		"""Get the missing score
			score = (total - missingvalue - outlier_counts) / total
		"""
		score = {}
		count = 0
		counts = 0
		for c in self.df.columns :
			if c != self.target :
				count += len (self.df[c]) - self.df[c].isnull().sum()
				counts += self.length

		score['Data Completeness'] = round (count / counts, 2)
		print ("Missing number: ", counts - count)
		self.Plot (score, "Completeness")

		return score

	def _get_duplicated (self) :
		"""Get the duplicated of dataset
			score = duplicated_count / total
		"""
		duplicated_list = self.df.duplicated ().values
		duplicated_count = Counter (duplicated_list)

		score = {}
		score['Data Duplicated'] = round (duplicated_count[False] / self.length, 2)
		print ("Duplicated number: ", self.length - duplicated_count[False])
		self.Plot (score, "Duplicated")

		return score

	def _get_class_parity (self, method="z_score") :
		"""Get class parity score
			class imbalance: Shannon Entropy (score: 1 -> best balance)
			class purity: (total - missingvalue)
			parity score = (class imbalance + class purity) / 2
		:param method: the method of getting outliers
		"""
		label_counts_list = [value for value in Counter (self.df[self.target].values).values ()]

		noisy_count = 0
		noisy_count += self.df[self.target].isnull().sum()
		if method == 'z_score' :
			count = 0
			label_zscore = {}
			z_score = (np.array (self.df[self.target]) - np.mean (self.df[self.target])) / np.std (self.df[self.target])

			n = 0
			for l in self.df[self.target].values :
				label_zscore[l] = z_score[n]
				n += 1
			
			t = 0
			for key in label_zscore :
				# remove outlier for calculating the entropy
				if np.abs (label_zscore[key]) > 3 :
					label_counts_list.pop (t)
					t -= 1

				t += 1

			for z in np.abs (z_score) :
				if z > 3 :
					count += 1

			# if count not in label_counts_list :
			noisy_count += count

		label_total = sum (label_counts_list)
		entro = 0

		div_list = [-(x / label_total) * math.log (x / label_total, len (label_counts_list)) for x in label_counts_list]
		entro += sum (div_list)

		score = {}
		score['Class Parity'] = round ((1 - noisy_count / self.length + entro) / 2, 2)

		self.Plot (score, "Class Parity")

		return score

	def _get_correlations (self, method="pearson") :
		"""Get correlation between the data
		"""
		# data = self.df.drop (labels=self.target, axis=1)
		cat_col = []
		for c in self.df.columns :
			if self.df[c].dtypes == "object" :
				cat_col.append (c)
		
		data_type = 2
		column_types = self.df.dtypes.values
		column_types = column_types == 'O'
		for c in column_types :
			if c and data_type == 2:
				data_type = 0
			elif not c and data_type == 0 :
				data_type = -1
			elif not c and data_type == 2 :
				data_type = 1
			elif c and data_type == 1 :
				data_type = -1

		if data_type == 0 :
			# data type is both categorical
			label = preprocessing.LabelEncoder ()
			data_encoded = pd.DataFrame ()

			for i in self.df.columns :
				data_encoded[i] = label.fit_transform (data[i])

			rows= []
			for var1 in data_encoded:
				col = []
				for var2 in data_encoded :
					cramers = self.CramerV (data_encoded[var1], data_encoded[var2]) # Cramer's V test
					col.append (round (cramers, 2)) # Keeping of the rounded value of the Cramer's V  
				rows.append (col)
			
			cramers_results = np.array (rows)
			cor_result = pd.DataFrame (cramers_results, columns = data_encoded.columns, index =data_encoded.columns)
		
		elif data_type == 1 :
			# data type is both numerical
			cor_result = self.df.corr (method=method)

		elif data_type == -1 :
			# data type is numerical and categorical
			# label = preprocessing.LabelEncoder ()

			for c in cat_col :
				lbe = preprocessing.LabelEncoder()
				self.df[c] = self.df[c].astype ('str')
				self.df[c] = lbe.fit_transform (self.df[c])

			cor_result = self.df.corr (method=method)

		cor_result = cor_result.drop (labels=self.target, axis=1)
		cor_result = cor_result.drop (labels=self.target, axis=0)
		cor_result = np.abs (cor_result) > 0.5
		high_cor_count = 0
		for row in cor_result.values :
			for column in row :
				if column :
					high_cor_count += 1

		high_cor_count = (high_cor_count - cor_result.shape[0]) / 2

		score = {}
		score['Correlation Dection'] = round ((cor_result.shape[0] - high_cor_count) / cor_result.shape[0], 2)

		self.Plot (score, "Correlation Score")

		return score

	def plot_sparse (self, col) :
		"""Plot the figure of sparse of each columns
		:param col: the column that to be plot
		"""
		stats = self.df[[col, self.target]].groupby (col).agg (['count', 'mean', 'sum'])
		stats = stats.reset_index ()
		stats.columns = [col, 'count', 'mean', 'sum']
		stats_sort = stats['count'].value_counts ().reset_index ()
		stats_sort = stats_sort.sort_values ('index')
		plt.figure (figsize=(15,4))
		plt.bar (stats_sort['index'].astype (str).values[0:20], stats_sort['count'].values[0:20])
		plt.title ('Frequency of ' + str(col))
		plt.xlabel ('Number frequency')
		plt.ylabel ('Frequency')

		plt.show ()

	def plot_top20(self, col):
		stats = self.df[[col, self.target]].groupby(col).agg(['count', 'mean', 'sum'])
		stats = stats.reset_index()
		stats.columns = [col, 'count', 'mean', 'sum']
		stats = stats.sort_values('count', ascending=False)
		fig, ax1 = plt.subplots(figsize=(15,4))
		ax2 = ax1.twinx()
		ax1.bar(stats[col].astype(str).values[0:20], stats['count'].values[0:20])
		ax1.set_xticklabels(stats[col].astype(str).values[0:20], rotation='vertical')
		ax2.plot(stats['mean'].values[0:20], color='red')
		ax2.set_ylim(0,1)
		ax2.set_ylabel('Mean Target')
		ax1.set_ylabel('Frequency')
		ax1.set_xlabel(col)
		ax1.set_title('Top20 ' + col + 's based on frequency')

		plt.show ()

# data = pd.read_csv ("/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data", sep="\\s+")
# qod = QoD (data, target="ratings")

# cat = 'user'
# te = data[[cat, 'ratings']].groupby (cat).mean ()
# te = te.reset_index ()
# te.columns = [cat, 'TE_' + cat]
# data = data.merge (te, how='left', on=cat)

# qod._get_correlations ()
# # outlier_dict = qod._get_completeness ()
# # completeness = qod._get_outlier ()
# # duplicated = qod._get_duplicated ()
# # parity = qod._get_class_parity ()

# # print (outlier_dict)
# # print (completeness)
# # print (duplicated)
# # print (parity)