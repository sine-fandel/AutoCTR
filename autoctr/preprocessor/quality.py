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
from collections import Counter

class QoD (object) :
	def __init__ (self, df, target='label') :
		self.df = df
		self.length = len (self.df)
		self.label = target

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

		plt.savefig ('./' + title + '_Score.png')

	def _get_outlier (self, method="z_score") :
		"""Get outlier score
			score = (total - outliers_count) / total
		:param method: the method of getting outlier score
		"""
		score = {}
		temp = 0
		counts = 0
		for c in self.df.columns :
			if self.df[c].dtype != 'object' and c != self.label:
				if method == 'z_score' :
					count = 0
					counts += self.df[c].count ()
					z_score = (np.array (self.df[c]) - np.mean (self.df[c])) / np.std (self.df[c])
					for z in np.abs (z_score) :
						# print (z)
						if z > 3 :
							count += 1

					temp += count
					
		score['Outlier Score'] = round ((counts - temp) / counts, 2)

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
			if c != self.label :
				count += len (self.df[c]) - self.df[c].isnull().sum()
				counts += self.length

		score['Completeness Score'] = round (count / counts, 2)

		self.Plot (score, "Completeness")

		return score

	def _get_duplicated (self) :
		"""Get the duplicated of dataset
			score = duplicated_count / total
		"""
		duplicated_list = self.df.duplicated ().values
		duplicated_count = Counter (duplicated_list)

		score = {}
		score['duplicated'] = round (duplicated_count[False] / self.length, 2)

		self.Plot (score, "Duplicated")

		return score

	def _get_class_parity (self, method="z_score") :
		"""Get class parity score
			class imbalance: Shannon Entropy (score: 1 -> best balance)
			class purity: (total - missingvalue)
			parity score = (class imbalance + class purity) / 2
		:param method: the method of getting outliers
		"""
		label_counts_list = [value for value in Counter (self.df[self.label].values).values ()]

		noisy_count = 0
		noisy_count += self.df[self.label].isnull().sum()
		if method == 'z_score' :
			count = 0
			label_zscore = {}
			z_score = (np.array (self.df[self.label]) - np.mean (self.df[self.label])) / np.std (self.df[self.label])

			n = 0
			for l in self.df[self.label].values :
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
		score['class parity'] = round ((1 - noisy_count / self.length + entro) / 2, 2)

		self.Plot (score, "Class Parity")

		return score


# qod = QoD ("/Users/apple/AutoCTR project/AutoCTR/autoctr/criteo_sample.txt", sep=",", label="label")

# outlier_dict = qod._get_completeness ()
# completeness = qod._get_outlier ()
# duplicated = qod._get_duplicated ()
# parity = qod._get_class_parity ()

# print (outlier_dict)
# print (completeness)
# print (duplicated)
# print (parity)