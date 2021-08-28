# -*- coding:utf-8 -*-
"""

Author: 
	Zhengxin Fang, 358746595@qq.com

Get the data profile

"""

from collections import OrderedDict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
from pandas.api import types
from six import string_types
from scipy.stats import chi2_contingency
import numpy as np


class Profiling(object):
	ALL = 'ALL'
	INCLUDE = 'INCLUDE'
	EXCLUDE = 'EXCLUDE'

	TYPE_BOOL = 'bool'
	TYPE_NUMERIC = 'numeric'
	TYPE_DATE = 'date'
	TYPE_CATEGORICAL = 'categorical'
	TYPE_CONSTANT = 'constant'
	TYPE_UNIQUE = 'unique'

	def __init__(self, df, outlier='z_score', correlation='pearson', plot=False):
		self.df = df
		self.length = len(df)
		self.corr = df.corr()
		self.plot = plot
		self.outlier = outlier
		self.correlation = correlation

		
	def __getitem__(self, column):
		if isinstance(column, str) and self._clean_column(column):
			return self._get_column_summary (column)

		if isinstance(column, int) and column < self.df.shape[1]:
			return self._get_column_summary (self.df.columns[column])

		if isinstance(column, (tuple, list)):
			error_keys = [k for k in column if not self._clean_column(k)]
			if len(error_keys) > 0:
				raise KeyError(', '.join(error_keys))
			return self.df[list(column)].values

		if isinstance(column, pd.Index):
			error_keys = [
				k for k in column.values if not self._clean_column(k)]
			if len(error_keys) > 0:
				raise KeyError(', '.join(error_keys))
			return self.df[column].values

		if isinstance(column, np.ndarray):
			error_keys = [k for k in column if not self._clean_column(k)]
			if len(error_keys) > 0:
				raise KeyError(', '.join(error_keys))
			return self.df[column].values

		raise KeyError(column)

	@property
	def columns_types(self):
		return pd.value_counts (self.columns_stats.loc['types'])

	def summary (self):
		return pd.concat([self.df.describe(), self._get_stats (self.outlier, self.correlation)], sort=True)[self.df.columns]

	@staticmethod
	def _number_format(x):
		eps = 0.000000001
		num_format = '{0:,.0f}' if abs(int(x) - x) < eps else '{0:,.2f}'
		return num_format.format(x)

	@classmethod
	def _percent(cls, x) :
		x = cls._number_format (100 * x)
		return '{}%'.format(x)

	def _clean_column(self, column):
		if not isinstance(column, (int, string_types)):
			raise ValueError('{} is not a valid column'.format(column))
		return column in self.df.columns

	def _get_stats(self, outlier, correlation) :
		counts = self.df.count ()
		counts.name = 'counts'
		uniques = self._get_uniques()
		missing = self._get_missing (counts)
		skew = self._get_skew ()
		kurt = self._get_kurt ()
		outlier = self._get_outlier (outlier)

		# stats = pd.concat([counts, uniques, missing, skew, kurt, correlation], axis=1, sort=True)
		stats = pd.concat([counts, uniques, missing, skew, kurt, outlier], axis=1, sort=True)

		# settings types
		stats['types'] = ''
		columns_info = self._get_columns_info(stats)
		
		# both categorical: 0
		# both numerical: 1
		# numerical and categorical: -1
		data_type = 1
		self.cat_col = []
		for ctype, columns in columns_info.items () :
			if ctype == 'categorical' and len (columns) == self.df.shape[1] :
				self.cat_col = columns
				data_type = 0
			elif ctype == 'categorical' and len (columns) == 0 :
				data_type = 1
			elif ctype == 'categorical' and len (columns) != 0 and len (columns) != self.df.shape[1] :
				self.cat_col = columns
				data_type = -1

		correlation = self._get_correlations (correlation, data_type)
		stats['correlation'] = correlation

		for ctype, columns in columns_info.items() :
			stats.loc[columns, 'types'] = ctype

		return stats.transpose()[self.df.columns]

	def _get_correlations (self, method, data_type) :
		if data_type == 0 :
			# data type is both categorical
			label = preprocessing.LabelEncoder ()
			data_encoded = pd.DataFrame ()

			for i in self.df.columns :
				data_encoded[i] = label.fit_transform (self.df[i])

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
			label = preprocessing.LabelEncoder ()
			for c in self.cat_col :
				self.df[c] = label.fit_transform (self.df[c])

			cor_result = self.df.corr (method=method)

		return pd.Series (dict (cor_result[self.df.columns[-1]]), name='correlation')


	def CramerV (self, var1, var2) :
		'''Cramer's V method
		use cramer's V to calculate the correlation between two categorical variable
		'''
		crosstab =np.array (pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
		stat = chi2_contingency (crosstab)[0] # Keeping of the test statistic of the Chi2 test
		obs = np.sum (crosstab) # Number of observations
		mini = min (crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
		
		return (stat / (obs * mini))

	def _get_outlier (self, method):
		outrate = {}
		for c in self.df.columns :
			if self.df[c].dtype != 'object' :
				if method == 'z_score' :
					count =0
					z_score = (np.array (self.df[c]) - np.mean (self.df[c])) / np.std (self.df[c])
					for z in np.abs (z_score) :
						# print (z)
						if z > 3 :
							count += 1

					outrate[c] = str (round (count / self.df[c].count () * 100, 2)) + '%'

				elif method == 'isolationforest' :
					iso = IsolationForest (behaviour='new', random_state=22, contamination=0.06)
					preds = iso.fit_predict (self.df[c].values.reshape (-1, 1))
					data = pd.DataFrame ()
					data['cluster'] = preds
					if -1 in data['cluster'] :
						outrate[c] = '{:.2f}%'.format (data['cluster'].value_counts ()[-1] / len (self.df[c].values) * 100)
					else :
						outrate[c] = '0.0%'
						
				elif method == 'rbc' :
					rbc = EllipticEnvelope (contamination=0.1)
					preds = rbc.fit_predict (self.df[c].values.reshape (-1, 1))
					data = pd.DataFrame ()
					data['cluster'] = preds
					outrate[c] = '{:.2f}%'.format (data['cluster'].value_counts ()[-1] / len (self.df[c].values) * 100)

				elif method == 'ocsvm' :
					ocsvm = svm.OneClassSVM (nu=0.1, kernel="rbf", gamma=0.1)
					preds = ocsvm.fit_predict (self.df[c].values.reshape (-1, 1))
					data = pd.DataFrame ()
					data['cluster'] = preds
					outrate[c] = '{:.2f}%'.format (data['cluster'].value_counts ()[-1] / len (self.df[c].values) * 100)

		return (pd.Series (outrate, name='outlier'))

	def _get_uniques(self):
		return pd.Series(dict((c, self.df[c].nunique()) for c in self.df.columns), name='uniques')

	def _get_missing(self, counts):
		count = self.length - counts
		count.name = 'missing'
		perc = (count / self.length).apply(self._percent)
		perc.name = 'missing_perc'
		return pd.concat([count, perc], axis=1, sort=True)

	def _get_skew (self) :
		skew = self.df.skew ()
		skew.name = 'skew'
		return skew

	def _get_kurt (self) :
		kurt = self.df.kurt ()
		kurt.name = 'kurtosis'
		return kurt

	def _get_columns_info(self, stats):
		column_info = {}
		column_info[self.TYPE_CONSTANT] = stats['uniques'][stats['uniques'] == 1].index
		column_info[self.TYPE_BOOL] = stats['uniques'][stats['uniques'] == 2].index
		rest_columns = self.get_columns(self.df,
										self.EXCLUDE,
										column_info['constant'].union(column_info['bool']))
		column_info[self.TYPE_NUMERIC] = pd.Index([c for c in rest_columns
												if types.is_numeric_dtype(self.df[c])])
		rest_columns = self.get_columns(
			self.df[rest_columns], self.EXCLUDE, column_info['numeric'])
		column_info[self.TYPE_DATE] = pd.Index([c for c in rest_columns
												if types.is_datetime64_dtype(self.df[c])])
		rest_columns = self.get_columns(
			self.df[rest_columns], self.EXCLUDE, column_info['date'])
		unique_columns = stats['uniques'][rest_columns] == stats['counts'][rest_columns]
		column_info[self.TYPE_UNIQUE] = stats['uniques'][rest_columns][unique_columns].index
		column_info[self.TYPE_CATEGORICAL] = stats['uniques'][rest_columns][~unique_columns].index

		return column_info

	""" Column summaries """

	def _get_deviation_of_mean(self, series, multiplier=3):
		"""
		Returns count of values deviating of the mean, i.e. larger than `multiplier` * `std`.
		:type series:
		:param multiplier:
		:return:
		"""
		capped_series = np.minimum(
			series, series.mean() + multiplier * series.std())
		count = pd.value_counts(series != capped_series)
		count = count[True] if True in count else 0
		perc = self._percent(count / self.length)
		return count, perc

	def _get_median_absolute_deviation(self, series, multiplier=3):
		"""
		Returns count of values larger than `multiplier` * `mad`
		:type series:
		:param multiplier:
		:return (array):
		"""
		capped_series = np.minimum(
			series, series.median() + multiplier * series.mad())
		count = pd.value_counts(series != capped_series)
		count = count[True] if True in count else 0
		perc = self._percent(count / self.length)
		return count, perc

	def _get_top_correlations(self, column, threshold=0.65, top=3):
		column_corr = np.fabs(self.corr[column].drop(column)).sort_values(ascending=False,
																		inplace=False)
		top_corr = column_corr[(column_corr > threshold)][:top].index
		correlations = self.corr[column][top_corr].to_dict()
		return ', '.join('{}: {}'.format(col, self._percent(val)) for
						col, val in correlations.items())

	def _get_numeric_summary(self, column):
		series = self.df[column]

		if self.plot:
			try:
				series.hist()
			except ImportError:
				pass

		stats = OrderedDict()
		stats['mean'] = series.mean()
		stats['std'] = series.std()
		stats['variance'] = series.var()
		stats['min'] = series.min()
		stats['max'] = series.max()
		stats['mode'] = series.mode()[0]

		for x in np.array([0.05, 0.25, 0.5, 0.75, 0.95]):
			stats[self._percent(x)] = series.quantile(x)

		stats['iqr'] = stats['75%'] - stats['25%']
		stats['kurtosis'] = series.kurt()
		stats['skewness'] = series.skew()
		stats['sum'] = series.sum()
		stats['mad'] = series.mad()
		stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] else np.nan
		stats['zeros_num'] = self.length - np.count_nonzero(series)
		stats['zeros_perc'] = self._percent(stats['zeros_num'] / self.length)
		deviation_of_mean, deviation_of_mean_perc = self._get_deviation_of_mean(
			series)
		stats['deviating_of_mean'] = deviation_of_mean
		stats['deviating_of_mean_perc'] = deviation_of_mean_perc
		deviating_of_median, deviating_of_median_perc = self._get_median_absolute_deviation(
			series)
		stats['deviating_of_median'] = deviating_of_median
		stats['deviating_of_median_perc'] = deviating_of_median_perc
		stats['top_correlations'] = self._get_top_correlations(column)

		return pd.concat([pd.Series(stats, name=column),
						self.columns_stats[column]],
						sort=True)

	def _get_date_summary(self, column):
		series = self.df[column]
		stats = {'min': series.min(), 'max': series.max()}
		stats['range'] = stats['max'] - stats['min']
		return pd.concat([pd.Series(stats, name=column),
						self.columns_stats[column]],
						sort=True)

	def _get_categorical_summary(self, column):
		series = self.df[column]
		# Only run if at least 1 non-missing value
		value_counts = series.value_counts()
		stats = {
			'top': '{}: {}'.format(value_counts.index[0], value_counts.iloc[0]),
		}
		return pd.concat([pd.Series(stats, name=column),
						self.columns_stats[column]],
						sort=True)

	def _get_constant_summary(self, column):
		return 'This is a constant value: {}'.format(self.df[column][0])

	def _get_bool_summary(self, column):
		series = self.df[column]

		stats = {}
		for class_name, class_value in dict(series.value_counts()).items():
			stats['"{}" count'.format(class_name)] = '{}'.format(class_value)
			stats['"{}" perc'.format(class_name)] = '{}'.format(
				self._percent(class_value / self.length))

		return pd.concat([pd.Series(stats, name=column),
						self.columns_stats[column]],
						sort=True)

	def _get_unique_summary(self, column):
		return self.columns_stats[column]

	def _get_column_summary(self, column):
		column_type = self.columns_stats.loc['types'][column]
		if column_type == self.TYPE_NUMERIC:
			return self._get_numeric_summary(column)
		if column_type == self.TYPE_CATEGORICAL:
			return self._get_categorical_summary(column)
		if column_type == self.TYPE_BOOL:
			return self._get_bool_summary(column)
		if column_type == self.TYPE_UNIQUE:
			return self._get_unique_summary(column)
		if column_type == self.TYPE_DATE:
			return self._get_date_summary(column)
		if column_type == self.TYPE_CONSTANT:
			return self._get_constant_summary(column)

	@property
	def constants(self):
		return self.df.columns[self.columns_stats.loc['types'] == 'constant']

	@property
	def categoricals(self):
		return self.df.columns[self.columns_stats.loc['types'] == 'categorical']

	@property
	def numerics(self):
		return self.df.columns[self.columns_stats.loc['types'] == 'numeric']

	@property
	def uniques(self):
		return self.df.columns[self.columns_stats.loc['types'] == 'unique']

	@property
	def bools(self):
		return self.df.columns[self.columns_stats.loc['types'] == 'bool']

	@property
	def missing_frac(self):
		return self.columns_stats.loc['missing'].apply(lambda x: float(x) / self.length)

	def get_columns(self, df, usage, columns=None):
		"""
		Returns a `data_frame.columns`.
		:param df: dataframe to select columns from
		:param usage: should be a value from [ALL, INCLUDE, EXCLUDE].
							this value only makes sense if attr `columns` is also set.
							otherwise, should be used with default value ALL.
		:param columns: * if `usage` is all, this value is not used.
						* if `usage` is INCLUDE, the `df` is restricted to the intersection
						between `columns` and the `df.columns`
						* if usage is EXCLUDE, returns the `df.columns` excluding these `columns`
		:return: `data_frame` columns, excluding `target_column` and `id_column` if given.
				`data_frame` columns, including/excluding the `columns` depending on `usage`.
		"""
		columns_excluded = pd.Index([])
		columns_included = df.columns

		if usage == self.INCLUDE:
			try:
				columns_included = columns_included.intersection(pd.Index(columns))
			except TypeError:
				pass
		elif usage == self.EXCLUDE:
			try:
				columns_excluded = columns_excluded.union(pd.Index(columns))
			except TypeError:
				pass

		columns_included = columns_included.difference(columns_excluded)
		return columns_included.intersection(df.columns)

# def run (data_path, outlier='z_score', correlation='pearson'):
#     """Entry point for console_scripts
#     """
#     data = pd.read_csv (data_path, sep='::')
#     profiling = Profiling (data)

#     print (profiling.summary ())


# if __name__ == "__main__" :
#     run (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-1m/ratings.dat')    
# 	# run (data_path='/Users/apple/project/AI + elearning project/data_profiling/test/west.csv')