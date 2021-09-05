from autoctr.preprocessor.inputs import Input
import pandas as pd

def run (data_path, outlier='z_score', correlation='pearson'):
	"""Entry point for console_scripts
	"""

	profiling = Input (data_path=data_path, sep=",")

	profiling.preprocessing ()


if __name__ == "__main__" :
	# data = pd.read_csv('/Users/apple/Downloads/tf-experiment/criteo_sample.txt')

	run (data_path='/Users/apple/Downloads/tf-experiment/criteo_sample.txt')


	