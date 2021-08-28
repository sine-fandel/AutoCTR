from autoctr.preprocessor.inputs import Input
import pandas as pd

def run (data_path, outlier='z_score', correlation='pearson'):
	"""Entry point for console_scripts
	"""
	data = pd.read_csv (data_path, sep='::')
	profiling = Input (data)

	print (profiling.preprocessing ())


if __name__ == "__main__" :
	run (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-1m/ratings.dat')    
	# run (data_path='/Users/apple/project/AI + elearning project/data_profiling/test/west.csv')