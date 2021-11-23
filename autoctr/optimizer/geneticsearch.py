import numpy as np
from sklearn.model_selection import KFold
from ..models import *
from sklearn.metrics import roc_auc_score, mean_squared_error

class GeneticHyperopt:
	def __init__(self, model_name, linear_feature_columns, dnn_feature_columns, task, device, metrics,
				train_model_input, train_y, test_model_input, test_y, maximize, epochs=10, batch_size=256,
				pop_size=40, num_gen=5, elite_percent=0.1, competitiveness=0.4, mutation_prob=0.2):
		self.learner = model_name
		self.task = task
		self.device = device
		self.batch_size = batch_size
		self.metrics = metrics
		self.dnn_feature_columns = dnn_feature_columns
		self.linear_feature_columns = linear_feature_columns
		self.train_model_input = train_model_input
		self.train_y = train_y
		self.test_model_input = test_model_input
		self.test_y = test_y
		self.epochs = epochs
		# self.num_folds = num_folds
		# self.indices = [(train_ind, val_ind) for (train_ind, val_ind) in KFold(n_splits=self.num_folds).split(X, y)]
		self.maximize = maximize
		self.pop_size = pop_size
		self.num_gen = num_gen
		self.num_elites = int(pop_size * elite_percent)  # check if even
		self.tournament_size = int(pop_size * competitiveness)
		self.mutation_prob = mutation_prob
		self.params = []

	def add_param(self, param):
		self.params.append(param)
		return self

	def _initialize_population(self):
		return [[param.sample() for param in self.params] for _ in range(self.pop_size)]

	def _to_param_dict(self, ind_params):
		param_dict = {}
		for i in range(len(self.params)):
			param_dict[self.params[i].name] = ind_params[i]
		return param_dict

	def _evaluate_individual(self, ind_params):
		param_dict = self._to_param_dict(ind_params)
		print (param_dict)
		if self.learner == "DeepFM" :
			learner_obj = DeepFM (linear_feature_columns=self.linear_feature_columns, dnn_feature_columns=self.dnn_feature_columns,
							task=self.task, device=self.device, **param_dict)

		score = 0
		if self.metrics == 1 :
			learner_obj.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"], )
		elif self.metrics == 0 :
			learner_obj.compile ("adam", "mse", metrics=["mse"], )
		
		learner_obj.fit (self.train_model_input, self.train_y, epochs=self.epochs, if_tune=1, batch_size=self.batch_size, verbose=2)
		pred_ans = learner_obj.predict (self.test_model_input, self.batch_size)

		if self.metrics == 1 :
			score = roc_auc_score (self.test_y, pred_ans)
		elif self.metrics == 0 :
			score = mean_squared_error (self.test_y, pred_ans)

		return score

	def _evaluate_population(self):
		return [self._evaluate_individual(ind) for ind in self.population]

	def _select_parents(self):
		parents = [None] * (self.pop_size - self.num_elites)
		for i in range(self.pop_size - self.num_elites):
			candidates = np.random.choice(np.arange(self.pop_size), self.tournament_size, replace=False)
			parents[i] = self.population[min(candidates)][:]
		return parents

	def _generate_children(self, parents):
		children = [None] * len(parents)
		for i in range(int(len(parents) / 2)):
			child1 = parents[2 * i]
			child2 = parents[2 * i + 1]
			for j in range(len(self.params)):
				if np.random.rand() > 0.5:
					temp = child1[j]
					child1[j] = child2[j]
					child2[j] = temp
			children[2 * i] = child1
			children[2 * i + 1] = child2
		return children

	def _mutate(self, children):
		for i in range(len(children)):
			child = children[i][:]
			for j in range(len(self.params)):
				if np.random.rand() < self.mutation_prob:
					child[j] = self.params[j].mutate(child[j])
			children[i] = child
		return children

	def evolve(self):
		self.population = self._initialize_population()
		for i in range(self.num_gen):
			# rank the population
			print("Generation", i)
			print("Calculating fitness...")
			fitness = self._evaluate_population()
			if self.maximize :
				fitness = [f * (-1) for f in fitness]

			rank = np.argsort(fitness)
			self.population = [self.population[r] for r in rank]

			print("Best individual:", self._to_param_dict(self.population[0]))
			print("Best score:", min(fitness))
			print("Population mean:", np.mean(fitness))

			# generate new generation
			print("Generating children...")
			parents = self._select_parents()
			children = self._generate_children(parents)
			children = self._mutate(children)
			self.population[self.num_elites:] = children

			print("---")

		return self._to_param_dict(self.population[0]), min(fitness)
