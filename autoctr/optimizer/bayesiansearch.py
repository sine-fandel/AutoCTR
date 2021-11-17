import warnings
from scipy.stats.stats import mode
import torch
import time

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from .core.util import UtilityFunction, acq_max, ensure_rng
from .core.logger import _get_default_logger
from .core.event import Events, DEFAULT_EVENTS
from .core.target_space import TargetSpace
from ..models import *

from alive_progress import alive_bar
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
	def __init__(self):
		self._queue = []

	@property
	def empty(self):
		return len(self) == 0

	def __len__(self):
		return len(self._queue)

	def __next__(self):
		if self.empty:
			raise StopIteration("Queue is empty, no more objects to retrieve.")
		obj = self._queue[0]
		self._queue = self._queue[1:]
		print (obj)
		return obj

	def next(self):
		return self.__next__()

	def add(self, obj):
		"""Add object to end of queue."""
		self._queue.append(obj)


class Observable(object):
	"""

	Inspired/Taken from
		https://www.protechtraining.com/blog/post/879#simple-observer
	"""
	def __init__(self, events):
		# maps event names to subscribers
		# str -> dict
		self._events = {event: dict() for event in events}

	def get_subscribers(self, event):
		return self._events[event]

	def subscribe(self, event, subscriber, callback=None):
		if callback is None:
			callback = getattr(subscriber, 'update')
		self.get_subscribers(event)[subscriber] = callback

	def unsubscribe(self, event, subscriber):
		del self.get_subscribers(event)[subscriber]

	def dispatch(self, event):
		for _, callback in self.get_subscribers(event).items():
			callback(event, self)


class BayesianOptimization (Observable) :
	"""Bayesian Optimization

	:param params: the hyperparameters that needed to be optimizated
	:param pbounds: If the value is an integer, it is used as the seed for creating a
        			numpy.random.RandomState. Otherwise the random state provieded it is used.
        			When set to None, an unseeded random state is generated.
	:param inputs: The needed data for building model.
	:param models: The model that needed to be optimizated.
	:param verbose: The level of verbosity.
	:param bounds_transformer: If provided, the transformation is applied to the bounds.
	"""
	def __init__ (self, model_name, epochs, save_path, max_evals, inputs, task="binary", device="cpu", 
				random_state=None, verbose=2, bounds_transformer=None, target="label", metrics="0",
				batch_size=256):
		self.model_name = model_name
		self.max_evals = max_evals
		self.target = target
		self.metrics = metrics
		self.device = device
		self.task = task
		self.save_path = save_path
		self.batch_size = batch_size
		if self.model_name == "DeepFM" :
			self.model = DeepFM
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "xDeepFM" :
			self.model = xDeepFM
			self.pbounds = {
							# "cin_layer_size": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
							# "cin_layer_size1": (16, 2048),
							# "cin_layer_size2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"l2_reg_cin": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "IFM" :
			self.model = IFM
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "DIFM" :
			self.model = DIFM
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "NFM" :
			self.model = NFM
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							"dnn_dropout": (0.0, 1.0),
							# "bi_dropout": (0.0, 1.0),
						}
		elif self.model_name == "ONN" :
			self.model = ONN
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "PNN" :
			self.model = PNN
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "DCN" :
			self.model = DCN
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_cross": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "DCNMix" :
			self.model = DCNMix
			self.pbounds = {
							"dnn_hidden_units1": (16, 2048),
							"dnn_hidden_units2": (16, 2048),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_cross": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.01, 0.1),
							# "low_rank": (4, 128),
							# "num_experts": (4, 32),
						}
		elif self.model_name == "AFN" :
			self.model = AFN
			self.pbounds = {
							# "ltl_hidden_size": (4, 2048),
							# "afn_dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
							"l2_reg_linear": (0.00001, 1.0),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}
		elif self.model_name == "AutoInt" :
			self.model = AutoInt
			self.pbounds = {
							"att_layer_num": [1, 10],
							# "att_head_num": ,
							# "dnn_hidden_units": (np.arange (16, 2048, 16), np.arange (16, 2048, 16)),
							"l2_reg_embedding": (0.00001, 1.0),
							"l2_reg_dnn": (0.01, 0.1),
							"init_std": (0.00001, 1.0),
							# "dnn_dropout": (0.0, 1.0),
						}

		self._random_state = ensure_rng (random_state)
		self.inputs = inputs

		# Data structure containing the function to be optimized, the bounds of
		# its domain, and a record of the evaluations we have done so far
		self._space = TargetSpace (self.target_fun, self.pbounds, random_state)
		# queue
		self._queue = Queue ()

		# Internal GP regressor
		self._gp = GaussianProcessRegressor(
			kernel=Matern(nu=2.5),
			alpha=1e-6,
			normalize_y=True,
			n_restarts_optimizer=5,
			random_state=self._random_state,
		)

		self._verbose = verbose
		self._bounds_transformer = bounds_transformer
		if self._bounds_transformer:
			self._bounds_transformer.initialize(self._space)

		self.epochs = epochs

		super (BayesianOptimization, self).__init__ (events=DEFAULT_EVENTS)


	def target_fun (self, **params_dict) :
		"""Bulid the target function for bayesian optimization

		:param model: the model that needed to be optimizated
		:param params_dict: the hyperparameters that needed to be optimizated
		"""
		model = self.model (self.inputs[4], self.inputs[5], task=self.task, device=self.device, **params_dict)

		if self.metrics == 1 :
			model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"], )
		elif self.metrics == 0 :
			model.compile ("adam", "mse", metrics=["mse"], )
		
		model.fit (self.inputs[2], self.inputs[0][self.target].values, batch_size=self.batch_size, epochs=self.epochs, verbose=2, earl_stop_patience=0, if_tune=1)
		pred_ans = model.predict (self.inputs[3], 256)

		if self.metrics == 1 :
			res = roc_auc_score (self.inputs[1][self.target].values, pred_ans)
		elif self.metrics == 0 :
			res = -mean_squared_error (self.inputs[1][self.target].values, pred_ans)

		return res, model

	@property
	def space(self):
		return self._space

	@property
	def max(self):
		return self._space.max()

	@property
	def res(self):
		return self._space.res()

	def register(self, params, target):
		"""Expect observation with known target"""
		self._space.register(params, target)
		self.dispatch(Events.OPTIMIZATION_STEP)

	def probe(self, params, lazy=True):
		"""Probe target of x"""
		if lazy:
			self._queue.add(params)
		else:
			self._space.probe(params)
			self.dispatch(Events.OPTIMIZATION_STEP)

	def suggest(self, utility_function):
		"""Most promissing point to probe next"""
		if len(self._space) == 0:
			return self._space.array_to_params(self._space.random_sample())

		# Sklearn's GP throws a large number of warnings at times, but
		# we don't really need to see them here.
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self._gp.fit(self._space.params, self._space.target)

		# Finding argmax of the acquisition function.
		suggestion = acq_max (
			ac=utility_function.utility,
			gp=self._gp,
			y_max=self._space.target.max(),
			bounds=self._space.bounds,
			random_state=self._random_state
		)

		return self._space.array_to_params (suggestion)

	def _prime_queue(self, init_points):
		"""Make sure there's something in the queue at the very beginning."""
		if self._queue.empty and self._space.empty:
			init_points = max (init_points, 1)

		for _ in range(init_points):
			self._queue.add (self._space.random_sample())

	def _prime_subscriptions (self) :
		if not any ([len (subs) for subs in self._events.values()]):
			_logger = _get_default_logger(self._verbose)
			self.subscribe(Events.OPTIMIZATION_START, _logger)
			self.subscribe(Events.OPTIMIZATION_STEP, _logger)
			self.subscribe(Events.OPTIMIZATION_END, _logger)

	def maximize (self, init_points=1, acq='ucb', kappa=2.576, kappa_decay=1, kappa_decay_delay=0, xi=0.0, **gp_params) :
		"""Mazimize your function"""
		# self._prime_subscriptions ()
		# self.dispatch (Events.OPTIMIZATION_START)
		# self._prime_queue(init_points)
		self.set_gp_params(**gp_params)
		n_iter = self.max_evals
		util = UtilityFunction(kind=acq,
							kappa=kappa,
							xi=xi,
							kappa_decay=kappa_decay,
							kappa_decay_delay=kappa_decay_delay)
		iteration = 0
		
		if self.metrics == 1 :
			best_score = 0
		elif self.metrics == 0 :
			best_score = 9999
			
		best_param = {}
		best_round = 0
		with alive_bar (self.max_evals) as bar :
			while not self._queue.empty or iteration < n_iter:
				try:
					x_probe = next(self._queue)
				except StopIteration:
					util.update_params()
					x_probe = self.suggest(util)
					iteration += 1

				self.probe(x_probe, lazy=False)

				if self._bounds_transformer:
					self.set_bounds(
						self._bounds_transformer.transform(self._space))

				if self.metrics == 1 and best_score < self.res[-1]['target'] :
					best_score = self.res[-1]['target']
					best_param = self.res[-1]['params']
					best_round = iteration
				elif self.metrics == 0 and best_score > -self.res[-1]['target'] :
					best_score = -self.res[-1]['target']
					best_param = self.res[-1]['params']
					best_round = iteration

				bar ()
				bar.text ("#%d  score: %.4f		Best score currently: %.4f" % (iteration, round (self.res[-1]['target'], 4), round (best_score, 4)))

		print ("Best Score: %.4f in %d" % (round (best_score, 4), round (best_round, 4)))
		print ("Saving best model ...")
		if "dnn_hidden_units1" in best_param :
				u1 = best_param.pop ('dnn_hidden_units1')
				u2 = best_param.pop ('dnn_hidden_units2')
				best_param['dnn_hidden_units'] = (round (u1), round (u2))

		res, best_model = self.target_fun (**best_param)
		torch.save (best_model, self.save_path + self.model.__name__ + "_" + str (1) +  ".pt")
		print ("The best model was saved in ", self.save_path)

		# #################################
		# ##loading and testing the model##
		# #################################
		# test_model = DeepFM (self.inputs[4], self.inputs[5], task=self.task, device=self.device, **best_param)
		# test_model.load_state_dict (torch.load ("/Users/apple/AutoCTR project/AutoCTR/PKL/bayesian/DeepFM.pth"))
		# res = test_model.predict (self.inputs[3], 256)
		# print (res)
		# test_model = torch.load ("/Users/apple/AutoCTR project/AutoCTR/PKL/bayesian/DeepFM_1.pt")
		# print (test_model.predict (self.inputs[3], 256))

		return best_param
		

	def set_bounds(self, new_bounds):
		"""
		A method that allows changing the lower and upper searching bounds

		Parameters
		----------
		new_bounds : dict
			A dictionary with the parameter name and its new bounds
		"""
		self._space.set_bounds(new_bounds)

	def set_gp_params(self, **params):
		self._gp.set_params(**params)

