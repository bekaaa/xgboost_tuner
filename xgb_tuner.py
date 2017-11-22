#!/usr/bin/env python
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import log


class xgb_tuner :
	#*******************************************************************************
	def __init__(self, dtrain, dvalid, params,
		logging = False, log_file_index = -1, rounds=400, esrounds=50, seed=0, nfolds=3) :
		#-------------------
		self.__dtrain = dtrain
		self.__dvalid = dvalid
		self.__logging = logging
		self.__rounds = rounds
		self.__esrounds = esrounds
		self.__nfolds = nfolds
		self.__seed = seed
		self.params = params
		np.random.seed(self.__seed)
		#-----------------
		self.cvfolds = None
		#--------------------
		if self.__logging : self.__init_log(log_file_index)
	#*****************************************************************************
	def __call__(self, param_names, param_grid, all_results=[]) :
		'''
		Call function, takes "parameter names" and grids as input,
		log and analyze them. Then for each grid run a self.step_cv()
		checks ouput for best score.
		return : best grid.
		'''
		# some assertion to make sure of input.
		assert type(param_names) == list
		assert type(param_grid) == list
		assert len(param_names) == len(param_grid[0])
		for p in param_names : assert p in self.params.keys()
		#---------------------------------------
		# ------- emptying some variables ----------------
		best_results = {}
		best_results['dev_sc'] = -1
		#----------------------------------------
		if self.__logging : log.msg('**** Starting grid-call search *********')
		for grid in param_grid :
			#------ set starter log message -----------------
			msg = "CV with "
			for i,v in enumerate(grid):
				msg+= "%s = %g, " % (param_names[i], v)
			if self.__logging : log.msg(msg)
			#------------------------------------------------------------
			#-------- call step_cv function on each grid -----------------
			results = None
			if all_results != [] :

				assert type(all_results) == list
				assert type(all_results[0]) == tuple
				assert len(all_results[0]) == 2
				assert type(all_results[0][0]) in (tuple, list)
				assert type(all_results[0][1]) == dict

				for item in all_results :
					if grid == item[0] :
						results = item[1]
						break

			if results == None :
				results = self.__step_cv(param_names, grid)
				all_results.append( (grid, results) )
			#------------------------------------------------------------
			# ------------ another log message ---------------
			msg = 'dev score : %g, fold-train score : %g, fold-dev score : %g, rounds : %d,'\
					'step time : %.1f minutes'\
					%(results['dev_sc'],
					  results['f_train_sc'],
					  results['f_dev_sc'],
					  results['rounds'],
					  results['step_time'])
			if self.__logging : log.msg(msg)
			else : print msg
			# -------------------------------------------------------
			# ---- checking for best score and assigning some other bests ;) ---------
			if results['dev_sc'] > best_results['dev_sc'] :
				best_results = results.copy()
				best_results['grid'] = grid
			#-------------------------------------------------------------------------

		#-------- last log message for grid-call search ----------------------
		if self.__logging : log.msg('****** End of grid-call search *********')
		msg =   'best dev score : %g, '\
				'best fold-train score : %g, '\
				'best fold-dev score : %g, '\
				'best rounds = %d, '\
				% (best_results['dev_sc'],
				   best_results['f_train_sc'],
				   best_results['f_dev_sc'],
				   best_results['rounds'] )

		for i,v in enumerate(best_results['grid']) :
			msg += 'best %s = %g, ' % (param_names[i], v)
		msg += '\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n'
		if self.__logging : log.msg(msg)
		print msg
		#---------------------------------------------------------------------------
		return best_results, all_results
	#***********************************************************************************
	def __step_cv(self, param_names, grid ) :
		''' it calls the xgb.cv function,
			input : parameter names and a list of values as the same size of names
			returns a set of : fold-train-score, fold-test-score, dev-score, best-n-trees,
			and time in minutes.
			~ analyses only on grid a time.
		 '''
		# update the params dict with the new grid.
		for p,v in zip(param_names, grid) :
			self.params[p] = v

		t = np.datetime64('now')

		cv_results = xgb.cv(self.params, self.__dtrain,
						num_boost_round=self.__rounds,
						early_stopping_rounds=self.__esrounds,
						seed=self.__seed,
						nfold=self.__nfolds,
						stratified = True,
						metrics=('auc'),
						fpreproc=self.__fpreproc,
						verbose_eval = False,
						callbacks=[self.__GetBestCVFolds()]
						)
		#-----------------------
		# predict dev-set and get it's gini score
		assert self.cvfolds != None
		dev_preds = np.zeros(self.__dvalid.num_row())
		for fold in self.cvfolds :
			dev_preds += fold.bst.predict(self.__dvalid)
		dev_preds /= len(self.cvfolds)
		dev_score = self.gini(self.__dvalid.get_label(), dev_preds)
		#--------------------------------------
		results = {
			'f_train_sc': 2 * np.max(cv_results['train-auc-mean']) - 1,
			'f_dev_sc'  : 2 * np.max(cv_results['test-auc-mean']) - 1,
			'dev_sc'    : dev_score,
			'rounds'    : np.argmax(cv_results['test-auc-mean']),
			'step_time' : (np.datetime64('now') - t).astype('int') / 60.
		}
		return results
	#********************************************************************************
	def gini(self, labels, preds) :
		'''defining Gini's Score function.'''
		return roc_auc_score(labels, preds) * 2. - 1
	#**********************************************************************************
	def predict(self, dtest) :
		_ = self.__step_cv(['eta'], [self.params['eta']])
		assert self.cvfolds != None
		preds = np.zeros(dtest.num_row())
		for fold in self.cvfolds :
			preds += fold.bst.predict(dtest)
		preds /= len(self.cvfolds)
		return preds
	#**********************************************************************************
	def __init_log(self, index) :
		assert index >= 0
		if index == 0 : index = "test"
		log.LOG_PATH = './logs/'
		try :
			_ = log.close()
		except :
			pass
		log.init('tuning_params-' + str(index) + '.log')
		log.msg('------------------initialized-----------------')
	#************************************************************************************
	def __fpreproc(self, dtrain_, dtest_, param_):
		''' passes to xgb.cv as a preprocessing function '''
		label = dtrain_.get_label()
		ratio = float(np.sum(label == 0)) / np.sum(label == 1)
		param_['scale_pos_weight'] = ratio
		return (dtrain_, dtest_, param_)
	#*************************************************************************************
	def __GetBestCVFolds(self) :
		'''passes to xgb.cv as a callback'''
		state = {}
		def init(env) :
			state['best_score'] = -1
			self.cvfolds = None
		def callback(env) :
			if env.iteration == env.begin_iteration :
				init(env)
			current_score = env.evaluation_result_list[1][1]
			if current_score > state['best_score'] :
				state['best_score'] = current_score
				self.cvfolds = env.cvfolds
		callback.before_iteration = False
		return callback
	#****************************************************************************************
