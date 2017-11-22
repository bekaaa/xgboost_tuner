#!/usr/bin/env python
import numpy as np
#import xgboost as xgb
from xgb_tuner import xgb_tuner
#from sklearn.model_selection import train_test_split
import log
'''
Using the xgb_tuner Class. I will tune the important parameters in xgboost.
This should work fine for most of the problems.
You could change the values if you want.
I choosed values which makes sense, and tried my best to cover all possibilities.

Tuning goes on 4 stages :
1 - Handling Imbalanced data.
	** max_delta_step and scale_pos_weight
	** scale_pos_weight is already assigned for each fold in xgb_tuner() with the ratio of
		negative examples to the positive examples.
	** max_delta_step not important to tune, but it should be a value from 0 to 10.
	** see this https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md#handle-imbalanced-dataset

2 - Controling model complexity :
	** max_depth, min_child_weight, and gamma
	** and those i will tune them for a large scale of values.

3 - control model's robusting to noise.
	** subsample and colsample_bytree

4 - Regularization terms.
	** alpha and lambda

5 - learning rate and number of rounds.
	** eta and num_rounds
'''
class parametizer :
	def __init__(self, train_file, preproc = None, test_file=None, dev_size=3000,
		log_file_index=-1) :

		self.seed = 0
		#------- prepare log file -----------------#
		assert log_file_index >= 0
		self.init_log(log_file_index)
		log.msg('****************************************')
		log.msg('*** log file initialized ********')
		#----------------------------------------------
		# ------- preparing data ----------------------- #
		log.msg('* preparing data')
		try :
			self.train = np.load(train_file)
			self.test = np.load(test_file) if test_file else np.array([])
		except :
			raise ValueError('Wrong train/test file input')
		#self.labels = self.train[:,0]
		#self.train = self.train[:,1:]
		if preproc : self.dtrain, self.dvalid, self.dtest = preproc(self.train, self.test)
		del self.train, self.test

		log.msg('data is ready to use.')
		# ------------ data is ready ----------------- #
		# -------------------------------------------- #
		# ------ initializa the parameters ----------- #
		self.params = {
			'max_delta_step' : 0,
			'scale_pos_weight' : 1, # calculated for each fold. #neg / #pos

			'max_depth' : 6,
			'min_child_weight' : 1,
			'gamma' : 0,

			'subsample' : 1,
			'colsample_bytree' : 1,

			'reg_alpha' : 0,
			'reg_lambda' : 1,

			'eta' : 0.01,

			'objective' : "binary:logistic",
			'eval_metric' : 'auc',
			'n_jobs' : -1,
			'random_seed' : self.seed
		}
		self.rounds = 800
		self.esrounds = 50 # early stop rounds.
		# ------------------------------------------ #
		# ---- initializing xgb_tuner object ------ #
		self.tuner = xgb_tuner(self.dtrain, self.dvalid, self.params,
			logging=True, log_file_index=log_file_index,
			rounds=self.rounds, esrounds=self.esrounds)
		#----------------------------------------------------
		del self.dtrain, self.dvalid
		log.msg('class is ready.')
	##############################################################
	def pred_dtest(self):
		log.msg('started predicting')
		return self.tuner.predict(self.dtest)
	###############################################################
	def doall(self) :
		'''
		tune all parameters.
		'''
		#terms, bst = self.tune_data_imbalancing()
		#terms, bst = self.tune_model_complexity()
		#for t,b in zip(terms, bst) :
		#	print "*-* best %s = %g " % (t,b)
		#	log.msg("*-* best %s = %g " % (t,b) )
		#	self.params[t] = b
		#print '\n'
		#------------------------
		terms, bst = self.tune_gamma()
		for t,b in zip(terms, bst) :
			print "*-* best %s = %g " % (t,b)
			log.msg("*-* best %s = %g " % (t,b) )
			self.params[t] = b
		print '\n'
		#---------------
		terms, bst = self.tune_model_robustness()
		for t,b in zip(terms, bst) :
			print "*-* best %s = %g " % (t,b)
			log.msg("*-* best %s = %g " % (t,b) )
			self.params[t] = b
		print '\n'
		#--------------------
		terms, bst = self.tune_regulr_terms('l2')
		for t,b in zip(terms, bst) :
			print "*-* best %s = %g " % (t,b)
			log.msg("*-* best %s = %g " % (t,b) )
			self.params[t] = b
		print '\n'
		#----------------------
		terms, bst = self.tune_regulr_terms('l1')
		for t,b in zip(terms, bst) :
			print "*-* best %s = %g " % (t,b)
			log.msg("*-* best %s = %g " % (t,b) )
			self.params[t] = b
		print '\n'
		#---------------------
		terms, bst = self.tune_eta()
		for t,b in zip(terms, bst) :
			print "*-* best %s = %g " % (t,b)
			log.msg("*-* best %s = %g " % (t,b) )
			self.params[t] = b
		print '\n'
	##################################################################
	def init_log(self, index) :
		if index == 0 : index = "test"
		log.LOG_PATH = './logs/'
		try :
			_ = log.close()
		except :
			pass
		log.init('tuning_params-' + str(index) + '.log')
	##############################################################
	def tune_model_complexity(self):
		'''
		This function can search for the suitable int value for two parameters, in range [0-20]
		'''
		terms = ['max_depth', 'min_child_weight']
		all_results = []
		#---------------------------------------------5
		pa1 = np.arange(1,10,2)
		pa2 = np.arange(1,10,2)
		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#-----------------------------------------------3
		pa1 = [ bst[0] - 1, bst[0], bst[0] + 1 ]
		pa2 = [ bst[1] - 1, bst[1], bst[1] + 1 ]
		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#--------------------------------------------------[Return]
		if bst[0] < 10 and bst[1] < 10 :
			return terms, bst
		#-------------------------------------------------------5/[1]
		# in case of either of them is 10.
		if bst[0] == 10 :
			pa1 = np.arange(11,20,2) # 5
		else :
			pa1 = [bst[0]]

		if bst[1] == 10 : # 5
			pa2 = np.arange(11,20,2)
		else :
			pa2 = [bst[1]]

		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#----------------------------------------------3/1
		if bst[0] >= 10 :
			pa1 = [ bst[0] - 1, bst[0], bst[0] + 1 ]
		else :#3
			pa1 = [bst[0]]

		if bst[1] >= 10 :
			pa2 = [ bst[1] - 1, bst[1], bst[1] + 1 ]
		else :#3
			pa2 = [bst[1]]

		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#----------------------------------------------
		return terms, bst
	##############################################################
	def tune_gamma(self):
		'''
		This function can search for the suitable int value for one parameters, in range [0-20]
		'''
		terms = ['gamma']
		all_results = []
		#---------------------------------------5
		grids = [ [i] for i in np.arange(1,10,2) ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#-----------------------------------------------3
		grids = [ [bst[0] - 1], [bst[0]], [bst[0] + 1] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#--------------------------------------------------
		if bst[0] == 10 :
			#-------------------------------------------------------5
			# best value is 10.
			grids = [ [i] for i in np.arange(11,20,2) ] # 5
			best_results, all_results = self.tuner(terms, grids, all_results)
			bst = best_results['grid']
			#-----------------------------------------------3
			grids = [ [bst[0] - 1], [bst[0]], [bst[0] + 1] ]
			best_results, all_results = self.tuner(terms, grids, all_results)
			bst = best_results['grid']
		#----------------------------------------------[5-7]
		if bst[0] == grids[0][0] : #5
			grids = [ [i] for i in np.arange(bst[0], bst[0]+1, .2) ]
		elif bst[0] == grids[1][0] : #7
			grids = [ [i] for i in np.arange(bst[0]-.6, bst[0]+.8, .2) ]
		else : #5
			grids = [ [i] for i in np.arange(bst[0]-.8, bst[0]+.1, .2) ]

		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#-------------------------------------------------2-3
		if bst[0] == 0 :
			grids = [[bst[0]], [bst[0]+.1]]
		else :
			grids = [ [bst[0]-.1], [bst[0]], [bst[0]+.1] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#--------------------------------------------
		return terms, bst
	#############################################################
	def tune_model_robustness(self) :
		'''
		tune two parameters to the best value in (0-1] to the second decimal point.
		values > 0.01,0.02,...0.1,.....0.98,.099,1
		'''
		terms = ['subsample', 'colsample_bytree']
		all_results = []
		#---------------------------------------5
		grids = [ (p1,p2) for p1 in np.arange(.1,1,.2) for p2 in np.arange(.1,1,.2) ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#----------------------------------------3
		pa1 = [ bst[0]-.1, bst[0], bst[0]+.1 ]
		pa2 = [ bst[1]-.1, bst[1], bst[1]+.1 ]
		# neither subsample nor colsample_bytree can be zero.
		if pa1[0] == 0 : pa1[0] = 0.01
		if pa2[0] == 0 : pa2[0] = 0.01
		grids = [(p1,p2) for p1 in pa1 for p2 in pa2]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#----------------------------------------------[5-7]
		if bst[0] == pa1[0] : #5
			if bst[0] == 0.01 :
				pa1 = np.arange(0.02, .1, .02)
			else :
				pa1 = np.arange(bst[0], bst[0]+.1, .02)
		elif bst[0] == pa1[1] : #7
			pa1 = np.arange(bst[0]-.06, bst[0]+.07, .02)
		else : #5
			pa1 = np.arange(bst[0]-.08, bst[0]+.01, .02)

		if bst[1] == pa2[0] : #5
			if bst[1] == 0.01 :
				pa2 = np.arange(0.02, .1, .02)
			else :
				pa2 = np.arange(bst[1], bst[1]+.1, .02)
		elif bst[1] == pa2[1] : #7
			pa2 = np.arange(bst[1]-.06, bst[1]+.07, .02)
		else : #5
			pa2 = np.arange(bst[1]-.08, bst[1]+.01, .02)

		grids = [ (p1,p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#----------------------------------------------3
		pa1 = [ bst[0]-0.01, bst[0], bst[0]+0.01 ]
		pa2 = [ bst[1]-0.01, bst[1], bst[1]+0.01 ]
		grids = [ (p1,p2) for p1 in pa1 for p2 in pa2 ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#------------------------------------------------
		return terms, bst
	###############################################################
	def tune_regulr_terms(self, type='l2'):
		best_results = []
		assert type in ['l1','l2']
		if type == 'l1' : terms = ['reg_alpha']
		elif type == 'l2' : terms = ['reg_lambda']
		#------------------------------------------5
		grids = [ [i] for i in np.arange(1,10,2) ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#-------------------------------------------3
		grids = [ [i] for i in [bst[0]-1, bst[0], bst[0]+1] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#------------------------------------------4-7
		if bst[0] == grids[0][0] :
			grids = [ [i] for i in np.arange(bst[0], bst[0]+.7, .2) ]
		elif bst[0] == grids[1][0] :
			grids = [ [i] for i in np.arange(bst[0]-.6, bst[0]+.7, .2) ]
		else :
			grids = [ [i] for i in np.arange(bst[0]-.6, bst[0], .2) ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#-------------------------------------------------3
		grids = [ [i] for i in [bst[0]-.1, bst[0], bst[0]+.1 ] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		#------------------------------------------------
		return terms, bst
	#################################################################
	def tune_regulr_terms_smaller_values(self, type='l2'):
		all_results = []
		assert type in ['l1','l2']
		if type == 'l1' : terms = ['reg_alpha']
		elif type == 'l2' : terms = ['reg_lambda']
		#------------------------------------------5
		grids = [ [i] for i in [0, 0.0001, 0.001, 0.01, 0.1] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		return terms, bst
	##############################################################
	def tune_eta(self):
		all_results = []
		terms = ['eta']
		grids = [ [i] for i in [.3, .2, .1, .01, .001] ]
		best_results, all_results = self.tuner(terms, grids, all_results)
		bst = best_results['grid']
		return terms, bst
	################################################################
