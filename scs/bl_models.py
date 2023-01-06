import numpy as np
import pandas as pd

from lifelines.fitters.coxph_fitter import CoxPHFitter

import rpy2.rinterface as ri
from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr

from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

import rpy2.robjects as ro
from torch import sparse_coo_tensor

from auton_survival.estimators import SurvivalModel
from ascvd import compute_ten_year_score
#os.environ['R_USER'] = 'C:\ProgramData\Anaconda3\Lib\site-packages\rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution


class ASCVDRiskCalculator:
	def __init__(self, dataset: pd.DataFrame):
		self.dataset = dataset

	def _get_single_risk(self, row):
		if row.isna().any():
			return np.nan
		return compute_ten_year_score(row['Sex'] == 'Male',
                                    row['Ethnicity'] == 'Black Non-Hispanic',
                                    row['Baseline_Smoke_status'] == 'Current',
                                    True if self.dataset=='allhat' else row['Baseline_History_Hypertension'] == 'Yes', 
									True if self.dataset == 'bari2d' else row['Baseline_History_Diabetes'] == 'Yes',
                                    int(row['Baseline_Age']),
                                    int(row['Baseline_Seated_Systolic_Blood_Pressure']) if self.dataset == 'allhat' else int(row['Baseline_Standing_Systolic_Blood_Pressure']),
                                    int(row['Baseline_Total_Cholesterol']),
                                    int(row['Baseline_HDL_Cholesterol']))

	def get_risk(self, features):
		if self.dataset not in ['allhat', 'bari2d']:
			raise RuntimeError("Incorrect dataset specified. Only allhat, bari2d supported.")
		return features.apply(self._get_single_risk, axis=1).values


class BinaryInteractionModel:
	def __init__(self, nlambdas = 1):
		self.nlambda = nlambdas

	def fit(self, features, outcomes, intervention):
		# Get indices of outcomes that need to be removed
		# Removal criteria: if duration < 5 years and study has been censored
		discard_idxs = ((outcomes['event'] == 0) & (outcomes['time'] < 365.25*5.0))
		
		features = features[~discard_idxs]
		outcomes = outcomes[~discard_idxs]
		intervention = intervention[~discard_idxs]

		x = features.values
		a = 2*(intervention.reshape(-1,1) - 0.5)
		features = np.hstack([x, a*x])

		# Preprocess data for GLMNet

		# all t = 0 events need to removed as they dont work with glmnet - coxnet. refer to this link: 
		# https://stats.stackexchange.com/questions/359444/how-to-use-method-lasso-in-cox-model-using-glmnet
		features = features[outcomes['time'] != 0]
		outcomes = outcomes[outcomes['time'] != 0]

		base = importr('base')
		glmnet = importr('glmnet')
		survival = importr('survival')

		n = features.shape[0]
		penalty_factor = np.ones(features.shape[1])
		penalty_factor[:x.shape[1]] = 0

		np_cv_rules = default_converter + numpy2ri.converter # np_cv_rules is used wherever numpy -> R operations need to be performed

		with localconverter(np_cv_rules) as cv:
			rPen_factor = r.c(penalty_factor)
			rFeatures = r.matrix(features, nrow = n)
			label = np.where(outcomes['time'].values > 5, 1, 0)

		rOutcomes = ro.vectors.IntVector(label)

		with localconverter(np_cv_rules) as cv:
			misc_arg_dict = {'penalty.factor':rPen_factor}
			if self.nlambda == 1:
				misc_arg_dict['lambda'] = r.c(np.array([0.0]))
			#else:
			#	misc_arg_dict['lambda'] = r.c(np.linspace(1e-3, 0, self.nlambda))

			self.model = glmnet.glmnet(x = rFeatures, y = rOutcomes, 
								family = 'binomial', alpha = 1.0,
								nlambda = self.nlambda, intercept=True,
								**misc_arg_dict)		

		self.model_coef = np.array(base.as_matrix(self.model['beta']))
		#raise KeyboardInterrupt

	def predict(self, features, num_selected_features=1):
		features = features.values

		d = self.model_coef.shape[0]//2
		coefs = self.model_coef[d:, :]
		nonzero_coef_counts = np.count_nonzero(coefs, axis=0)
		preds = -features.dot(coefs)
		
		while (len(nonzero_coef_counts[nonzero_coef_counts == num_selected_features]) == 0):
			num_selected_features -= 1
			
			if num_selected_features == 0:
				print("Required number of selected features does not exist in generated sparse predictions")
				break

		selected_features_idx = np.argmin(np.abs(nonzero_coef_counts-num_selected_features))
		return preds[:, selected_features_idx], num_selected_features


class CoxVirtualTwins:

	def __init__(self, alpha=0, nlambdas = 1):
		
		self.alpha = alpha
		self.nlambda = nlambdas
		self.fitted = False
	
	
	def fit(self, features, outcomes, intervention):
		
		ft_names = list(features.columns)

		x_ft_names = ['X_'+ft_name for ft_name in ft_names]
		a1_ft_names = ['A(1)_'+ft_name for ft_name in ft_names]
		a0_ft_names = ['A(0)_'+ft_name for ft_name in ft_names]
		
		ft_names = x_ft_names + a1_ft_names #+ a0_ft_names

		treated_model = CoxPHFitter()
		treated_model.fit(features.loc[intervention==1].join(outcomes.loc[intervention==1]), 'time', 'event' )

		control_model = CoxPHFitter()
		control_model.fit(features.loc[intervention==0].join(outcomes.loc[intervention==0]), 'time', 'event' )

		pred_treated_outcomes = treated_model.predict_survival_function(features, times=5*365.25)  
		pred_control_outcomes = control_model.predict_survival_function(features, times=5*365.25)  

		pred_cate = pred_treated_outcomes - pred_control_outcomes


		#numpy2ri.activate()
		base = importr('base')
		glmnet = importr('glmnet')

		np_cv_rules = default_converter + numpy2ri.converter # np_cv_rules is used wherever numpy -> R operations need to be performed

		with localconverter(np_cv_rules) as cv:
			rFeatures = r.matrix(features.values, nrow = features.shape[0])

			rOutcomes = ro.vectors.IntVector(pred_cate.values[0] > 0)

			self.model = glmnet.glmnet(x = rFeatures, y = rOutcomes, 
								family = 'binomial', alpha = 1.0,
								nlambda = self.nlambda, intercept=True)		

		self.model_coef = np.array(base.as_matrix(self.model['beta']))
		self.fitted = True
	
		# n = features.shape[0]
		# outcomes['event'] = outcomes['event'].astype(int)

		# penalty_factor = np.ones(features.shape[1])
		# penalty_factor[:x.shape[1]] = 0

		# np_cv_rules = default_converter + numpy2ri.converter # np_cv_rules is used wherever numpy -> R operations need to be performed

		# with localconverter(np_cv_rules) as cv:
		# 	rPen_factor = r.c(penalty_factor)
		# 	rFeatures = r.matrix(features, nrow = n)
		# 	time, event = outcomes['time'].values, outcomes['event'].values

		# rOutcomes = survival.Surv(ro.vectors.FloatVector(time), ro.vectors.BoolVector(event))

		# with localconverter(np_cv_rules) as cv:
		# 	misc_arg_dict = {'penalty.factor':rPen_factor}
		# 	if self.nlambda == 1:
		# 		misc_arg_dict['lambda'] = r.c(np.array([0.0]))
		# 	#else:
		# 	#	misc_arg_dict['lambda'] = r.c(np.linspace(1e-3, 0, self.nlambda))

		# 	model = glmnet.glmnet(x = rFeatures, y = rOutcomes, 
		# 						family = 'cox', alpha = self.alpha,
		# 						nlambda = self.nlambda, intercept=True,
		# 						**misc_arg_dict)


	def predict(self, features, num_selected_features=1):
		
		if not self.fitted:
			raise Exception('Model not fitted')
		
		features = features.values

		coefs = self.model_coef
		nonzero_coef_counts = np.count_nonzero(coefs, axis=0)
		preds = features.dot(coefs)
		
		while (len(nonzero_coef_counts[nonzero_coef_counts == num_selected_features]) == 0):
			num_selected_features -= 1
			
			if num_selected_features == 0:
				print("Required number of selected features does not exist in generated sparse predictions")
				break

		selected_features_idx = np.argmin(np.abs(nonzero_coef_counts-num_selected_features))
		return preds[:, selected_features_idx], num_selected_features


class CoxInteractionModel:

	def __init__(self, alpha=0, nlambdas = 1, fitter='coxph'):
		
		self.alpha = alpha
		self.fitter = fitter
		self.nlambda = nlambdas
		self.fitted = False
	
	def fit(self, features, outcomes, intervention):
				
		ft_names = list(features.columns)

		x_ft_names = ['X_'+ft_name for ft_name in ft_names]
		a1_ft_names = ['A(1)_'+ft_name for ft_name in ft_names]
		a0_ft_names = ['A(0)_'+ft_name for ft_name in ft_names]
		
		ft_names = x_ft_names + a1_ft_names #+ a0_ft_names


		if self.fitter == 'coxph':
			x = features.values
			a = 2*(intervention.reshape(-1,1) - 0.5)
			features =  pd.DataFrame(data=np.hstack([x, a*x]), columns=ft_names, index=features.index)
			model = CoxPHFitter(l1_ratio=1., penalizer=self.alpha)
			model.fit(features.join(outcomes), 'time', 'event' )

		elif self.fitter == 'glmnet':
			x = features.values
			a = 2*(intervention.reshape(-1,1) - 0.5)
			#a = intervention.reshape(-1,1)
			features = np.hstack([x, a*x])
			# Preprocess data for GLMNet

			# all t = 0 events need to removed as they dont work with glmnet - coxnet. refer to this link: https://stats.stackexchange.com/questions/359444/how-to-use-method-lasso-in-cox-model-using-glmnet
			features = features[outcomes['time'] != 0]
			outcomes = outcomes[outcomes['time'] != 0]

			#numpy2ri.activate()
			base = importr('base')
			glmnet = importr('glmnet')
			survival = importr('survival')

			n = features.shape[0]
			outcomes['event'] = outcomes['event'].astype(int)

			penalty_factor = np.ones(features.shape[1])
			penalty_factor[:x.shape[1]] = 0

			np_cv_rules = default_converter + numpy2ri.converter # np_cv_rules is used wherever numpy -> R operations need to be performed

			with localconverter(np_cv_rules) as cv:
				rPen_factor = r.c(penalty_factor)
				rFeatures = r.matrix(features, nrow = n)
				time, event = outcomes['time'].values, outcomes['event'].values

			rOutcomes = survival.Surv(ro.vectors.FloatVector(time), ro.vectors.BoolVector(event))

			with localconverter(np_cv_rules) as cv:
				misc_arg_dict = {'penalty.factor':rPen_factor}
				if self.nlambda == 1:
					misc_arg_dict['lambda'] = r.c(np.array([0.0]))
				#else:
				#	misc_arg_dict['lambda'] = r.c(np.linspace(1e-3, 0, self.nlambda))

				model = glmnet.glmnet(x = rFeatures, y = rOutcomes, 
									family = 'cox', alpha = self.alpha,
									nlambda = self.nlambda, intercept=True,
									**misc_arg_dict)

		else:
			raise Exception(f'Unrecognized fitter: {self.fitter}')
		
		self.model = model
		self.model_coef = np.array(base.as_matrix(model['beta']))


		self.fitted = True

	def predict(self, features, num_selected_features=1):
		
		if not self.fitted:
			raise Exception('Model not fitted')
		
		if self.fitter == 'coxph':
			d = int(self.model.params_.values.shape[0]/2)

			return -features.values.dot(self.model.params_.values[d:]), self.model.params_.values[d:] 

		elif self.fitter == 'glmnet':
			features = features.values

			d = self.model_coef.shape[0]//2
			coefs = self.model_coef[d:, :]
			nonzero_coef_counts = np.count_nonzero(coefs, axis=0)
			preds = features.dot(coefs)
			
			while (len(nonzero_coef_counts[nonzero_coef_counts == num_selected_features]) == 0):
				num_selected_features -= 1
				
				if num_selected_features == 0:
					print("Required number of selected features does not exist in generated sparse predictions")
					break

			selected_features_idx = np.argmin(np.abs(nonzero_coef_counts-num_selected_features))
			return preds[:, selected_features_idx], num_selected_features