from scs.datasets_retro import load_allhat_antihypertensive_dataset, load_peace_dataset, load_accord_dataset, load_bari_2d_dataset, load_allhat_lipid_dataset
from sklearn.model_selection import train_test_split

import numpy as np

import sys
sys.path.append('/zfsauton2/home/vsanil/projects/aihc/cardiovascular_clinical_trials/sparse_cox_subgroup')

from auton_survival.preprocessing import Preprocessor
from auton_survival.metrics import treatment_effect

def load_data(dataset, train_size, data_dir,  random_state, get_original=False, **kwargs):
	if dataset == 'allhat1':
		outcomes, features, interventions, cat_feats, num_feats = load_allhat_antihypertensive_dataset(location=data_dir, outcome='CCVD')
		intervention = 'Lisinopril'
		outcomes = outcomes[interventions!='Chlorthalidone']	
		features = features[interventions!='Chlorthalidone']
		interventions = interventions[interventions!='Chlorthalidone']
	if dataset == 'allhat2':
		outcomes, features, interventions, cat_feats, num_feats = load_allhat_antihypertensive_dataset(location=data_dir, outcome='CCVD')
		intervention = 'Chlorthalidone'
		interventions[interventions!='Chlorthalidone'] = 'Amlodipine/Lisinopril'
	if dataset == 'peace':
		outcomes, features, interventions, cat_feats, num_feats = load_peace_dataset(location=data_dir, outcome='PRIMARY')
		intervention = 'Trandolapril'
	if dataset == 'bari2d_card':
		outcomes, features, interventions, cat_feats, num_feats = load_bari_2d_dataset(location=data_dir, intervention='cardtrt')
		intervention = 'Medical Therapy'
	if dataset == 'bari2d_diab':
		outcomes, features, interventions, cat_feats, num_feats = load_bari_2d_dataset(location=data_dir, intervention='diabtrt')
		intervention = 'Insulin Sensitizing'
	if dataset == 'accord_glycemia':
		outcomes, features, interventions, cat_feats, num_feats = load_accord_dataset(location=data_dir, intervention='Glycemia')
		intervention = 'Intensive Glycemia'
	# # One-Hot Encode and Standard Normalized the features.
	if not get_original:
		features = Preprocessor(cat_feat_strat='ignore').fit_transform(features[cat_feats+num_feats], cat_feats=cat_feats, num_feats=num_feats)

	if train_size == 1.0:    
		return outcomes, None, features, None, interventions, None
	else:
		stratify_inds = (interventions==intervention).values + 2*outcomes.event.values
		shuffle = True

		outcomes_train, outcomes_test, features_train, features_test, interventions_train, interventions_test = train_test_split(outcomes, 
																																features, 
																																interventions,
																																train_size=train_size,  
																																shuffle=shuffle, random_state=random_state, 
																																stratify=stratify_inds)

	return outcomes_train, outcomes_test, features_train, features_test, interventions_train, interventions_test, intervention





def get_predictions(model, features, d, selected_features):

	from scs.bl_models import CoxInteractionModel, BinaryInteractionModel, CoxVirtualTwins, ASCVDRiskCalculator
	from scs import models, new_model_gl

	if isinstance(model, (BinaryInteractionModel, CoxInteractionModel)):
		if d: return -model.predict(features, selected_features)[0]
		else: return model.predict(features, selected_features)[0] 

	elif isinstance(model, CoxVirtualTwins):
		if d: return model.predict(features, selected_features)[0]
		else: return -model.predict(features, selected_features)[0] 

	elif isinstance(model, ASCVDRiskCalculator):
		return model.get_risk(features)	

	elif isinstance(model[list(model.keys())[0]], models.SparseCoxSubgroup):
		return model[selected_features].predict_proba(features.values)[:, d]

	elif isinstance(model[list(model.keys())[0]], new_model_gl.SparseCoxSubgroup):
		return model[selected_features].predict_proba(features.values)[:, d]

	elif isinstance(model[list(model.keys())[0]], new_model_gl.SparseCoxSubgroup):
		return model[selected_features].predict_proba(features.values)[:, d]

	else:
		return model[selected_features].predict_proba(features.values)[:, d]



def aggregate_results(model, features, outcomes, interventions, intervention, d, selected_features, ranks, n_bootstrap=None):

	rmsts, hrs, risks = [], [], []
	
	preds = get_predictions(model, features, d, selected_features)

	n = len(outcomes)

	for rank in ranks:

		rank = int((rank*n)/100)

		prioritization_index = np.argsort(preds)
		predictions = np.zeros_like(prioritization_index)
		predictions[prioritization_index[-rank:]] = 1

		hr = treatment_effect('hazard_ratio', outcomes[predictions==1], 
													(interventions==intervention)[predictions==1],
													n_bootstrap=n_bootstrap)
		hrs.append(hr)

		rmst = treatment_effect('restricted_mean', outcomes[predictions==1], 
													(interventions==intervention)[predictions==1],
													horizons=5*365.25,
													n_bootstrap=n_bootstrap)
		rmsts.append(rmst)

		risk = treatment_effect('survival_at', outcomes[predictions==1], 
														(interventions==intervention)[predictions==1],
														horizons=5*365.25,
														n_bootstrap=n_bootstrap)
		risks.append(risk)

	return rmsts, hrs, risks


#def aggregate_results_ascvd(model, features, outcomes, intervnetions, intervention)