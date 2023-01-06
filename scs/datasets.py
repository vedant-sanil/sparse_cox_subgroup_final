from sklearn.utils import shuffle

import numpy as np
import pandas as pd


def load_accord_dataset(location, intervention='Glycemia', random_seed=0, **kwargs):

    assert intervention in ['Glycemia', 'Lipid', 'BP']

    endpoint = kwargs.get('endpoint', 'po')

    outcomes, features = load_accord(location=location, endpoint=endpoint)
    
    BP_patients = (features.arm==1.0)|(features.arm==2.0)|(features.arm==3.0)|(features.arm==4.0)
    lipid_patients = ~BP_patients

    glycemic_interventions = (features['arm']==1)|(features['arm']==2)|(features['arm']==5)|(features['arm']==6)

    if intervention == 'Glycemia':
        interventions = glycemic_interventions
        interventions[interventions==True] = "Standard Glycemia"
        interventions[interventions==False] = "Intensive Glycemia" 
    else:
        features['glycemic_arm'] = glycemic_interventions
        if intervention == 'BP':
            features = features.loc[BP_patients]
            outcomes = outcomes.loc[BP_patients]
            interventions = (features.arm==1) | (features.arm==3)
            interventions[interventions==True] = "Intensive BP"
            interventions[interventions==False] = "Standard BP" 

        elif intervention == 'Lipid':
            features = features.loc[lipid_patients]
            outcomes = outcomes.loc[lipid_patients]
            interventions = (features.arm==6) | (features.arm==8)
            interventions[interventions==True] = "Lipid Placebo"
            interventions[interventions==False] = "Lipid Fibrate"

    outcomes['time'] = outcomes['time']*365

    outcomes, features, interventions = shuffle(outcomes, features, interventions, random_state=random_seed) 

    cat_feats = ['female','cvd_hx_baseline', 'raceclass','x1diab', 'x2mi', 
                'x2stroke', 'x2angina','cabg','ptci','cvdhist','orevasc','x2hbac11','x2hbac9',
                'x3malb','x3lvh','x3sten','x4llmeds','x4gender','x4hdlf', 'x4hdlm','x4bpmeds',
                'x4notmed','x4smoke','x4bmi']

    num_feats = ['baseline_age', 'sbp', 'dbp', 'hr', 'chol', 'trig', 'vldl', 
                 'ldl', 'hdl','fpg', 'alt', 'cpk', 'potassium', 'screat', 'gfr',
                 'ualb', 'ucreat', 'uacr'] 

    if intervention in ['BP', 'Lipid']:
        cat_feats.append('glycemic_arm')

    all_feats = list(set(cat_feats + num_feats))

    features = features[all_feats]

    return outcomes, features, interventions, cat_feats, num_feats


def load_accord(endpoint=None, features=None, location=''):  

  # Default Baseline Features to include: 
  if features is None:

    print("No Features Specified!! using default baseline features.")

    features = {
      
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/accord_key.sas7bdat': ['female', 'baseline_age', 'arm', 
                                                                                            'cvd_hx_baseline', 'raceclass',
                                                                                            'treatment'],
                
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/bloodpressure.sas7bdat': ['sbp', 'dbp', 'hr'],

                'ACCORD/4-Data Sets - CRFs/4a-CRF Data Sets/f01_inclusionexclusionsummary.sas7bdat': ['x1diab', 'x2mi', 
                'x2stroke', 'x2angina','cabg','ptci','cvdhist','orevasc','x2hbac11','x2hbac9','x3malb','x3lvh','x3sten','x4llmeds',
                'x4gender','x4hdlf', 'x4hdlm','x4bpmeds','x4notmed','x4smoke','x4bmi'],
                
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/lipids.sas7bdat': ['chol', 'trig', 'vldl', 'ldl', 'hdl'],
                                
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/otherlabs.sas7bdat': ['fpg', 'alt', 'cpk', 
                                                                                         'potassium', 'screat', 'gfr',
                                                                                         'ualb', 'ucreat', 'uacr'],
            }
           
                
    # outcomes  = {'ACCORD_Private/Data Sets - Analysis 201604/CVDOutcomes_201604.sas7bdat':['censor_po','type_po',                                                   
    #                                             'fuyrs_po', 'fuyrs_po7p', 'censor_tm', 'type_tm', 
    #                                             'fuyrs_tm', 'fuyrs_tm7p', 'censor_cm', 'type_nmi', 'fuyrs_nmi7p', 'censor_nst',
    #                                             'type_nst', 'fuyrs_nst', 'fuyrs_nst7p', 'censor_tst', 'fuyrs_tst', 'fuyrs_tst7p'
    #                                             'censor_chf', 'fuyrs_chf', 'censor_ex', 'type_ex', 'fuyrs_ex', 'fuyrs_ex7p', 
    #                                             'censor_maj', 'type_maj', 'fuyrs_maj7p']
    #               }
        
  if endpoint is None:
    print("No Endpoint specified, using primary study endpoint.")
    endpoint = 'po'

  # Set the outcome variable,
  event = 'censor_'+endpoint
  time = 'fuyrs_'+endpoint

  outcome_tbl = 'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/cvdoutcomes.sas7bdat'

  outcomes, features = _load_generic_biolincc_dataset(outcome_tbl=outcome_tbl, 
                                                      time_col=time, 
                                                      event_col=event,
                                                      features=features,
                                                      id_col='MaskID',
                                                      location=location, 
                                                      visit_col='Visit', 
                                                      baseline_visit=(b'BLR', b'S01'))
  outcomes['event'] = 1-outcomes['event'] 
  outcomes['time'] = outcomes['time']

  outcomes = outcomes.loc[outcomes['time']>1.0]
  features = features.loc[outcomes.index]

  outcomes['time'] = outcomes['time']-1 

  return outcomes, features


def _load_generic_biolincc_dataset(outcome_tbl, time_col, event_col, features, id_col,
                                   visit_col=None, baseline_visit=None, location=''):

  if not isinstance(baseline_visit, (tuple, set, list)):
    baseline_visit = [baseline_visit]

  # List of all features to extract
  all_features = []
  for feature in features:
    all_features+=features[feature]
  all_features = list(set(all_features)) # Only take the unqiue columns

  if '.sas' in outcome_tbl: outcomes = pd.read_sas(location+outcome_tbl, index=id_col)
  elif '.csv' in outcome_tbl: outcomes = pd.read_csv(location+outcome_tbl, index_col=id_col, encoding='latin-1')
  else: raise NotImplementedError()

  outcomes = outcomes[[time_col, event_col]]

  dataset = outcomes.copy()
  dataset.columns = ['time', 'event']

  for feature in features:
    
    if '.sas' in outcome_tbl: table = pd.read_sas(location+feature, index=id_col)
    elif '.csv' in outcome_tbl: table = pd.read_csv(location+feature, index_col=id_col)
    else: raise NotImplementedError()

    if (visit_col is not None) and (visit_col in table.columns):
      mask = np.zeros(len(table[visit_col])).astype('bool')
      for baseline_visit_ in baseline_visit:
        mask = mask | (table[visit_col]==baseline_visit_)
      table = table[mask] 
    table = table[features[feature]]
    print(table.shape)
    dataset = dataset.join(table)

  outcomes = dataset[['time', 'event']]
  features = dataset[all_features]

  outcomes = _encode_cols_index(outcomes)
  features = _encode_cols_index(features)

  return outcomes, features

def _encode_cols_index(df):

  columns = df.columns

  # Convert Objects to Strings
  for col in columns:
    if df[col].dtype == 'O':
      df.loc[:, col] = df[col].values.astype(str)

  # If Index is Object, covert to String
  if df.index.dtype == 'O':
   df.index = df.index.values.astype(str)

  return df
