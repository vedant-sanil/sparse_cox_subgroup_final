import torch

from .utils import weighted_breslow_estimator
from .utils import weighted_partial_ll_loss
#from .utils import soft_threshold
from .utils import IdentifiableLinear

import numpy as np

from tqdm import tqdm
from joblib import delayed, Parallel

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.util import Surv

import pandas as pd

# import sys
# sys.path.append('/zfsauton2/home/vsanil/projects/aihc/cardiovascular_clinical_trials/auton-survival')
#from auton_survival.metrics import treatment_effect

#effect_size = 5


# def soft_threshold(beta, lambd):
#   return torch.nn.Softshrink(lambd=lambd)(beta)


def soft_threshold(beta, lambd):

  """https://www.cs.ubc.ca/~schmidtm/Courses/540-W17/L5.pdf"""

  group_norm = torch.norm(beta, dim=0)

  shrinked_norm = torch.max(torch.zeros_like(group_norm), group_norm - lambd)
  group_mag = beta/group_norm
  shrinked_vals = group_mag*shrinked_norm 

  return shrinked_vals


class SparseCoxSubgroup():
  """ Sparse Cox Subgroup Mixture to discover Heterogenous Effects."""

  def __init__(self, k=2, alpha=1e3, step_size=1e-5, n_epochs=100, n_restarts=10, bias=False, random_seed=100,
               debug=False, n_threads=1, temp=1, effect_size=5, params={}):
    
    assert isinstance(k, int) and (k>1)
    assert isinstance(alpha, float) and (alpha>=0)
    assert isinstance(step_size, float) and (step_size>0)
    assert isinstance(n_epochs, int) and (n_epochs>0)
    assert isinstance(n_restarts, int) and (n_restarts>0)
    
    self.k = k
    self.alpha = alpha
    self.step_size = step_size
    self.n_epochs = n_epochs
    self.n_restarts = n_restarts
    self.bias = bias
    self.random_seed = random_seed
    self.debug = debug
    self.n_threads = n_threads
    self.results = []
    self.gating_weight = 1
    self.temp = temp
    self.effect_size = effect_size
    self.params = params

    self.fitted = False

  def fitted_log_likelihood():
    raise NotImplementedError()

  def _get_posteriors(self):
    raise NotImplementedError()

  def initialize(self, x):
    if 'gate' in self.params:
        gate_params = IdentifiableLinear(x.shape[1], self.k, bias=self.bias, linear=self.params['gate'].linear)
    else:
        gate_params = IdentifiableLinear(x.shape[1], self.k, bias=self.bias).float()

    if 'expert' in self.params:
        expert_params = torch.from_numpy(self.params['expert'].detach().numpy())
    else:
        expert_params = np.zeros(x.shape[1]).reshape(-1, 1).astype(np.float32)
        expert_params = torch.tensor(expert_params, requires_grad=True)

    if 'effect' in self.params:
        effect_params = torch.tensor(self.params['effect'].detach().numpy())
    else:
        if self.k == 2:
            effect_params = np.array([1e-1,-1e-1]).astype(np.float32)
        elif self.k==3:
            effect_params = np.array([1e-1, -1e-1, 0]).astype(np.float32)
        
            effect_params = torch.tensor(effect_params, requires_grad=True)

    cumulative_hazard = None
    if 'cumulative_hazard' in self.params:
        cumulative_hazard = torch.from_numpy(self.params['cumulative_hazard'].detach().numpy())

    return gate_params, expert_params, effect_params, cumulative_hazard


  def _e_step(self, parameters, x, t, e, a):
    """ The Expectation-Step of the EM-Algorithm."""

    expert_lrisks = x.mm(parameters['expert']).expand(x.shape[0], self.k)
    if self.k == 2:
      effects = torch.stack([a*parameters['effect'][0], 
                             0*a*parameters['effect'][0]], dim=1)
    elif self.k == 3:
      effects = torch.stack([a*parameters['effect'][0], 
                             -a*parameters['effect'][0],
                                0*a*parameters['effect'][0]], dim=1)
    # effects = parameters['effect'] 
    # #if self.k==3: effects = torch.cat([effects, torch.zeros(1)])
    # effects = a.reshape(-1, 1)*effects.expand(len(a), -1)
    

    expert_lrisks = effects + expert_lrisks.clone() 
    unnormalized_probs = []

    for i in range(self.k):
      
      k_lrisks = expert_lrisks[:, i].detach().cpu().numpy() 

      k_bl_cum_hazard = parameters['cumulative_hazard']#[i]
      k_bl_cum_hazard = np.array(k_bl_cum_hazard.loc[[float(t_) for t_ in t]])
      
      k_risks = np.exp(np.clip(k_lrisks, -50, 50))

      k_bl_survival = np.exp(-np.array(k_bl_cum_hazard))

      k_survival = k_bl_survival**k_risks
      k_hazards = k_risks * k_survival
    
      k_unnormalized_probs = np.where(e.detach().numpy().astype(bool),
                                      k_hazards, k_survival)
      unnormalized_probs.append(k_unnormalized_probs)	

    unnormalized_probs = np.array(unnormalized_probs).T
    
    #gate_logits = x.mm(parameters['gate'])
    gate_logits = parameters['gate'](x)/self.temp
    gate_probs = torch.nn.functional.softmax(gate_logits, dim=1).detach()#.cpu().numpy()

    unnormalized_probs = gate_probs*unnormalized_probs

    posteriors = unnormalized_probs / torch.sum(unnormalized_probs, dim=1, keepdim=True)


    return posteriors.detach().cpu().numpy()


  def _m_step(self, parameters, x, t, e, a, posteriors, optimizer, update_breslow=True):
    """The Maximization-Step of the EM-Algorithm."""
    
  #  optimizer.zero_grad()
    posteriors = torch.from_numpy(posteriors).to(t.device)
 
    expert_lrisks = x.mm(parameters['expert']).expand(x.shape[0], self.k)
    if self.k == 2:
      effects = torch.stack([a*parameters['effect'][0], 
                           0*a*parameters['effect'][0]], dim=1)
    elif self.k == 3:
      effects = torch.stack([a*parameters['effect'][0], 
                            -a*parameters['effect'][0],
                           0*a*parameters['effect'][0]], dim=1)
    # effects = parameters['effect'] 
    # #if self.k==3: effects = torch.cat([effects, torch.zeros(1)])
    # effects = a.reshape(-1, 1)*effects.expand(len(a), -1)

    expert_lrisks = effects + expert_lrisks.clone() 

    # expert_loss = 0.
    # for i in range(self.k):
    expert_loss = weighted_partial_ll_loss(expert_lrisks.reshape(-1), 
                                           torch.stack([t]*self.k).T.reshape(-1), 
                                           torch.stack([e]*self.k).T.reshape(-1), 
                                           weights=posteriors.reshape(-1))

    gate_logits = parameters['gate'](x)/self.temp
    #gate_logits = x.mm(parameters['gate']) #+ 1000
    gate_loss = torch.nn.functional.log_softmax(gate_logits, dim=1)
    gate_loss = -(posteriors*gate_loss).sum() 

    loss = (expert_loss+self.gating_weight*gate_loss)

    loss.backward()
    #optimizer.step()

    loss = loss + self.alpha*torch.abs(parameters['gate'].linear.weight).sum()

    if self.debug:
      print("\n", float(expert_loss), float(gate_loss), float(loss))

    with torch.no_grad():

      # Compute the Genralized Gradient
      grad_gate = parameters['gate'].linear.weight.data - (self.step_size*parameters['gate'].linear.weight.grad)
      parameters['gate'].linear.weight.data = soft_threshold(grad_gate, self.step_size*self.alpha)
      parameters['gate'].linear.weight.grad = None

      if self.bias:
        grad_gate = parameters['gate'].linear.bias.data - (self.step_size*parameters['gate'].linear.bias.grad)
        parameters['gate'].linear.bias.data = grad_gate
        parameters['gate'].linear.bias.grad = None

      grad_expert = parameters['expert'].data - (self.step_size*parameters['expert'].grad)
      parameters['expert'].data = grad_expert
      parameters['expert'].grad = None

      grad_effect = parameters['effect'].data - (self.step_size*parameters['effect'].grad)
      parameters['effect'].data = grad_effect
      parameters['effect'].grad = None

      if update_breslow:
        # expert_lrisks = a.reshape(-1, 1)*parameters['effect'].expand(x.shape[0], -1)
        # expert_lrisks += x.mm(parameters['expert']).expand(x.shape[0], self.k) 
        expert_lrisks = x.mm(parameters['expert']).expand(x.shape[0], self.k)
        if self.k == 2:
          effects = torch.stack([a*parameters['effect'][0], 
                              0*a*parameters['effect'][0]], dim=1)
        elif self.k == 3:
          effects = torch.stack([a*parameters['effect'][0], 
                                -a*parameters['effect'][0],
                              0*a*parameters['effect'][0]], dim=1)
        # effects = parameters['effect'] 
        # #if self.k==3: effects = torch.cat([effects, torch.zeros(1)])
        # effects = a.reshape(-1, 1)*effects.expand(len(a), -1)
        
        expert_lrisks =  effects + expert_lrisks.clone()

        cumulative_hazard = weighted_breslow_estimator(torch.stack([t]*self.k).T.reshape(-1).cpu().numpy(), 
                                                       torch.stack([e]*self.k).T.reshape(-1).cpu().numpy(),
                                                       risks=expert_lrisks.reshape(-1).exp().detach().cpu().numpy(),
                                                       weights=posteriors.reshape(-1).detach().cpu().numpy())

        parameters['cumulative_hazard'] = pd.Series(cumulative_hazard[1], index=cumulative_hazard[0])



    return parameters, float(loss)/x.shape[0]
  
  def _to_torch(self, **kwargs):
    for arg in kwargs:
      yield torch.from_numpy(kwargs.get(arg)).float()
  
  def _fit(self, x, t, e, a, bs=256, random_seed=0):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    #a = 2*(a-0.5)

    # Fit a Cox Model to inititlize parameters

    base_cum_hazard = nelson_aalen_estimator(e.astype(bool), t)
  
    # Initialize the Data as Torch Tensors
    x, t, e, a = self._to_torch(x=x, t=t, e=e, a=a)

    gate_params, expert_params, effect_params, cumulative_hazard = self.initialize(x)
    if cumulative_hazard is not None:
        parameters['cumulative_hazard'] = cumulative_hazard

    parameters = {}

    # cumulative_hazards = {} 
    # for i in range(self.k):
    #   hazards = base_cum_hazard[1]
    #   cumulative_hazards[i] = pd.Series(hazards, index=np.unique(t.detach().cpu().numpy()))
    #parameters['cumulative_hazards'] = cumulative_hazards

    #effect_params = torch.tensor(effect_params, requires_grad=True)    
    #effect_params = torch.tensor(np.array([-1., 1.]), requires_grad=True)
    #effect_params = effect_params - 0.5

    parameters['gate'] = gate_params
    parameters['expert'] = expert_params
    parameters['effect'] = effect_params

    params = [parameters[param] for param in ['expert', 'effect']]+list(parameters['gate'].parameters())

    optimizer = torch.optim.Adam(params, lr=self.step_size)

    losses_ = [] 
    for i in tqdm(range(self.n_epochs), desc='Gradient Update Steps'):
      # EM Loop

      # Batch data
      n = x.shape[0]
      batches = (n // bs) + 1

      epoch_q_loss = 0
      for i in range(batches):
        xb = x[i*bs:(i+1)*bs]
        tb = t[i*bs:(i+1)*bs]
        eb = e[i*bs:(i+1)*bs]
        ab = a[i*bs:(i+1)*bs]
        with torch.no_grad():
          if parameters.get('cumulative_hazard', None) is not None: 
            posteriors = self._e_step(parameters, xb, tb, eb, ab)
          else:
            posteriors = np.random.random((x.shape[0], self.k))
            posteriors = posteriors/posteriors.sum(axis=1).reshape(-1,1)

        parameters, q_loss = self._m_step(parameters, xb, tb, eb, ab,
                                          posteriors, optimizer,
                                          update_breslow=(i%1)==0)
        
        epoch_q_loss += q_loss

      losses_.append(epoch_q_loss)
      #print(losses_)

    self.losses.append(losses_)

    return parameters, losses_


  def fit(self, x, t, e, a, batch_size=256, treat_prio=None, selection_criteria='hr'):

    if treat_prio is None and selection_criteria == 'hr':
        raise Exception('Using Hazard Ratio as selection criteria requires treatment priority to be set')

    self.negative_log_likelihood = np.inf
    self.losses = []
    best_params, max_hr, outcomes = None, -np.inf, pd.DataFrame({'time':t, 'event':e})
    if len(self.results) == 0:
      results = Parallel(n_jobs=self.n_threads)(delayed(self._fit)(x, t, e, a, bs=batch_size, random_seed=self.random_seed*(i+1)) for i in range(self.n_restarts))
      self.results = results
    self.fitted = True

    for i, result in enumerate(self.results):
      parameters, negative_log_likelihood = result[0], result[1][-1]

      # Compute difference in Hazard Ratios for each restart
      self.parameters = parameters

      if selection_criteria == 'nll':
        if negative_log_likelihood < self.negative_log_likelihood:
          self.negative_log_likelihood = negative_log_likelihood
          best_params = parameters
          print("best yet:", i)
    self.parameters = best_params		
    self.fitted = True

    return results[1]
  
  def set_params(self, restart_num):
    print(f"Setting model parameters to restart number {restart_num}")
    self.parameters = self.results[restart_num][0]
    self.fitted = True			


  def predict_proba(self, x):

    if not self.fitted:
      raise ValueError("Model not fitted.")

    x = list(self._to_torch(x=x))[0]

    with torch.no_grad():
      return torch.nn.functional.log_softmax(self.parameters['gate'](x)/self.temp,
                                             dim=1).exp().detach().cpu().numpy()