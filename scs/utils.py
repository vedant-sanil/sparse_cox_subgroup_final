import numpy as np
import torch

# def weighted_partial_ll_loss(lrisks, t, e, eps=1e-10, weights=None, type='naive'):

#   if weights is None:
#     weights = np.ones_like(t)

#   t = t + eps*np.random.random(len(t))

#   sidx = np.argsort(-t)

#   t, e, lrisks, weights = t[sidx], e[sidx], lrisks[sidx], weights[sidx]

#   lrisks = lrisks+weights.log() 
#   lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)

#   #plls = weights*(lrisks - lrisksdenom)
#   #plls = weights*(lrisks - lrisksdenom)
#   plls = lrisks - lrisksdenom

#   pll = plls[e == 1]

#   pll = torch.sum(pll)

#   return -pll

def weighted_partial_ll_loss(lrisks, t, e, eps=1e-10, weights=None, type='naive'):

  if weights is None:
    weights = np.ones_like(t)

  t = t + eps*np.random.random(len(t))

  sidx = np.argsort(-t)

  t, e, lrisks, weights = t[sidx], e[sidx], lrisks[sidx], weights[sidx]

  if type == 'naive':
    lrisksdenom = torch.logcumsumexp(lrisks+weights.log() , dim = 0)
    plls = weights*(lrisks - lrisksdenom)
  else:
    lrisks = lrisks+weights.log() 
    lrisksdenom = torch.logcumsumexp(lrisks , dim = 0)
    plls = (lrisks - lrisksdenom)

  pll = plls[e == 1]
  pll = torch.sum(pll)

  return -pll

def weighted_breslow_estimator(t, e, risks=None, weights=None):

  if risks is None:
    risks = np.ones_like(t)
  
  if weights is None:
    weights = np.ones_like(t)

  # sort the data by time
  idx = np.argsort(t)
  t, e, risks, weights = t[idx], e[idx], risks[idx], weights[idx]

  # compute changepoints all t
  diff = np.where((t[1:] - t[:-1]).astype(bool))[0] + 1
  diff = np.array([0] + list(diff))

  # compute changepoints for uncensored t
  diff_uncensored = np.where((t[e==1][1:] - t[e==1][:-1]).astype(bool))[0]
  diff_uncensored = np.array(list(diff_uncensored) + [len(t[e==1])-1])

  t_failed_counts = np.cumsum(weights[e==1])
  t_failed_counts = np.insert(t_failed_counts[diff_uncensored], 0, 0)
  t_failed_counts = t_failed_counts[1:] - t_failed_counts[:-1]

  # compute the number of deaths at each time point.
  t_failed_unique, _ = np.unique(t[e==1], return_counts=True)
  t_all_unique, _ = np.unique(t, return_counts=True)

  # find the indices of the failed times
  _, _, idx = np.intersect1d(t_failed_unique, t_all_unique, return_indices=True)

  t_failed_all = np.zeros_like(t_all_unique).astype(float)
  t_failed_all[idx] = t_failed_counts

  # compute the number of patients at-risk.
  adjusted_at_risk_counts = np.cumsum(weights[::-1]*risks[::-1])[::-1][diff]

  estimate = np.cumsum(t_failed_all/adjusted_at_risk_counts)

  return t_all_unique, estimate

def soft_threshold(beta, lambd):
  return torch.nn.Softshrink(lambd=lambd)(beta)



# class IdentifiableLinear(torch.nn.Module):

#   """
#   Softmax and LogSoftmax with K classes in pytorch are over specfied and lead to
#   issues of mis-identifiability. This class is a wrapper for linear layers that 
#   are correctly specified with K-1 columns. The output of this layer for the Kth
#   class is all zeros. This allows direct application of pytorch.nn.LogSoftmax
#   and pytorch.nn.Softmax.
#   """

#   def __init__(self, in_features, out_features, bias=True):

#     super(IdentifiableLinear, self).__init__()

#     assert out_features>0; "Output features must be greater than 0"

#     self.out_features = out_features
#     self.in_features = in_features
#     self.linear = torch.nn.Linear(in_features, max(out_features-1, 1), bias=bias)

#   @property
#   def bias(self):
#     return self.linear.bias

#   @property
#   def weight(self):
#     return self.linear.weight
  
#   def forward(self, x):
#     if self.out_features == 1:
#       return self.linear(x).reshape(-1, 1)
#     else:
#       zeros = torch.zeros(len(x), 1, device=x.device)
#       return torch.cat([self.linear(x), zeros], dim=1)


class IdentifiableLinear(torch.nn.Module):

  """
  Softmax and LogSoftmax with K classes in pytorch are over specfied and lead to
  issues of mis-identifiability. This class is a wrapper for linear layers that 
  are correctly specified with K-1 columns. The output of this layer for the Kth
  class is all zeros. This allows direct application of pytorch.nn.LogSoftmax
  and pytorch.nn.Softmax.
  """

  def __init__(self, in_features, out_features, bias=True, linear=None):

    super(IdentifiableLinear, self).__init__()

    assert out_features>0; "Output features must be greater than 0"

    self.out_features = out_features
    self.in_features = in_features
    if linear is None:
      self.linear = torch.nn.Linear(in_features, max(out_features-1, 1), bias=bias)
    else:
      self.linear = linear

  @property
  def bias(self):
    return self.linear.bias

  @property
  def weight(self):
    return self.linear.weight
  
  def forward(self, x):
    #return self.linear(x)

    if self.out_features == 1:
      return self.linear(x).reshape(-1, 1)
    else:
      zeros = torch.zeros(len(x), 1, device=x.device)
      return torch.cat([self.linear(x), zeros], dim=1)
