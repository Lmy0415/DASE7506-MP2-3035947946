"""Calibrate evidence aggregation over candidate answers, rather than traces."""
import numpy as np
from scipy.special import logsumexp
from common import parse_answer

def pool(bank,scores,temp,beta):
 out=scores.copy()
 for i,row in enumerate(bank):
  answers=[parse_answer(c['text']) for c in row['candidates']];groups={}
  for j,a in enumerate(answers):groups.setdefault(a if a is not None else ('invalid',j),[]).append(j)
  for group in groups.values():
   s=scores[i,group];aggregate=temp*logsumexp(s/temp)-beta*temp*np.log(len(group));out[i,group]=aggregate
 return out
