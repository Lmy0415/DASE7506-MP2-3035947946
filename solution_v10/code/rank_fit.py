"""Shared fixed-baseline pairwise logistic ranking fit and metrics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,shutil,time
from pathlib import Path
import numpy as np,joblib
from scipy.optimize import minimize
from scipy.special import expit
from common import load_inputs,read_jsonl,parse_answer
from features import labels
from relation_constraints import features
import base_submission

def metrics(s,y,h):
 c=y[np.arange(len(y)),s.argmax(1)];return {'correct':int(c.sum()),'n':len(y),'accuracy':float(c.mean()*100),'hard_correct':int(c[h].sum()),'hard_n':int(h.sum())}
def hard(b):return np.array([len({a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None})>1 for r in b])
def rows(ids):return (np.asarray(ids)[:,None]*4+np.arange(4)).ravel()
def fit(x,base,y,l2):
 pi=[];ni=[];ww=[]
 for i,t in enumerate(y):
  pos=np.flatnonzero(t);neg=np.flatnonzero(t==0)
  for p in pos:
   for n in neg:pi.append(i*4+p);ni.append(i*4+n);ww.append(1/(len(pos)*len(neg)))
 dx=x[pi]-x[ni];db=base.ravel()[pi]-base.ravel()[ni];ww=np.asarray(ww);ww/=ww.sum()
 def obj(w):
  margin=dx@w+db;g=-ww*expit(-margin)
  return float(ww@np.logaddexp(0,-margin)+l2/2*(w@w)),np.asarray(dx.T@g).ravel()+l2*w
 result=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9,'gtol':1e-6,'maxcor':10})
 if not result.success:raise RuntimeError(result.message)
 return result.x

