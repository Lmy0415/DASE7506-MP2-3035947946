"""Reference-mined relation expert with abstention-aware learned rank correction."""
from pathlib import Path
import numpy as np,joblib
import base_submission,target_submission
from reference_patterns import features
from common import parse_answer
from relation_constraints import hypotheses

def constrain(bank,scores):
 selected=scores.copy()
 for i,row in enumerate(bank):
  expected={h['expected'] for h in hypotheses(row['question'])}
  if len(expected)!=1:continue
  matches=np.array([parse_answer(c['text']) in expected for c in row['candidates']])
  if matches.any() and not matches.all():selected[i,~matches]=scores[i].min()-1.
 return selected

def score_components(bank,arrays,checkpoint=None):
 root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/model_bundle'
 with np.load(root/'correction.npz',allow_pickle=False) as z:
  mode=str(z['base']);base=(base_submission if mode=='v6' else target_submission).score_components(bank,arrays,root/'base')['selected']
  if mode=='v9':base=constrain(bank,base)
  x=features(bank,joblib.load(root/'patterns.joblib'))
  if str(z['view'])=='strong':x*= (np.maximum(x[:,:,2],x[:,:,3])>=.4)[:,:,None]
  delta=x@z['weight'];selected=base+float(z['strength'])*delta
 return {'base':base,'delta':delta,'selected':selected}

def predict(bank,arrays,checkpoint=None):
 if not bank:return []
 return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
