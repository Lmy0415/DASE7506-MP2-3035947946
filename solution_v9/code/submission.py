"""Target-binding verifier followed by conservative question-side constraints."""
from pathlib import Path
import numpy as np
from common import parse_answer
from relation_constraints import hypotheses
import target_submission

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
 base=target_submission.score_components(bank,arrays,root/'target_base')['selected']
 return {'base':base,'selected':constrain(bank,base)}

def predict(bank,arrays,checkpoint=None):
 if not bank:return []
 return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
