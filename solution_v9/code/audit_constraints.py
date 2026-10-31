"""Read train/validation only; report bounded rule coverage and intervention ablations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,collections
import numpy as np,joblib
from common import load_inputs,read_jsonl,parse_answer
from features import labels
from relation_constraints import hypotheses,features
from train_constraints import metrics,hard,fit
import base_submission
root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';result={}
for split in ['train','validation']:
 b,arrays=load_inputs(split,data);refs=read_jsonl(data/f'gsm8k_{split}.jsonl');y=labels(b,refs);h=hard(b)
 if split=='train':
  _,base=joblib.load(root.parent/'solution_v7/runs/full_features.joblib')
 else:base=base_submission.score_components(b,arrays,root.parent/'solution_v6/runs/model_bundle')['selected']
 counts=collections.defaultdict(lambda:{'covered':0,'reference_agreement':0,'expected_in_candidates':0});cases=[];override=base.copy()
 for i,(row,ref) in enumerate(zip(b,refs)):
  hs=hypotheses(row['question'])
  for hyp in hs:
   c=counts[hyp['family']];c['covered']+=1;c['reference_agreement']+=hyp['expected']==parse_answer(ref['answer'])
   matches=[j for j,candidate in enumerate(row['candidates']) if parse_answer(candidate['text'])==hyp['expected']];c['expected_in_candidates']+=bool(matches)
   cases.append({'row':i,'question':row['question'],'hypothesis':hyp,'reference_agrees':hyp['expected']==parse_answer(ref['answer']),'matching_candidates':matches})
  answers={z['expected'] for z in hs}
  if len(answers)==1:
   chosen=[j for j,c in enumerate(row['candidates']) if parse_answer(c['text']) in answers]
   if chosen:override[i,chosen]+=1000
 # An explicit override is diagnostic only, not a selected deployment method.
 result[split]={'family_counts':dict(counts),'baseline':metrics(base,y,h),'override_ablation':metrics(override,y,h),'cases':cases}
 if split=='train':
  soft_weight=fit(features(b,'shared'),base,y,.1)
  with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:ids=z['holdout_ids']
  _,_,_,outer=joblib.load(root.parent/'solution_v7/runs/internal_features.joblib');delta=override-base
  result[split]['strict_internal_override']=metrics(outer+delta[ids],y[ids],h[ids])
 else:
  score=base+(features(b,'shared')@soft_weight).reshape(-1,4)
  result[split]['nonzero_soft_ablation']=metrics(score,y,h)
(root/'runs/constraint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({s:{k:v for k,v in r.items() if k!='cases'} for s,r in result.items()},indent=2))
