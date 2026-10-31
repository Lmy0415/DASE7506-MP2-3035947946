"""Train-only crossfit screening for grouping candidate answers."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_bank,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from cluster_pool import pool
p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';b=load_bank('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
_,s6=joblib.load(p/'solution_v7/runs/full_features.joblib')
with np.load(p/'solution_v10/runs/honest_scores.npz') as z:s9=z['full_base']
seeds=[311,312,313];folds=[list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.arange(len(b)),h)) for seed in seeds];results=[]
for name,base in [('v6',s6),('v9',s9)]:
 baseline=stats(base,y,h,base);results.append({'base':name,'temp':0.,'beta':0.,'strength':0.,**baseline})
 for t in [.05,.1,.2,.4,.8,1.6]:
  for beta in [0,.25,.5,.75,1.]:
   a=pool(b,base,t,beta);r={'base':name,'temp':t,'beta':beta,'strength':1.,**stats(a,y,h,base)};results.append(r)
best=max(results,key=lambda r:(r['hard_correct'],-r['strength'],r['beta']));(out/'pool_search.json').write_text(json.dumps({'selected':best,'records':results,'per_question_oof':True,'no_test_used':True},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
if best['strength']:
 vb=load_bank('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb)
 if best['base']=='v6':
  with np.load(p/'solution_v8/runs/validation_scores.npz') as z:base=z['base']
 else:
  with np.load(p/'solution_v9/runs/validation_scores.npz') as z:base=z['selected']
 val=pool(vb,base,best['temp'],best['beta']);res={'selected':best,'validation_base':stats(base,vy,vh,base),'validation_candidate':stats(val,vy,vh,base)};(out/'pool_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
