"""Screen independent pre-existing reference-free ranker outputs on train OOF only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from common import load_bank,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';b=load_bank('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
with np.load(root.parent/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
results=[]
for name in ['cv_math_cosine50_r2','cv_math_h256_r2_e20','cv_math_r1_e20','cv_nomath_r1_e20']:
 with np.load(root.parent/'kimi_mp2/DASE7506-MP2-main/runs'/f'{name}_scores.npz') as z:
  a=z['oof'];assert (z['oof_count']>=1).all()
 a=a-a.mean(1,keepdims=True);s=base-base.mean(1,keepdims=True);scale=s.std()/a.std();a*=scale
 for alpha in [0.,.1,.2,.35,.5,1.]:
  z=s+alpha*a;r={'model':name,'alpha':alpha,'scale':float(scale),**stats(z,y,h,s)};results.append(r)
(root/'runs/independent_crossfit.json').write_text(json.dumps({'records':results,'source':'pre-existing crossfitted train predictions; no validation or test used','nested_outer_exclusion':False},indent=2)+'\n')
print(json.dumps(results,indent=2))
