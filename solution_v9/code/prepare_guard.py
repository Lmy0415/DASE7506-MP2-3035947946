"""Reproduce v8 outer holdout; evaluate a fixed relation guard without fitting it."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,shutil
from pathlib import Path
import numpy as np,joblib
from common import load_inputs,read_jsonl
from features import labels
from train_constraints import fit,rows,metrics,hard
from submission import constrain,score_components
root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data'
b,_=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
_,ibase,_,hbase=joblib.load(root.parent/'solution_v7/runs/internal_features.joblib');x=joblib.load(root.parent/'solution_v8/runs/target_features_v2.joblib')[:,:37]
w=fit(x[rows(tr)],ibase,y[tr],.001);outer=hbase+(x[rows(ho)]@w).reshape(-1,4);guard=constrain([b[int(i)] for i in ho],outer)
internal={'v6':metrics(hbase,y[ho],h[ho]),'v8':metrics(outer,y[ho],h[ho]),'guarded_v8':metrics(guard,y[ho],h[ho])}
assert internal['guarded_v8']['correct']>=internal['v8']['correct']
bundle=out/'model_bundle';shutil.copytree(root.parent/'solution_v8/runs/model_bundle',bundle/'target_base',dirs_exist_ok=True)
# Earlier soft-constraint ablation assets are not part of the chosen inference method.
(bundle/'target.npz').unlink(missing_ok=True)
if (bundle/'base').exists():shutil.rmtree(bundle/'base')
vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);s=score_components(vb,va,bundle)
np.savez(out/'validation_scores.npz',**s)
result={'internal':internal,'validation':{k:metrics(v,vy,vh) for k,v in s.items()},'test_used':False,'guard_fitted':False,'selection':'v8 plus fixed question-side relation guard; internal non-regression required'}
changes=[]
for i in np.flatnonzero(s['base'].argmax(1)!=s['selected'].argmax(1)):
 old=int(s['base'][i].argmax());new=int(s['selected'][i].argmax());changes.append({'row':int(i),'old':old,'new':new,'old_correct':bool(vy[i,old]),'new_correct':bool(vy[i,new])})
result['changes']=changes;(out/'guard_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
