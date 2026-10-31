"""Calibrate disagreement among original members using training-only OOF."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from legacy_submission import member_scores
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'solution_v10/code'))
from rank_fit import fit

def make(members,ind,mode):
 m=members-members.mean(-1,keepdims=True);parts=[m]
 if mode=='extended':
  center=members-members.mean(1,keepdims=True);other=ind-ind.mean(1,keepdims=True);parts.extend([other[:,:,None],center*other[:,:,None]])
 return np.concatenate(parts,-1)
def flat(x):return sparse.csr_matrix(x.reshape(-1,x.shape[-1]))
def main():
 p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';b,a=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
 with np.load(out/'member_oof.npz') as z:members=z['members']
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'training/independent_expert/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vind=z['validation'].mean(0)
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vm=np.stack(member_scores(vb,va,p/'solution_v9/runs/model_bundle/target_base/base/legacy'),-1)
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261110).split(np.arange(len(b)),h));results=[]
 for mode in ['members','extended']:
  x=make(members,ind,mode);vx=make(vm,vind,mode)
  for l2 in [.001,.01,.1,1.]:
   delta=np.zeros((len(b),4))
   for ti,hi in folds:
    sd=x[ti].reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flat(x[ti]/sd),base[ti],y[ti],l2);delta[hi]=(x[hi]/sd)@w
   for alpha in [.25,.5,1.]:results.append({'mode':mode,'l2':l2,'strength':alpha,**stats(base+alpha*delta,y,h,base)})
 results.append({'mode':'none','l2':1.,'strength':0.,**stats(base,y,h,base)});best=max(results,key=lambda r:(r['hard_correct'],-r['strength'],r['l2']));(out/'member_calibration_search.json').write_text(json.dumps({'selected':best,'records':results,'fully_nested':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if not best['strength']:return
 x=make(members,ind,best['mode']);vx=make(vm,vind,best['mode']);sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flat(x/sd),base,y,best['l2']);val=vbase+best['strength']*((vx/sd)@w);res={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(val,vy,vh,vbase)};(out/'member_calibration_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
