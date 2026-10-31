"""Train-only crossfit of conservative new calculation evidence."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from expanded_calculations import features
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'solution_v10/code'))
from rank_fit import fit

def main():
 p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,_=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);cache=out/'expanded_features.npz'
 if cache.exists():
  with np.load(cache) as z:x=z['train']
 else:x=features(b);np.savez_compressed(cache,train=x)
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 usable=(x[:,:,0]>0);bad=(x[:,:,5]>0);print(json.dumps({'available_candidates':int(usable.sum()),'bad_candidates':int(bad.sum()),'correct_among_bad':float(y[bad].mean()) if bad.any() else None,'seconds':time.perf_counter()-tick}),flush=True)
 rr=[];folds=list(StratifiedKFold(5,shuffle=True,random_state=20261105).split(np.arange(len(b)),h));preds={}
 for view,xx in [('all',x[:,:,:12]),('errors',x[:,:,[2,3,4,5,8,9,11]])]:
  for l2 in [.001,.01,.1,1.]:
   d=np.zeros((len(b),4))
   for ti,hi in folds:
    sd=xx[ti].reshape(-1,xx.shape[-1]).std(0).clip(.05);v=xx[ti]/sd;w=fit(sparse.csr_matrix(v.reshape(-1,v.shape[-1])),base[ti],y[ti],l2);d[hi]=(xx[hi]/sd)@w
   preds[(view,l2)]=d
   for strength in [.25,.5,1.]:rr.append({'view':view,'l2':l2,'strength':strength,**stats(base+strength*d,y,h,base)})
   print(json.dumps({'view':view,'l2':l2,'seconds':time.perf_counter()-tick}),flush=True)
 rr.append({'view':'none','l2':1.,'strength':0.,**stats(base,y,h,base)});best=max(rr,key=lambda r:(r['hard_correct'],-r['strength'],r['l2']));(out/'expanded_search.json').write_text(json.dumps({'selected':best,'records':rr,'crossfit_nested_training_library':False},indent=2)+'\n');np.savez_compressed(out/'expanded_oof.npz',**{f'{a}_{b}':v for (a,b),v in preds.items()});print(json.dumps({'selected':best}),flush=True)
 if not best['strength']:return
 xx=x[:,:,:12] if best['view']=='all' else x[:,:,[2,3,4,5,8,9,11]];sd=xx.reshape(-1,xx.shape[-1]).std(0).clip(.05);w=fit(sparse.csr_matrix((xx/sd).reshape(-1,xx.shape[-1])),base,y,best['l2'])
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vx=features(vb);vx=vx[:,:,:12] if best['view']=='all' else vx[:,:,[2,3,4,5,8,9,11]]
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 delta=(vx/sd)@w;np.savez(out/'expanded_model.npz',weight=w,sd=sd,view=np.array(best['view']),strength=best['strength']);np.savez(out/'expanded_validation_scores.npz',base=vbase,delta=delta);s=vbase+best['strength']*delta;res={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(s,vy,vh,vbase),'seconds':time.perf_counter()-tick};(out/'expanded_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
