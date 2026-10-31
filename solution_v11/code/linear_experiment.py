"""Crossfit regularized centered linear rankers as complementary experts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,stats

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs';project=root.parent;data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=e-e.mean(1,keepdims=True);hand=features(b,e)
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);ve=va['embeddings'];vcx=ve-ve.mean(1,keepdims=True);vhand=features(vb,ve)
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261101).split(np.arange(len(b)),h));cache=out/'linear_oof.npz'
 if cache.exists():
  with np.load(cache) as z:scores={k:z[k] for k in z.files}
 else:
  scores={}
  for view in ['embedding','combined']:
   x=cx if view=='embedding' else np.concatenate([cx,hand-hand.mean(1,keepdims=True)],-1)
   vx=vcx if view=='embedding' else np.concatenate([vcx,vhand-vhand.mean(1,keepdims=True)],-1)
   for sampling in ['all','hard']:
    for alpha in [100.,1000.,10000.]:
     oof=np.zeros((len(b),4));vals=[]
     for ti,hi in folds:
      if sampling=='hard':ti=ti[h[ti]]
      sd=x[ti].reshape(-1,x.shape[-1]).std(0).clip(.05);train=x[ti]/sd;test=x[hi]/sd;model=Ridge(alpha=alpha,fit_intercept=False,solver='lsqr',tol=1e-4).fit(train.reshape(-1,train.shape[-1]),(y[ti]-y[ti].mean(1,keepdims=True)).ravel());oof[hi]=model.predict(test.reshape(-1,test.shape[-1])).reshape(-1,4);vals.append(model.predict((vx/sd).reshape(-1,vx.shape[-1])).reshape(-1,4))
     key=f'{view}_{sampling}_{int(alpha)}';scores[key]=oof;scores[key+'_val']=np.mean(vals,0);print(json.dumps({'model':key,'seconds':time.perf_counter()-tick}),flush=True)
  np.savez_compressed(cache,**scores)
 results=[]
 for key in scores:
  if key.endswith('_val'):continue
  t=scores[key];scale=(base-base.mean(1,keepdims=True)).std()/t.std()
  for alpha in [.1,.25,.5,1.]:
   s=base+alpha*scale*t;results.append({'model':key,'strength':alpha,'scale':float(scale),**stats(s,y,h,base)})
 results.append({'model':'none','strength':0.,'scale':0.,**stats(base,y,h,base)});best=max(results,key=lambda r:(r['hard_correct'],-r['strength']))
 (out/'linear_search.json').write_text(json.dumps({'records':results,'selected':best,'baseline':stats(base,y,h,base),'crossfit_nested_training_library':False},indent=2)+'\n')
 if best['model']!='none':
  s=vbase+best['strength']*best['scale']*scores[best['model']+'_val'];result={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(s,vy,vh,vbase),'training_seconds':time.perf_counter()-tick};(out/'linear_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
