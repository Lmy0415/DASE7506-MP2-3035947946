"""Bounded train holdout search followed by one validation comparison."""
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

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data';start=time.perf_counter()
 b,_=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
 with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
 _,ibase,_,hbase=joblib.load(root.parent/'solution_v7/runs/internal_features.joblib');_,fbase=joblib.load(root.parent/'solution_v7/runs/full_features.joblib')
 cache=out/'constraint_features.joblib'
 if cache.exists():x=joblib.load(cache)
 else:x=features(b);joblib.dump(x,cache)
 print(json.dumps({'features':list(x.shape),'seconds':time.perf_counter()-start,'baseline':metrics(hbase,y[ho],h[ho])}),flush=True)
 results=[{'view':'shared','l2':.1,'strength':0.,**metrics(hbase,y[ho],h[ho])}]
 for view in ['shared','joint']:
  xx=x[:,-3:] if view=='shared' else x
  for l2 in [.001,.01,.1]:
   w=fit(xx[rows(tr)],ibase,y[tr],l2);delta=(xx[rows(ho)]@w).reshape(-1,4)
   for strength in [.25,.5,1.]:
    r={'view':view,'l2':l2,'strength':strength,**metrics(hbase+strength*delta,y[ho],h[ho])};results.append(r);print(json.dumps(r),flush=True)
 best=max(results,key=lambda r:(r['hard_correct'],r['correct'],r['l2'],-r['strength'],r['view']=='shared'))
 (out/'internal_search.json').write_text(json.dumps(results,indent=2)+'\n')
 (out/'training_plan.json').write_text(json.dumps({'selected':best,'baseline':metrics(hbase,y[ho],h[ho]),'outer_holdout_labels_excluded_from_fits':True,'baseline_crossfit_cache':'solution_v7/runs/internal_features.joblib','candidate_strengths':[.25,.5,1],'candidate_l2':[.001,.01,.1],'test_used':False},indent=2)+'\n')
 xx=x[:,-3:] if best['view']=='shared' else x;w=fit(xx,fbase,y,best['l2'])
 bundle=out/'model_bundle';bundle.mkdir(exist_ok=True)
 shutil.copytree(root.parent/'solution_v6/runs/model_bundle',bundle/'base',dirs_exist_ok=True)
 np.savez(bundle/'target.npz',weight=w,view=np.array(best['view']),strength=np.array(best['strength']))
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb)
 base=base_submission.score_components(vb,va,bundle/'base')['selected'];vx=features(vb,best['view']);delta=(vx@w).reshape(-1,4);selected=base+best['strength']*delta
 np.savez(out/'validation_scores.npz',base=base,delta=delta,selected=selected)
 result={'baseline':metrics(base,vy,vh),'selected':metrics(selected,vy,vh),'recipe':best,'seconds':time.perf_counter()-start}
 (out/'validation_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 changes=[]
 for i in np.flatnonzero(base.argmax(1)!=selected.argmax(1)):
  old=int(base[i].argmax());new=int(selected[i].argmax());changes.append({'row':int(i),'old':old,'new':new,'old_correct':bool(vy[i,old]),'new_correct':bool(vy[i,new])})
 (out/'validation_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
if __name__=='__main__':main()
