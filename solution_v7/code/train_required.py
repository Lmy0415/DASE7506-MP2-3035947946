"""Matched binary-required-quantity ablation, with train-only nested selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from quantity_relations import candidate_features
from relation_models import fit_models,predictions
from train_evidence import fit_correction,apply,metrics,hard
from base_submission import score_components

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data';start=time.perf_counter();b,ar=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b);graphs=joblib.load(out/'graphs.joblib')
 with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
 _,ibase,_,hbase=joblib.load(out/'internal_features.joblib');_,fbase=joblib.load(out/'full_features.joblib')
 def cross(ids,seed):
  rr=[None]*len(ids);pp=np.zeros((len(ids),9))
  for fold,(ti,hi) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(ids,h[ids])):
   m=fit_models([b[i] for i in ids[ti]],[refs[i] for i in ids[ti]],binary=True);rp,pr=predictions(m,[b[i] for i in ids[hi]])
   for k,r in zip(hi,rp):rr[k]=r
   pp[hi]=pr;print(json.dumps({'binary_crossfit':len(ids),'fold':fold}),flush=True)
  return candidate_features([b[i] for i in ids],rr,pp,[graphs[i] for i in ids])[0]
 ix=cross(tr,20261020);m=fit_models([b[i] for i in tr],[refs[i] for i in tr],binary=True);rp,pp=predictions(m,[b[i] for i in ho]);hx,_=candidate_features([b[i] for i in ho],rp,pp,[graphs[i] for i in ho]);records=[]
 for l2 in [.001,.01,.1]:
  state=fit_correction(ix,ibase,y[tr],l2,list(range(7)))
  for strength in [.25,.5,1.]:records.append({'head':'binary_required','columns':list(range(7)),'l2':l2,'strength':strength,**metrics(apply(hx,hbase,state,strength),y[ho],h[ho])})
 best=max(records,key=lambda r:(r['hard_correct'],r['correct'],r['l2'],-r['strength']));(out/'binary_training_plan.json').write_text(json.dumps({'selected':best,'internal_search':records,'baseline':metrics(hbase,y[ho],h[ho])},indent=2)+'\n');print(json.dumps({'binary_plan':best}),flush=True)
 fx=cross(np.arange(len(b)),20261021);state=fit_correction(fx,fbase,y,best['l2'],best['columns']);state['strength']=np.array(best['strength']);np.savez(out/'binary_correction.npz',**state);model=fit_models(b,refs,binary=True);joblib.dump(model,out/'binary_quantity_model.joblib',compress=0)
 bv,av=load_inputs('validation',data);yv=labels(bv,read_jsonl(data/'gsm8k_validation.jsonl'));hh=hard(bv);rv,pv=predictions(model,bv);vx,gg=candidate_features(bv,rv,pv);base=score_components(bv,av,root.parent/'solution_v6/runs/model_bundle')['selected'];ss=apply(vx,base,state,best['strength']);result={'v6_baseline':metrics(base,yv,hh),'binary_required':metrics(ss,yv,hh),'training_seconds':time.perf_counter()-start,'test_access':'none','predefined_strengths':[{'scale':scale,**metrics(apply(vx,base,state,best['strength']*scale),yv,hh)} for scale in [0,.25,.5,1.]]};np.savez_compressed(out/'binary_validation_scores.npz',baseline=base,required=ss,features=vx,labels=yv,hard=hh);(out/'BINARY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
