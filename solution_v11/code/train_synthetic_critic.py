"""Crossfit graph semantics with reference-derived structural counterexamples."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from program_path_critic import H,representation,matrices
from synthetic_programs import generate
from sklearn.preprocessing import normalize

def fit(x,xr,xn,synthidx,graphs,y,ids,l2,synthetic_weight):
 pos=[];neg=[];rp=[];rn=[];sp=[];sn=[];weights=[];rweights=[];sweights=[];ref_present=xr.getnnz(1)>0
 for i in ids:
  p=[j for j in range(4) if y[i,j] and graphs[i][j]['available']];n=[j for j in range(4) if not y[i,j] and graphs[i][j]['available']]
  for a in p:
   for z in n:pos.append(i*4+a);neg.append(i*4+z);weights.append(1/max(1,len(p)*len(n)))
  if ref_present[i]:
   for z in n:rp.append(i);rn.append(i*4+z);rweights.append(.35/max(1,len(n)))
   for j in synthidx[i]:sp.append(i);sn.append(j);sweights.append(synthetic_weight/max(1,len(synthidx[i])))
 chunks=[]
 if pos:chunks.append(x[pos]-x[neg])
 if rp:chunks.append(xr[rp]-x[rn])
 if sp:chunks.append(xr[sp]-xn[sn])
 dx=sparse.vstack(chunks,format='csr');ww=np.array(weights+rweights+sweights);ww/=ww.sum()
 def loss(w):
  m=dx@w;g=-ww*expit(-m);return float(ww@np.logaddexp(0,-m)+.5*l2*(w@w)),np.asarray(dx.T@g).ravel()+l2*w
 res=minimize(loss,np.zeros(dx.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':140,'ftol':1e-8,'gtol':1e-5,'maxcor':8})
 if not res.success:raise RuntimeError(str(res.message))
 return res.x

def main():
 p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b);graphs=joblib.load(out/'path_graphs.joblib');x,xr=joblib.load(out/'path_critic_features.joblib');cache=out/'synth_graph_features.joblib'
 if cache.exists():xn,synthidx=joblib.load(cache)
 else:
  dictionaries=[];synthidx=[]
  for row,ref in zip(b,refs):
   refgraph,negative=generate(row,ref);ids=[]
   for g in negative:ids.append(len(dictionaries));dictionaries.append(representation(row,g))
   synthidx.append(ids)
  xn=normalize(H.transform(dictionaries),norm='l2',copy=False);joblib.dump((xn,synthidx),cache)
 print(json.dumps({'synthetic_programs':xn.shape[0],'questions_with_counterexamples':sum(bool(v) for v in synthidx),'seconds':time.perf_counter()-tick}),flush=True)
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261111).split(np.arange(len(b)),h));records=[];oofs={}
 for wt in [.5,1.]:
  for l2 in [.001,.01,.1]:
   raw=np.zeros((len(b),4))
   for k,(ti,hi) in enumerate(folds):
    w=fit(x,xr,xn,synthidx,graphs,y,ti,l2,wt);ids=(hi[:,None]*4+np.arange(4)).ravel();raw[hi]=np.asarray(x[ids]@w).reshape(-1,4)
   available=(x.getnnz(1)>0).reshape(-1,4);mean=(raw*available).sum(1,keepdims=True)/available.sum(1,keepdims=True).clip(1);raw=(raw-mean)*available;oofs[f'{wt}_{l2}']=raw;scale=(base-base.mean(1,keepdims=True)).std()/raw.std().clip(.01)
   for strength in [.1,.25,.5,1.]:records.append({'synthetic_weight':wt,'l2':l2,'strength':strength,'scale':float(scale),**stats(base+strength*scale*raw,y,h,base)})
   print(json.dumps({'weight':wt,'l2':l2,'seconds':time.perf_counter()-tick}),flush=True)
 records.append({'synthetic_weight':0.,'l2':1.,'strength':0.,'scale':0.,**stats(base,y,h,base)});best=max(records,key=lambda r:(r['hard_correct'],-r['strength'],r['l2']));(out/'synthetic_search.json').write_text(json.dumps({'selected':best,'records':records,'fully_nested':False},indent=2)+'\n');np.savez_compressed(out/'synthetic_oof.npz',**oofs);print(json.dumps({'selected':best}),flush=True)
 if not best['strength']:return
 w=fit(x,xr,xn,synthidx,graphs,y,np.arange(len(b)),best['l2'],best['synthetic_weight']);vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb)
 from quantity_graphs import candidate_graph
 vg=[[candidate_graph(r,c['text']) for c in r['candidates']] for r in vb];vx,_=matrices(vb,vg);raw=np.asarray(vx@w).reshape(-1,4);available=(vx.getnnz(1)>0).reshape(-1,4);mean=(raw*available).sum(1,keepdims=True)/available.sum(1,keepdims=True).clip(1);raw=(raw-mean)*available
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:basev=z['selected']
 res={'selected':best,'validation_base':stats(basev,vy,vh,basev),'validation_candidate':stats(basev+best['strength']*best['scale']*raw,vy,vh,basev),'seconds':time.perf_counter()-tick};np.savez(out/'synthetic_model.npz',weight=w,scale=best['scale'],strength=best['strength']);np.savez(out/'synthetic_validation_scores.npz',raw=raw);(out/'synthetic_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
