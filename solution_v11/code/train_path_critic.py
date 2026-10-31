"""Crossfit structured candidate-vs-reference program comparisons."""
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
from program_path_critic import matrices
from quantity_graphs import candidate_graph

def fit(x,xr,graphs,ref_present,y,ids,l2):
 pos=[];neg=[];rp=[];rn=[];weights=[];rweights=[]
 for i in ids:
  p=[j for j in range(4) if y[i,j] and graphs[i][j]['available']];n=[j for j in range(4) if not y[i,j] and graphs[i][j]['available']]
  for a in p:
   for z in n:pos.append(i*4+a);neg.append(i*4+z);weights.append(1/max(1,len(p)*len(n)))
  if ref_present[i]:
   for z in n:rp.append(i);rn.append(i*4+z);rweights.append(.35/max(1,len(n)))
 matrices=[]
 if pos:matrices.append(x[pos]-x[neg])
 if rp:matrices.append(xr[rp]-x[rn])
 dx=sparse.vstack(matrices,format='csr');ww=np.asarray(weights+rweights);ww/=ww.sum()
 def loss(w):
  m=dx@w;grad=-ww*expit(-m)
  return float(ww@np.logaddexp(0,-m)+.5*l2*(w@w)),np.asarray(dx.T@grad).ravel()+l2*w
 opt=minimize(loss,np.zeros(dx.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':130,'ftol':1e-8,'gtol':1e-5,'maxcor':8})
 if not opt.success:raise RuntimeError(str(opt.message))
 return opt.x

def main():
 project=Path(__file__).resolve().parents[2];root=project/'solution_v11';out=root/'runs';data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();bank,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(bank,refs);h=hard(bank);graphcache=out/'path_graphs.joblib'
 if graphcache.exists():graphs=joblib.load(graphcache)
 else:graphs=[[candidate_graph(r,c['text']) for c in r['candidates']] for r in bank];joblib.dump(graphs,graphcache)
 cache=out/'path_critic_features.joblib'
 if cache.exists():x,xr=joblib.load(cache)
 else:x,xr=matrices(bank,graphs,refs);joblib.dump((x,xr),cache)
 print(json.dumps({'candidates':x.shape,'reference_graphs':int((xr.getnnz(1)>0).sum()),'candidate_graphs':int((x.getnnz(1)>0).sum()),'seconds':time.perf_counter()-tick}),flush=True)
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261102).split(np.arange(len(bank)),h));rp=xr.getnnz(1)>0;results=[];oofs={}
 for l2 in [.001,.01,.1]:
  raw=np.zeros((len(bank),4))
  for fold,(ti,hi) in enumerate(folds):
   w=fit(x,xr,graphs,rp,y,ti,l2);rows=(hi[:,None]*4+np.arange(4)).ravel();raw[hi]=np.asarray(x[rows]@w).reshape(-1,4);print(json.dumps({'l2':l2,'fold':fold,'seconds':time.perf_counter()-tick}),flush=True)
  # Unavailable programs are unknown, not negative evidence; center valid-only.
  available=(x.getnnz(1)>0).reshape(-1,4);center=np.sum(raw*available,1,keepdims=True)/available.sum(1,keepdims=True).clip(1);raw=(raw-center)*available;oofs[l2]=raw
  scale=(base-base.mean(1,keepdims=True)).std()/raw.std().clip(.01)
  for alpha in [.1,.25,.5,1.]:results.append({'l2':l2,'strength':alpha,'scale':float(scale),**stats(base+alpha*scale*raw,y,h,base)})
 np.savez_compressed(out/'path_critic_oof.npz',**{str(k):v for k,v in oofs.items()})
 results.append({'l2':1.,'strength':0.,'scale':0.,**stats(base,y,h,base)});best=max(results,key=lambda z:(z['hard_correct'],-z['strength'],z['l2']));(out/'path_critic_search.json').write_text(json.dumps({'records':results,'selected':best,'crossfit_nested_training_library':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if not best['strength']:return
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vg=[[candidate_graph(r,c['text']) for c in r['candidates']] for r in vb];vx,_=matrices(vb,vg)
 w=fit(x,xr,graphs,rp,y,np.arange(len(bank)),best['l2']);vr=np.asarray(vx@w).reshape(-1,4);available=(vx.getnnz(1)>0).reshape(-1,4);center=np.sum(vr*available,1,keepdims=True)/available.sum(1,keepdims=True).clip(1);vr=(vr-center)*available
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 result={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(vbase+best['strength']*best['scale']*vr,vy,vh,vbase),'seconds':time.perf_counter()-tick};(out/'path_critic_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 np.savez(out/'path_critic_model.npz',weight=w,l2=best['l2'],strength=best['strength'],scale=best['scale'])
if __name__=='__main__':main()
