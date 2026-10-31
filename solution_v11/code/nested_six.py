"""Fit all six expert features without using outer holdout labels.
Hyperparameters were selected in prior train OOF experiments, so this is a
holdout-of-fit check, not a pristine independent model-selection test.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time,joblib,sys
from pathlib import Path
import numpy as np,torch
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,stats
from submission import constrain
from multi_gate_more import make,fit_predict,metric
from train_program_critic import fit as critic_fit
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'solution_v10/code'))
from rank_fit import fit
sys.path.append(str(Path(__file__).resolve().parents[2]/'solution_v6/code'))
from train_evidence import fit_correction,apply

def flat(x):return sparse.csr_matrix(x.reshape(-1,x.shape[-1]))
def cross_delta(x,base,y,h,l2,outerx,seed):
 n=len(y);d=np.zeros((n,4))
 for ti,hi in StratifiedKFold(5,shuffle=True,random_state=seed).split(np.arange(n),h):
  sd=x[ti].reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flat(x[ti]/sd),base[ti],y[ti],l2);d[hi]=(x[hi]/sd)@w
 sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flat(x/sd),base,y,l2);return d,(outerx/sd)@w

def main():
 torch.set_num_threads(4);p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];hand=features(b,e)
 with np.load(p/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
 ix,ibase,hx,hbase=joblib.load(p/'solution_v7/runs/internal_features.joblib');tb=[b[i] for i in tr];hb=[b[i] for i in ho];yt=y[tr];ht=h[tr];yh=y[ho];hh=h[ho]
 with np.load(out/'nested_independent.npz') as z:ind=z['inner'];hind=z['outer'];np.testing.assert_array_equal(tr,z['train_ids']);np.testing.assert_array_equal(ho,z['holdout_ids'])
 tx=joblib.load(p/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37);d,hd=cross_delta(tx[tr],ibase,yt,ht,.001,tx[ho],20261026);base=constrain(tb,ibase+d);hbase9=constrain(hb,hbase+hd);print(json.dumps({'v9_baseline':stats(hbase9,yh,hh,hbase9),'elapsed':time.perf_counter()-tick}),flush=True)
 # Role model predictions were crossfit solely within outer training and then
 # fitted on outer training to produce the outer holdout features.
 role=np.zeros_like(ibase)
 for ti,vi in StratifiedKFold(5,shuffle=True,random_state=20261113).split(np.arange(len(tr)),ht):
  state=fit_correction(ix[ti],ibase[ti],yt[ti],.01,list(range(7)));role[vi]=apply(ix[vi],ibase[vi],state)
 state=fit_correction(ix,ibase,yt,.01,list(range(7)));hrole=apply(hx,hbase,state)
 # Independently centered embedding ridge expert.
 cx=e-e.mean(1,keepdims=True);lin=np.zeros_like(ibase);hl=[]
 for ti,vi in StratifiedKFold(5,shuffle=True,random_state=20261114).split(np.arange(len(tr)),ht):
  xx=cx[tr[ti]];sd=xx.reshape(-1,xx.shape[-1]).std(0).clip(.05);model=Ridge(alpha=100.,fit_intercept=False,solver='lsqr',tol=1e-4).fit((xx/sd).reshape(-1,xx.shape[-1]),(yt[ti]-yt[ti].mean(1,keepdims=True)).ravel());lin[vi]=model.predict((cx[tr[vi]]/sd).reshape(-1,cx.shape[-1])).reshape(-1,4);hl.append(model.predict((cx[ho]/sd).reshape(-1,cx.shape[-1])).reshape(-1,4))
 scale=(base-base.mean(1,keepdims=True)).std()/lin.std();lin=base+.25*scale*lin;hlin=hbase9+.25*scale*np.mean(hl,0)
 # Original graph critic, fitted only to outer-training references and candidates.
 graphs=joblib.load(p/'solution_v7/runs/graphs.joblib');x,xr=joblib.load(out/'critic_features.joblib');present=xr.getnnz(1)>0;raw=np.zeros_like(ibase)
 for ti,vi in StratifiedKFold(5,shuffle=True,random_state=20261115).split(np.arange(len(tr)),ht):
  w=critic_fit(x,xr,graphs,present,y,tr[ti],.1);ids=(tr[vi,None]*4+np.arange(4)).ravel();raw[vi]=np.asarray(x[ids]@w).reshape(-1,4)
 w=critic_fit(x,xr,graphs,present,y,tr,.1);ids=(ho[:,None]*4+np.arange(4)).ravel();hraw=np.asarray(x[ids]@w).reshape(-1,4)
 def center(raw,ids):
  availability=(x.getnnz(1)>0).reshape(-1,4)[ids];mean=(raw*availability).sum(1,keepdims=True)/availability.sum(1,keepdims=True).clip(1);return (raw-mean)*availability
 raw=center(raw,tr);hraw=center(hraw,ho);scale=(base-base.mean(1,keepdims=True)).std()/raw.std().clip(.01);cri=base+.25*scale*raw;hcri=hbase9+.25*scale*hraw
 with np.load(out/'expanded_features.npz') as z:exp=z['train'][:,:,:12]
 d,hd=cross_delta(exp[tr],base,yt,ht,.1,exp[ho],20261116);exp=base+.25*d;hexp=hbase9+.25*hd
 models=[base,ind,lin,role,cri,exp];hmodels=[hbase9,hind,hlin,hrole,hcri,hexp];scales=[np.std(s-s.mean(1,keepdims=True)).clip(.05) for s in models]
 with np.load(p/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 dummy=np.zeros((len(tr),4,1));hdummy=np.zeros((len(ho),4,1));xx,ii,eligible=make(models,tb,hand[tr],calc[tr],dummy,dummy,'basic',scales);vx,vi,valid=make(hmodels,hb,hand[ho],calc[ho],hdummy,hdummy,'basic',scales);probs=[];idx=np.arange(len(tr))
 for seed in [201,202,203]:
  for ti,vi2 in StratifiedKFold(5,shuffle=True,random_state=seed).split(idx,ht):
   _,sc,model=fit_predict(xx,yt,ii,eligible,ht,ti,vi2,.1);vv=model.predict_proba(sc.transform(vx.reshape(-1,vx.shape[-1])))[:,1].reshape(len(ho),4);vv[~valid]=0;probs.append(vv)
 prob=np.mean(probs,0);baseline=metric(np.zeros_like(prob),1.,vi,yh,hh);candidate=metric(prob,.6,vi,yh,hh);record={'outer_train':len(tr),'outer_holdout':len(ho),'baseline_v9':baseline,'six_expert_candidate':candidate,'outer_holdout_labels_used_in_fit':False,'hyperparameters_previously_screened_on_full_train':True,'elapsed_seconds':time.perf_counter()-tick};(out/'nested_six.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
