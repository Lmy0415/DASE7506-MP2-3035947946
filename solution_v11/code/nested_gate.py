"""Strict outer holdout for a question-dependent two-expert gate.
No outer-holdout labels are used in either specialist or gate fitting.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,joblib
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard
from disagreement_gate import make,summarize
project=Path(__file__).resolve().parents[2];sys.path.insert(0,str(project/'training/independent_expert/code'))
from ranker import train_ranker

def main():
 root=project/'solution_v11';out=root/'runs';data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();bank,ar=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(bank,refs);h=hard(bank);hand=features(bank,ar['embeddings'])
 with np.load(project/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
 _,ibase,_,hbase=joblib.load(project/'solution_v7/runs/internal_features.joblib')
 with np.load(project/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 cache=project/'training/independent_expert/runs/cache/train_base_v3_features.npz'
 with np.load(cache) as z:xf=z['features']
 store=out/'nested_independent.npz'
 if store.exists():
  with np.load(store) as z:inner=z['inner'];outer=z['outer'];np.testing.assert_array_equal(tr,z['train_ids']);np.testing.assert_array_equal(ho,z['holdout_ids'])
 else:
  inner=np.zeros((len(tr),4));cnt=np.zeros(len(tr));outer=[];seeds=[17,29,41,59,73,83,97,109,127,139]
  for repeat in range(2):
   for fold,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=1729+repeat*1009).split(tr,h[tr])):
    seed=seeds[repeat*5+fold];model=train_ranker(xf,y,tr[ti],hidden_dim=256,seed=seed,epochs=50,learning_rate=.001,weight_decay=.0001,dropout=.1,objective='ce',cosine=True)
    inner[vi]+=model.scores(xf[tr[vi]]);cnt[vi]+=1;outer.append(model.scores(xf[ho]));print(json.dumps({'repeat':repeat,'fold':fold,'elapsed':time.perf_counter()-tick}),flush=True)
  inner/=cnt[:,None];outer=np.mean(outer,axis=0);np.savez_compressed(store,inner=inner,outer=outer,train_ids=tr,holdout_ids=ho)
 bt=[bank[i] for i in tr];bh=[bank[i] for i in ho];dummy=np.zeros((len(bank),4,1));scales=(np.std(ibase-ibase.mean(1,keepdims=True)).clip(.05),np.std(inner-inner.mean(1,keepdims=True)).clip(.05))
 x,ii,jj,d=make(ibase,inner,bt,hand[tr],calc[tr],dummy[tr],dummy[tr],'basic',scales);vx,vi,vj,vd=make(hbase,outer,bh,hand[ho],calc[ho],dummy[ho],dummy[ho],'basic',scales)
 ids=np.arange(len(tr));bc=y[tr][ids,ii].astype(bool);oc=y[tr][ids,jj].astype(bool);info=h[tr]&d&(bc!=oc);scaler=StandardScaler().fit(x[info]);clf=LogisticRegression(C=1.,max_iter=1000).fit(scaler.transform(x[info]),oc[info].astype(int));prob=clf.predict_proba(scaler.transform(vx))[:,1]
 baseline=summarize(np.zeros(len(ho),bool),vi,vj,y[ho],h[ho],vd);result=summarize(prob>=.55,vi,vj,y[ho],h[ho],vd)
 record={'outer_train':len(tr),'outer_holdout':len(ho),'gate_train_informative':int(info.sum()),'baseline':baseline,'candidate':result,'outer_holdout_excluded_from_every_fit':True,'decision_threshold':.55,'penalty_c':1.,'seconds':time.perf_counter()-tick};(out/'nested_gate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
