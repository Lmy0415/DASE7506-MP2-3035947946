"""Coverage/reliability on held-out train predictions; no test reads."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,collections
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import accuracy_score
from common import load_inputs,read_jsonl
from features import labels
from rank_fit import metrics,hard
from train_patterns import paired
from local_relations import fit,pair_texts,distribution,H
from programs import question_numbers,induce
root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b)
with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
result={}
for name,folder,cache in [('retrieval','runs','retrieval_crossfit.joblib'),('local_relations','runs_local','local_crossfit.joblib')]:
 ix,hx,fx=joblib.load(root/folder/cache)
 with np.load(root/folder/'honest_scores.npz') as z:base=z['full_base']
 p=fx[:,:,0].argmax(1);best=fx[np.arange(len(p)),p];usable=best[:,0]>0
 result[name]={'reference_crossfit_coverage_questions':int(usable.sum()),'n':len(b),'coverage_percent':float(usable.mean()*100),'top_support_accuracy_when_usable':float(y[np.arange(len(p))[usable],p[usable]].mean()*100),'hard_coverage_questions':int((usable&h).sum()),'confidence_slices':[]}
 for cutoff in [.15,.3,.4,.5,.6,.7]:
  active=usable&(best[:,2]>=cutoff)&(best[:,4]>=.1)
  s=base.copy();s[active]=-100;s[np.flatnonzero(active),p[active]]=100
  result[name]['confidence_slices'].append({'cutoff':cutoff,'n':int(active.sum()),'precision':float(y[np.arange(len(p))[active],p[active]].mean()*100) if active.any() else None,**paired(base,s,y)})
 state=joblib.load(root/folder/'model_bundle/patterns.joblib')
 if name=='local_relations':
  total=0;aliases=0;groups=0
  for n,g in state['bins'].items():
   counter=collections.Counter(v.tobytes() for v in g['matrix']);total+=len(g['programs']);aliases+=sum(c for c in counter.values() if c>1);groups+=sum(c>1 for c in counter.values())
  result[name]['indistinguishable_programs_under_pair_features']={'total_programs':total,'programs_in_colliding_groups':aliases,'colliding_groups':groups}
# Auxiliary relation accuracy measured on the same strict outer holdout.
state=fit([b[int(i)] for i in tr],[refs[int(i)] for i in tr]);texts=[];truth=[]
for i in ho:
 row=b[int(i)];ref=refs[int(i)];vv,_=question_numbers(row['question'])
 if len(vv)!=len(set(vv)) or not 2<=len(vv)<=8:continue
 node=induce(row['question'],ref['answer'])
 if node is None:continue
 dd=distribution(node,len(vv));tt=pair_texts(row['question'])
 for text,probs in zip(tt,dd):
  if probs.max()==1:texts.append(text);truth.append(int(probs.argmax()))
pred=state['model'].predict(H.transform(texts));counter=collections.Counter(truth)
result['local_relations']['strict_heldout_pair_task']={'n_unambiguous_pairs':len(truth),'accuracy_percent':float(accuracy_score(truth,pred)*100),'majority_percent':100*max(counter.values())/len(truth),'label_counts':dict(counter)}
(root/'runs/diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
