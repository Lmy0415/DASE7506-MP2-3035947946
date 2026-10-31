"""Reference-label holdout diagnostics and a no-reference-relevance ablation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import confusion_matrix,classification_report
from common import load_inputs,read_jsonl
from features import labels
from relation_models import fit_models,predictions
from quantity_relations import reference_targets
from train_evidence import fit_correction,apply,metrics,hard
root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data';b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b)
with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
m=fit_models([b[i] for i in tr],[refs[i] for i in tr]);rr,pp=predictions(m,[b[i] for i in ho]);gold,profile=reference_targets([b[i] for i in ho],[refs[i] for i in ho]);true=[];pred=[];root_true=[];root_pred=[]
for r,p,g,pf in zip(rr,pp,gold,profile):
 if g is not None:
  valid=g>=0;true.extend(g[valid].tolist());pred.extend(r.argmax(1)[valid].tolist())
 if pf is not None:root_true.append(int(pf[:5].argmax()));root_pred.append(int(p[:5].argmax()))
quality={'quantity_role_confusion_matrix':confusion_matrix(true,pred,labels=[0,1,2]).tolist(),'label_order':['unused','increases_answer','decreases_answer'],'quantity_role_accuracy':float(np.mean(np.array(true)==pred)*100),'n':len(true),'majority_positive_accuracy':float(np.mean(np.array(true)==1)*100),'goal_root_operation_accuracy':float(np.mean(np.array(root_true)==root_pred)*100),'goal_n':len(root_true),'report':classification_report(true,pred,labels=[0,1,2],output_dict=True,zero_division=0),'interpretation':'Agreement with reference-program-derived weak labels; not a human semantic correctness evaluation.'}
ix,ibase,hx,hbase=joblib.load(out/'internal_features.joblib');records=[]
for l2 in [.001,.01,.1]:
 st=fit_correction(ix,ibase,y[tr],l2,[0,1,2,3])
 for strength in [.25,.5,1.]:records.append({'view':'unweighted_ledger','columns':[0,1,2,3],'l2':l2,'strength':strength,**metrics(apply(hx,hbase,st,strength),y[ho],h[ho])})
best=max(records,key=lambda r:(r['hard_correct'],r['correct'],r['l2'],-r['strength']));fx,fbase=joblib.load(out/'full_features.joblib');state=fit_correction(fx,fbase,y,best['l2'],best['columns']);state['strength']=np.array(best['strength']);np.savez(out/'unweighted_correction.npz',**state)
with np.load(out/'validation_scores.npz') as z:vx=z['features'];base=z['baseline'];yv=z['labels'];hh=z['hard']
ss=apply(vx,base,state,best['strength']);quality['unweighted_ablation']={'internal_best':best,'validation':metrics(ss,yv,hh),'internal_search':records};np.savez_compressed(out/'unweighted_scores.npz',scores=ss)
(out/'diagnostics.json').write_text(json.dumps(quality,indent=2)+'\n');print(json.dumps(quality,indent=2))
