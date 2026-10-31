"""Nested training holdout chooses a small repair-evidence ranking correction."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl,parse_answer
from features import features,labels,lexical_texts
from calculation_evidence import evidence_features
from legacy_crossfit import fit
from legacy_submission import member_scores

def hard(b):return np.array([len({a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None})>1 for r in b])
def metrics(s,y,h):
 c=y[np.arange(len(y)),s.argmax(1)];return {'correct':int(c.sum()),'n':len(y),'accuracy':float(c.mean()*100),'hard_correct':int(c[h].sum()),'hard_n':int(h.sum()),'hard_accuracy':float(c[h].mean()*100)}

def fit_correction(x,base,y,l2,columns):
 xx=x[:,:,columns];sd=xx.reshape(-1,len(columns)).std(0).clip(.05);xx=xx/sd;dx=[];db=[];weights=[]
 for i,t in enumerate(y):
  pos=np.flatnonzero(t);neg=np.flatnonzero(t==0)
  for p in pos:
   for n in neg:dx.append(xx[i,p]-xx[i,n]);db.append(base[i,p]-base[i,n]);weights.append(1/(len(pos)*len(neg)))
 dx=np.array(dx);db=np.array(db);weights=np.array(weights);weights/=weights.sum()
 def objective(theta):
  w=theta[:-1];temp=theta[-1];margin=dx@w+db*temp;g=-weights*expit(-margin)
  loss=np.sum(weights*np.logaddexp(0,-margin))+.5*l2*np.dot(w,w)+.01*(temp-.5)**2
  grad=np.r_[dx.T@g+l2*w,np.dot(db,g)+.02*(temp-.5)]
  return loss,grad
 res=minimize(objective,np.r_[np.zeros(len(columns)),.5],jac=True,method='L-BFGS-B',bounds=[(None,None)]*len(columns)+[(.05,5)],options={'maxiter':500,'ftol':1e-11})
 if not res.success:raise RuntimeError(res.message)
 return {'weight':res.x[:-1],'temperature':np.array(res.x[-1]),'sd':sd,'columns':np.array(columns),'l2':np.array(l2)}
def apply(x,base,state,strength=1):return base*state['temperature']+strength*(x[:,:,state['columns']]/state['sd']@state['weight'])

def main():
 torch.set_num_threads(4);root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data';start=time.perf_counter()
 b,ar=load_inputs('train',data);e=ar['embeddings'];y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);ev,details=evidence_features(b);np.savez_compressed(out/'train_evidence.npz',features=ev,labels=y,hard=h)
 (out/'train_evidence_details.json').write_text(json.dumps(details)+'\n')
 f=features(b,e);cx=np.concatenate([e,e-e.mean(1,keepdims=True),f],-1);hv=HashingVectorizer(n_features=32768,ngram_range=(1,2),alternate_sign=True,norm='l2');text=hv.transform(lexical_texts(b,True));row_ids=lambda ids:(np.array(ids)[:,None]*4+np.arange(4)).ravel()
 def baseline(tr,ho):
  context=fit(cx[tr],y[tr],cx[ho],17,.01,30,[10,30]);emb=fit(e[tr],y[tr],e[ho],29,.1,40,[40]);lex=Ridge(alpha=10,solver='lsqr').fit(text[row_ids(tr)],y[tr].ravel()).predict(text[row_ids(ho)]).reshape(-1,4)
  return (context[10]+context[30]+emb[40]+lex)/4
 tr,ho=train_test_split(np.arange(len(b)),test_size=.2,random_state=20261019,stratify=h)
 nested=out/'nested_base.npz'
 if nested.exists():
  with np.load(nested) as z:
   np.testing.assert_array_equal(tr,z['train_ids']);np.testing.assert_array_equal(ho,z['holdout_ids']);inner=z['inner'];outer=z['outer']
 else:
  inner=np.zeros((len(tr),4))
  for k,(ti,hi) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=20261020).split(tr,h[tr])):
   inner[hi]=baseline(tr[ti],tr[hi]);print(json.dumps({'inner_fold':k,'done':len(hi)}),flush=True)
  outer=baseline(tr,ho);np.savez_compressed(nested,train_ids=tr,holdout_ids=ho,inner=inner,outer=outer)
 records=[];options=[];baseline_result=metrics(outer,y[ho],h[ho]);print(json.dumps({'nested_holdout_baseline':baseline_result}),flush=True)
 for gate in ['group','local']:
  candidate_ev=ev.copy()
  if gate=='local':
   relevant=(ev[:,:,0]+ev[:,:,1]+ev[:,:,4]+ev[:,:,9])>0;candidate_ev[:,:,:14]*=relevant[:,:,None]
  for view,columns in [('repair_only',[0,1,8,9,10,11]),('all',list(range(14)))]:
   for l2 in [.0001,.001,.01,.1]:
    st=fit_correction(candidate_ev[tr],inner,y[tr],l2,columns)
    for strength in [.25,.5,1.]:
     ss=apply(candidate_ev[ho],outer,st,strength);rr={'gate':gate,'view':view,'columns':columns,'l2':l2,'strength':strength,**metrics(ss,y[ho],h[ho])};records.append(rr);options.append(rr)
 best=max(options,key=lambda r:(r['hard_correct'],r['correct'],-len(r['columns']),r['l2'],-r['strength'],r['gate']=='local'))
 (out/'internal_search.json').write_text(json.dumps(records,indent=2)+'\n');(out/'training_plan.json').write_text(json.dumps({'selected':best,'baseline':baseline_result,'outer_split_seed':20261019,'inner_split_seed':20261020,'nested_holdout':True,'every_inner_base_prediction_omits_its_question':True,'outer_holdout_labels_excluded_from_all_fits':True},indent=2)+'\n');print(json.dumps({'plan':best}),flush=True)
 # All-train cross-fitted legacy scores generated by the exact same frozen recipe.
 old=root.parent/'solution_v4/runs/oof/oof_scores.npz'
 with np.load(old) as z:oof=z['scores'];np.testing.assert_array_equal(y,z['labels'])
 if best['gate']=='local':
  relevant=(ev[:,:,0]+ev[:,:,1]+ev[:,:,4]+ev[:,:,9])>0;ev[:,:,:14]*=relevant[:,:,None]
 state=fit_correction(ev,oof,y,best['l2'],best['columns']);state['local_gate']=np.array(best['gate']=='local');state['strength']=np.array(best['strength']);np.savez(out/'correction.npz',**state)
 # Diagnostic train reliability; does not select on official validation labels.
 train_stats={'repair_source_candidates':0,'matching_repair_sources':0,'matched_repair_correct':0,'selected_repair_matches':0,'selected_repair_fixes':0,'selected_repair_breaks':0,'box_mismatch_candidates':0}
 for i,aa in enumerate(details):
  chosen=int(oof[i].argmax())
  for j,z in enumerate(aa):
   train_stats['box_mismatch_candidates']+=int(z['box_mismatch']);target=z['repaired_answer']
   if target is None:continue
   train_stats['repair_source_candidates']+=1;matches=[k for k,a in enumerate(aa) if k!=j and a['answer']==target]
   if matches:
    train_stats['matching_repair_sources']+=1;correct=bool(y[i,matches[0]]);train_stats['matched_repair_correct']+=correct
    if j==chosen:
     train_stats['selected_repair_matches']+=1;train_stats['selected_repair_fixes']+=int(not y[i,j] and correct);train_stats['selected_repair_breaks']+=int(y[i,j] and not correct)
 bv,av=load_inputs('validation',data);yv=labels(bv,read_jsonl(data/'gsm8k_validation.jsonl'));hh=hard(bv);xv,_=evidence_features(bv,local_gate=best['gate']=='local');base=np.mean(member_scores(bv,av,root.parent/'solution/runs/model_bundle'),0);s=apply(xv,base,state,best['strength']);result={'baseline':metrics(base,yv,hh),'correction':metrics(s,yv,hh),'training_evidence':train_stats,'training_seconds':time.perf_counter()-start,'test_access':'none','oof_provenance_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
 np.savez_compressed(out/'validation_scores.npz',baseline=base,corrected=s,labels=yv,hard=hh,features=xv);(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
