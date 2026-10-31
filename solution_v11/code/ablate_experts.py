"""Question-crossfitted leave-one-expert-out validation ablations.

This is development evidence, not fully nested. It changes gate inputs and
re-fits the gate while retaining the expert scores and recipe for the other
branches. Input data is read from the supplied train and validation files.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import sys,json,time,argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl,parse_answer
from features import features,labels
from ensemble_experiment import hard,stats
from gate_features import make
from multi_gate_more import fit_predict
from calculation_evidence import evidence_features
from final_submission import score_components

def chosen(bank,base,prob):
 out=base.copy();top=prob.argmax(1);active=prob.max(1)>=.6
 for i in np.flatnonzero(active):
  answer=parse_answer(bank[i]['candidates'][int(top[i])]['text'])
  for j,c in enumerate(bank[i]['candidates']):
   if answer is not None and parse_answer(c['text'])==answer:out[i,j]=base[i].max()+1.
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 p=Path(__file__).resolve().parents[2];t=time.perf_counter();b,a=load_inputs('train',args.data_dir);y=labels(b,read_jsonl(args.data_dir/'gsm8k_train.jsonl'));h=hard(b);idx=np.arange(len(b));hand=features(b,a['embeddings'])
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'training/independent_expert/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof']
 with np.load(p/'solution_v11/runs/linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100']
 with np.load(p/'solution_v11/runs/role_oof.npz') as z:role=z['train']
 with np.load(p/'solution_v11/runs/critic_oof.npz') as z:critic=base+.25*128.0030667*z['0.1']
 with np.load(p/'solution_v11/runs/expanded_oof.npz') as z:extra=base+.25*z['all_0.1']
 with np.load(p/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 models=[base,ind,lin,role,critic,extra];names=['base','independent','linear','role','critic','expanded']
 vb,va=load_inputs('validation',args.data_dir);vy=labels(vb,read_jsonl(args.data_dir/'gsm8k_validation.jsonl'));vh=hard(vb);vhnd=features(vb,va['embeddings']);vcalc=evidence_features(vb)[0]
 comp=score_components(vb,va,p/'solution_v11/runs/final_bundle');vbase=comp['base'];vmodels=[vbase,comp['independent'],comp['linear'],comp['role'],comp['critic'],comp['expanded']]
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261028).split(idx,h));saved={}
 for omit in [-1,1,2,3,4,5]:
  key='all_experts' if omit<0 else 'omit_'+names[omit];m=[s for i,s in enumerate(models) if i!=omit];vm=[s for i,s in enumerate(vmodels) if i!=omit];scales=np.asarray([np.std(s-s.mean(1,keepdims=True)).clip(.05) for s in m]);x,ii,eligible=make(m,b,hand,calc,scales);vx,vi,ve=make(vm,vb,vhnd,vcalc,scales);oof=np.zeros_like(base);vals=[]
  for tr,te in folds:
   oof[te],sc,clf=fit_predict(x,y,ii,eligible,h,tr,te,.1);vv=clf.predict_proba(sc.transform(vx.reshape(-1,vx.shape[-1])))[:,1].reshape(len(vb),4);vv[~ve]=0;vals.append(vv)
  saved[key]=(chosen(b,base,oof),chosen(vb,vbase,np.mean(vals,axis=0)));print(json.dumps({'variant':key,'elapsed_seconds':round(time.perf_counter()-t,2)}),flush=True)
 control,vcontrol=saved['all_experts'];result={'protocol':'paired leave-one-expert-out; each gate refit with same 5 question folds; test unused','baseline_expert_always_retained':True,'experts_not_fully_nested':True,'validation_used_during_development':True,'variants':{}}
 for name,(s,vs) in saved.items():result['variants'][name]={'training_oof':stats(s,y,h,control),'validation':stats(vs,vy,vh,vcontrol)}
 result['elapsed_seconds']=round(time.perf_counter()-t,2);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
