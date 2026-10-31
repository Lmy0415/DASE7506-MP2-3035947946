"""Crossfit question-to-relationship learning, no test data."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from two_number_solver import text_view,targets,score,template_values

def train(bank,refs,ids,C):
 texts=[];classes=[];weights=[]
 for i in ids:
  t=targets(bank[i],refs[i])
  for c in t:texts.append(text_view(bank[i]['question']));classes.append(c);weights.append(1/len(t))
 v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000,sublinear_tf=True,dtype=np.float64);x=v.fit_transform(texts)
 m=LogisticRegression(C=C,max_iter=1000,solver='lbfgs').fit(x,classes,sample_weight=weights)
 return v,m

def main():
 project=Path(__file__).resolve().parents[2];root=project/'solution_v11';out=root/'runs';data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b)
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 coverage=sum(bool(template_values(r['question'])) for r in b);usable=sum(bool(targets(r,ref)) for r,ref in zip(b,refs));print(json.dumps({'two_quantity':coverage,'train_recovered':usable}),flush=True)
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261104).split(np.arange(len(b)),h));results=[];saves={}
 for C in [.1,1.,10.]:
  oof=np.zeros((len(b),4))
  for k,(ti,hi) in enumerate(folds):
   v,m=train(b,refs,ti,C);oof[hi]=score([b[i] for i in hi],v,m);print(json.dumps({'C':C,'fold':k,'seconds':time.perf_counter()-tick}),flush=True)
  saves[str(C)]=oof
  for alpha in [.1,.25,.5,1.,2.]:results.append({'C':C,'strength':alpha,**stats(base+alpha*oof,y,h,base)})
 results.append({'C':1.,'strength':0.,**stats(base,y,h,base)});best=max(results,key=lambda z:(z['hard_correct'],-z['strength']));(out/'two_number_search.json').write_text(json.dumps({'selected':best,'records':results,'reference_coverage':usable,'candidate_question_coverage':coverage,'nested_outer_exclusion':False},indent=2)+'\n');np.savez_compressed(out/'two_number_oof.npz',**saves)
 if not best['strength']:print(json.dumps(best));return
 v,m=train(b,refs,np.arange(len(b)),best['C']);vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vs=score(vb,v,m)
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 result={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(vbase+best['strength']*vs,vy,vh,vbase),'seconds':time.perf_counter()-tick};(out/'two_number_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
