"""Crossfit distinct contrastive neural objectives as complementary experts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,stats
from loss_experiment import fit

def main():
 torch.set_num_threads(4);p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=np.concatenate([e,e-e.mean(1,keepdims=True),features(b,e)],-1)
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);ve=va['embeddings'];vcx=np.concatenate([ve,ve-ve.mean(1,keepdims=True),features(vb,ve)],-1)
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 cache=out/'neural_oof.npz';folds=list(StratifiedKFold(3,shuffle=True,random_state=20261106).split(np.arange(len(b)),h));preds={};results=[]
 if cache.exists():
  with np.load(cache) as z:preds={k:z[k] for k in z.files}
 else:
  for mode in ['equal','pair']:
   raw=np.zeros((len(b),4));val=[]
   for k,(ti,hi) in enumerate(folds):
    v=np.concatenate([cx[hi],vcx],axis=0);s=fit(cx[ti],y[ti],h[ti],v,17+k*12,mode);raw[hi]=s[:len(hi)];val.append(s[len(hi):]);print(json.dumps({'mode':mode,'fold':k,'seconds':time.perf_counter()-tick}),flush=True)
   preds[mode]=raw;preds[mode+'_val']=np.mean(val,0)
  np.savez_compressed(cache,**preds)
 for mode in ['equal','pair']:
  raw=preds[mode];scale=(base-base.mean(1,keepdims=True)).std()/(raw-raw.mean(1,keepdims=True)).std()
  for alpha in [.1,.25,.5,1.]:results.append({'mode':mode,'strength':alpha,'scale':float(scale),**stats(base+alpha*scale*raw,y,h,base)})
 results.append({'mode':'none','strength':0.,'scale':0.,**stats(base,y,h,base)});best=max(results,key=lambda r:(r['hard_correct'],-r['strength']));(out/'neural_search.json').write_text(json.dumps({'selected':best,'records':results,'train_only_crossfit':True,'fully_nested':False},indent=2)+'\n')
 if best['mode']=='none':print(json.dumps(best));return
 val=vbase+best['strength']*best['scale']*preds[best['mode']+'_val'];res={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(val,vy,vh,vbase),'seconds':time.perf_counter()-tick};(out/'neural_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
