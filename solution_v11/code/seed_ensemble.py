"""Reduce seed variance; reuse saved paired-split baseline predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import train_test_split
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,fit,stats

def main():
 torch.set_num_threads(4);root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=np.concatenate([e,e-e.mean(1,keepdims=True),features(b,e)],-1);out=[];start=time.perf_counter()
 for split in [20261023,20261024,20261025,20261028,20261029,20261030]:
  tr,ho=train_test_split(np.arange(len(b)),test_size=.2,random_state=split,stratify=h)
  with np.load(root/'runs'/f'ensemble-{split}.npz') as z:c0=z['c0'];c1=z['c1'];e0=z['e0'];lex=z['lex'];np.testing.assert_array_equal(ho,z['holdout'])
  ca=[c0.mean(0)];ch=[c1.mean(0)]
  for seed in [29,41,53]:
   cache=root/'runs'/f'seed-{split}-{seed}.npz'
   if cache.exists():
    with np.load(cache) as z:aa=z['all'];hh=z['hard']
   else:
    a=fit(cx[tr],y[tr],np.ones(len(tr),bool),cx[ho],seed,.01,30,[10,30]);v=fit(cx[tr],y[tr],h[tr],cx[ho],seed,.01,30,[10,30]);aa=(a[10]+a[30])/2;hh=(v[10]+v[30])/2;np.savez_compressed(cache,all=aa,hard=hh)
   ca.append(aa);ch.append(hh)
  ca=np.mean(ca,axis=0);ch=np.mean(ch,axis=0);base=(c0.sum(0)+e0+lex)/4
  options={'base':base,'many_all':(2*ca+e0+lex)/4,'many_hard':(2*ch+e0+lex)/4,'many_mixed':(ca+ch+e0+lex)/4,'conservative_mixed':((ca+ch)/2+c0.mean(0)+e0+lex)/4}
  for k,s in options.items():out.append({'split':split,'method':k,**stats(s,y[ho],h[ho],base)})
  print(json.dumps({'split':split,'results':out[-5:],'elapsed':time.perf_counter()-start}),flush=True)
  (root/'runs/seed_results.json').write_text(json.dumps({'records':out,'official_validation_test_used':False},indent=2)+'\n')
if __name__=='__main__':main()
