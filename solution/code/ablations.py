"""Matched linear-model ablations fitted on train only; no test access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import load_inputs,read_jsonl
from features import features,labels
from train import metrics
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
es=[];fs=[];ys=[]
for split in ['train','validation']:
 b,arr=load_inputs(split,a.data_dir);es.append(arr['embeddings']);fs.append(features(b,es[-1]));ys.append(labels(b,read_jsonl(a.data_dir/f'gsm8k_{split}.jsonl')))
with np.load(a.output/'development_labels.npz') as z:hard=z['hard']
result=[]
for name,xx in [
 ('absolute_embedding_plus_features',[np.concatenate([e,f],-1) for e,f in zip(es,fs)]),
 ('centered_embedding_plus_features',[np.concatenate([e,e-e.mean(1,keepdims=True),f],-1) for e,f in zip(es,fs)]),
 ('context_without_handcrafted',[np.concatenate([e,e-e.mean(1,keepdims=True)],-1) for e in es]),
 ('answer_consensus_only',[f[...,:3] for f in fs]),
 ('context_without_format_features',[np.concatenate([e,e-e.mean(1,keepdims=True),f[...,:14],f[...,52:66],f[...,-3:]],-1) for e,f in zip(es,fs)])]:
 t=time.perf_counter();mu=xx[0].reshape(-1,xx[0].shape[-1]).mean(0);sd=xx[0].reshape(-1,len(mu)).std(0).clip(.05)
 zz=[((x-mu)/sd).reshape(-1,len(mu)) for x in xx]
 m=LogisticRegression(C=.01,max_iter=400).fit(zz[0],ys[0].ravel());s=m.decision_function(zz[1]).reshape(-1,4)
 result.append({'name':name,**metrics(s,ys[1],hard),'seconds':time.perf_counter()-t});print(json.dumps(result[-1]),flush=True)
(a.output/'ablations.json').write_text(json.dumps(result,indent=2)+'\n')
