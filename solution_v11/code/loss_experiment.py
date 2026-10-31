"""Paired multi-split candidate-ranking objective tests, no validation/test."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import train_test_split
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,stats

def fit(x,y,mask,v,seed,loss_name):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu=x.reshape(-1,x.shape[-1]).mean(0);sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);xx=torch.tensor(((x-mu)/sd)[mask]);yy=torch.tensor(y[mask]);vx=torch.tensor((v-mu)/sd)
 net=torch.nn.Sequential(torch.nn.Linear(len(mu),96),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,1));opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.01);out=[]
 for epoch in range(1,31):
  net.train()
  for idx in np.array_split(rng.permutation(len(yy)),20):
   s=net(xx[idx]).squeeze(-1);t=yy[idx]
   if loss_name=='pair':
    pairs=t[:,:,None]*(1-t[:,None,:]);diff=s[:,None,:]-s[:,:,None];n=pairs.sum((1,2)).clamp(min=1);loss=(torch.nn.functional.softplus(diff)*pairs).sum((1,2)).div(n).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,t)
   else:
    logp=torch.nn.functional.log_softmax(s,dim=1);loss=-(logp*t).sum(1).div(t.sum(1).clamp(min=1)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,t)
   opt.zero_grad();loss.backward();opt.step()
  if epoch in [10,30]:
   net.eval()
   with torch.no_grad():out.append(net(vx).squeeze(-1).numpy())
 return np.mean(out,axis=0)

def main():
 torch.set_num_threads(4);p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';d=p/'DASE7506-MP2-main/code/data';b,ar=load_inputs('train',d);y=labels(b,read_jsonl(d/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=np.concatenate([e,e-e.mean(1,keepdims=True),features(b,e)],-1);records=[];tick=time.perf_counter()
 for seed in [20261023,20261024,20261025,20261028,20261029,20261030]:
  tr,ho=train_test_split(np.arange(len(b)),test_size=.2,random_state=seed,stratify=h)
  with np.load(out/f'ensemble-{seed}.npz') as z:c0=z['c0'];e0=z['e0'];lex=z['lex']
  base=(c0.sum(0)+e0+lex)/4;cache=out/f'loss-{seed}.npz'
  if cache.exists():
   with np.load(cache) as z:pair=z['pair'];equal=z['equal']
  else:
   pair=fit(cx[tr],y[tr],h[tr],cx[ho],17,'pair');equal=fit(cx[tr],y[tr],h[tr],cx[ho],17,'equal');np.savez_compressed(cache,pair=pair,equal=equal)
  for name,s in [('pair',pair),('equal',equal)]:
   for a in [.25,.5,1.]:
    pred=base+.5*a*(s-c0.mean(0));records.append({'split':seed,'loss':name,'strength':a,**stats(pred,y[ho],h[ho],base)})
  print(json.dumps({'split':seed,'records':records[-6:],'seconds':time.perf_counter()-tick}),flush=True);(out/'loss_results.json').write_text(json.dumps({'records':records,'test_used':False},indent=2)+'\n')
if __name__=='__main__':main()
