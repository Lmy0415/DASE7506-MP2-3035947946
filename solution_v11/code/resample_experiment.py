"""Question-preserving candidate resampling as group robustness training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import train_test_split
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard,stats

def train(x,y,xa,ya,v,seed):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu=x.reshape(-1,x.shape[-1]).mean(0);sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);xx=torch.tensor((np.concatenate([x,xa])-mu)/sd);yy=torch.tensor(np.concatenate([y,ya]));vx=torch.tensor((v-mu)/sd)
 net=torch.nn.Sequential(torch.nn.Linear(len(mu),96),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,1));opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.01);sn=[]
 for epoch in range(1,31):
  net.train()
  for idx in np.array_split(rng.permutation(len(yy)),20):
   s=net(xx[idx]).squeeze(-1);t=yy[idx];loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t==0,-1e9),1)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,t);opt.zero_grad();loss.backward();opt.step()
  if epoch in [10,30]:
   net.eval()
   with torch.no_grad():sn.append(net(vx).squeeze(-1).numpy())
 return np.mean(sn,axis=0)

def main():
 torch.set_num_threads(4);p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';start=time.perf_counter();b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];f=features(b,e);cx=np.concatenate([e,e-e.mean(1,keepdims=True),f],-1)
 cache=out/'resample_features.npz';ids=np.flatnonzero(h)
 if cache.exists():
  with np.load(cache) as z:perms=z['perms'];af=z['f'];np.testing.assert_array_equal(ids,z['ids'])
 else:
  rng=np.random.default_rng(20261108);perms=[];new=[]
  for i in ids:
   # Different possible answers and one correct candidate must remain.
   for trial in range(100):
    perm=rng.integers(0,4,4)
    if y[i,perm].sum()>0 and y[i,perm].sum()<4 and len(set(perm))>=2:break
   else:perm=np.arange(4)
   rr=dict(b[i]);rr['candidates']=[b[i]['candidates'][int(j)] for j in perm];new.append(rr);perms.append(perm)
  perms=np.asarray(perms);ae=e[ids[:,None],perms];af=features(new,ae);np.savez_compressed(cache,perms=perms,f=af,ids=ids)
 ae=e[ids[:,None],perms];acx=np.concatenate([ae,ae-ae.mean(1,keepdims=True),af],-1);ay=y[ids[:,None],perms];records=[]
 for seed in [20261023,20261024,20261025,20261028,20261029,20261030]:
  tr,ho=train_test_split(np.arange(len(b)),test_size=.2,random_state=seed,stratify=h);allowed=np.isin(ids,tr)
  with np.load(out/f'ensemble-{seed}.npz') as z:c0=z['c0'];e0=z['e0'];lex=z['lex']
  base=(c0.sum(0)+e0+lex)/4;path=out/f'resample-{seed}.npz'
  if path.exists():
   with np.load(path) as z:s=z['scores']
  else:s=train(cx[tr],y[tr],acx[allowed],ay[allowed],cx[ho],17);np.savez_compressed(path,scores=s)
  for alpha in [.25,.5,1.]:records.append({'split':seed,'strength':alpha,**stats(base+.5*alpha*(s-c0.mean(0)),y[ho],h[ho],base)})
  print(json.dumps({'split':seed,'results':records[-3:],'seconds':time.perf_counter()-start}),flush=True);(out/'resample_results.json').write_text(json.dumps({'records':records,'groups_from_train_only':True},indent=2)+'\n')
if __name__=='__main__':main()
