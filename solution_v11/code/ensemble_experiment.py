"""Paired multi-split test of disagreement sampling in the whole base ensemble."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl,parse_answer
from features import features,labels,lexical_texts

def hard(b):return np.array([len({a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None})>1 for r in b])
def fit(x,y,mask,v,seed,decay,end,checkpoints):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu=x.reshape(-1,x.shape[-1]).mean(0);sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);xx=torch.tensor(((x-mu)/sd)[mask]);yy=torch.tensor(y[mask]);vx=torch.tensor((v-mu)/sd)
 net=torch.nn.Sequential(torch.nn.Linear(len(mu),96),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,1));opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=decay);out={}
 for epoch in range(1,end+1):
  net.train()
  for idx in np.array_split(rng.permutation(len(yy)),20):
   s=net(xx[idx]).squeeze(-1);t=yy[idx];loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t==0,-1e9),1)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,t);opt.zero_grad();loss.backward();opt.step()
  if epoch in checkpoints:
   net.eval()
   with torch.no_grad():out[epoch]=net(vx).squeeze(-1).numpy()
 return out

def stats(s,y,h,base):
 c=y[np.arange(len(y)),s.argmax(1)].astype(bool);b=y[np.arange(len(y)),base.argmax(1)].astype(bool)
 return {'correct':int(c.sum()),'hard_correct':int(c[h].sum()),'hard_n':int(h.sum()),'hard_fixes':int((h&~b&c).sum()),'hard_breaks':int((h&b&~c).sum())}
def main():
 torch.set_num_threads(4);root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=np.concatenate([e,e-e.mean(1,keepdims=True),features(b,e)],-1)
 text=HashingVectorizer(n_features=32768,ngram_range=(1,2),alternate_sign=True,norm='l2').transform(lexical_texts(b,True));idx=lambda a:(a[:,None]*4+np.arange(4)).ravel();results=[];start=time.perf_counter()
 for seed in [20261023,20261024,20261025,20261028,20261029,20261030]:
  tr,ho=train_test_split(np.arange(len(b)),test_size=.2,random_state=seed,stratify=h);cache=root/'runs'/f'ensemble-{seed}.npz'
  if cache.exists():
   with np.load(cache) as z:c0=z['c0'];c1=z['c1'];e0=z['e0'];e1=z['e1'];lex=z['lex'];np.testing.assert_array_equal(ho,z['holdout'])
  else:
   ca=fit(cx[tr],y[tr],np.ones(len(tr),bool),cx[ho],17,.01,30,[10,30]);ch=fit(cx[tr],y[tr],h[tr],cx[ho],17,.01,30,[10,30]);ea=fit(e[tr],y[tr],np.ones(len(tr),bool),e[ho],29,.1,40,[40]);eh=fit(e[tr],y[tr],h[tr],e[ho],29,.1,40,[40]);lex=Ridge(alpha=10,solver='lsqr').fit(text[idx(tr)],y[tr].ravel()).predict(text[idx(ho)]).reshape(-1,4)
   c0=np.stack([ca[10],ca[30]]);c1=np.stack([ch[10],ch[30]]);e0=ea[40];e1=eh[40];np.savez_compressed(cache,c0=c0,c1=c1,e0=e0,e1=e1,lex=lex,holdout=ho)
  baseline=(c0.sum(0)+e0+lex)/4
  options={'base':baseline,'hard_context':(c1.sum(0)+e0+lex)/4,'mixed_context':((c0+c1).sum(0)/2+e0+lex)/4,'hard_context_embedding':(c1.sum(0)+e1+lex)/4,'mixed_all':((c0+c1).sum(0)/2+(e0+e1)/2+lex)/4}
  for name,s in options.items():results.append({'split':seed,'method':name,**stats(s,y[ho],h[ho],baseline)})
  print(json.dumps({'split':seed,'results':results[-5:],'elapsed':time.perf_counter()-start}),flush=True)
  (root/'runs/ensemble_results.json').write_text(json.dumps({'records':results,'validation_test_used':False},indent=2)+'\n')
if __name__=='__main__':main()
