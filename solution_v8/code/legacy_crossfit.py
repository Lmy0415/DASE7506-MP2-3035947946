"""Exact frozen v1 neural training recipe, for nested training-only calibration."""
import numpy as np
import torch
def fit(x,y,eval_x,seed,decay,end,checkpoints):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu=x.reshape(-1,x.shape[-1]).mean(0);sd=x.reshape(-1,len(mu)).std(0).clip(.05);xx=torch.tensor((x-mu)/sd);yy=torch.tensor(y);vx=torch.tensor((eval_x-mu)/sd)
 m=torch.nn.Sequential(torch.nn.Linear(len(mu),96),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,1));opt=torch.optim.AdamW(m.parameters(),lr=.0006,weight_decay=decay);outputs={}
 for epoch in range(1,end+1):
  m.train()
  for idx in np.array_split(rng.permutation(len(y)),20):
   s=m(xx[idx]).squeeze(-1);t=yy[idx];loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t==0,-1e9),1)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,t);opt.zero_grad();loss.backward();opt.step()
  if epoch in checkpoints:
   m.eval()
   with torch.no_grad():outputs[epoch]=m(vx).squeeze(-1).numpy()
 return outputs
