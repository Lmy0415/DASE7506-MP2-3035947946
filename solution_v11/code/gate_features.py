"""Per-question gate features with fixed training-set scales."""
import numpy as np
from common import parse_answer

def make(models,bank,hand,calc,scales):
 n=len(bank);ids=np.arange(n);base=models[0];ii=base.argmax(1);norm=[(s-s.mean(1,keepdims=True))/sc for s,sc in zip(models,scales)]
 answers=[[parse_answer(c['text']) for c in r['candidates']] for r in bank];votes=np.zeros((n,4))
 for s in models[1:]:
  tops=s.argmax(1)
  for i,(a,t) in enumerate(zip(answers,tops)):
   for j in range(4):
    if a[j] is not None and a[j]==a[t]:votes[i,j]+=1
 eligible=np.array([[a[j] is not None and a[j]!=a[k] and votes[i,j]>0 for j in range(4)] for i,(a,k) in enumerate(zip(answers,ii))])
 feats=[]
 for j in range(4):
  block=[]
  for s in norm:
   order=np.sort(s,axis=1);block.extend([s[:,j]-s[ids,ii],order[:,-1]-order[:,-2],s[:,j]-s.max(1),np.std(s,1)])
  block.extend([votes[:,j],hand[:,j,1]-hand[ids,ii,1],hand[:,j,2]-hand[ids,ii,2]])
  for f in [hand[:,:,:14],calc[:,:,:8]]:block.extend((f[:,j]-f[ids,ii]).T)
  feats.append(np.stack(block,1))
 return np.stack(feats,1),ii,eligible
