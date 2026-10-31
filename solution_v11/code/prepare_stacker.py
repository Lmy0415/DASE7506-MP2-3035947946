"""Assemble train-only crossfitted ensemble scores and deterministic features."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import joblib,numpy as np
from pathlib import Path
from common import load_inputs,read_jsonl
from features import features,labels
from calculation_evidence import evidence_features
from expanded_calculations import features as expanded
from target_features import features as target_features

def hard(bank):
 from common import parse_answer
 return np.array([len({a for c in row['candidates'] if (a:=parse_answer(c['text'])) is not None})>1 for row in bank])
def stack(models,hand,calc,role,target,expand):
 values=[]
 for s in models:
  z=s-s.mean(1,keepdims=True);values.extend([z[:,:,None],(z-z.max(1,keepdims=True))[:,:,None]])
 values.extend([hand[:,:,:54],calc[:,:,:14],role[:,:,:11],target,expand[:,:,:12]])
 return np.concatenate(values,axis=-1).astype('float32')
p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';b,a=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);hand=features(b,a['embeddings'])
with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
with np.load(p/'training/independent_expert/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vind=z['validation'].mean(0)
with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100'];vlin=z['embedding_all_100_val']
with np.load(out/'role_oof.npz') as z:ro=z['train'];vro=z['validation']
with np.load(out/'critic_oof.npz') as z:cri=base+.25*128.0030667*z['0.1']
with np.load(out/'expanded_oof.npz') as z:exp=base+.25*z['all_0.1']
with np.load(out/'small_program_oof.npz') as z:small=base+2*z['10.0']
with np.load(out/'neural_oof.npz') as z:neural=base+.5*.3699988662*z['equal'];vneural=z['equal_val']
with np.load(p/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
with np.load(out/'expanded_features.npz') as z:exf=z['train']
role,_=joblib.load(p/'solution_v7/runs/full_features.joblib');target=joblib.load(p/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37)
x=stack([base,ind,lin,ro,cri,exp,small,neural],hand,calc,role,target,exf)
vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vhand=features(vb,va['embeddings']);vcalc=evidence_features(vb)[0];vtarget=target_features(vb,'numeric').toarray().reshape(-1,4,37);vexf=expanded(vb)
with np.load(p/'solution_v7/runs/validation_scores.npz') as z:vrole=z['features']
with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
with np.load(out/'extra_validation_scores.npz') as z:vcri=vbase+.25*128.0030667*z['critic'];vsmall=vbase+2*z['small_program']
with np.load(out/'expanded_validation_scores.npz') as z:vexp=vbase+.25*z['delta']
vx=stack([vbase,vind,vbase+.25*7.9411391*vlin,vro,vcri,vexp,vsmall,vbase+.5*.3699988662*vneural],vhand,vcalc,vrole,vtarget,vexf)
np.savez_compressed(out/'stacker_features.npz',train=x,base=base,labels=y,hard=h,validation=vx,vbase=vbase,vy=vy,vhard=vh);print('prepared',x.shape,vx.shape)
