"""Build candidate gate and five linear experts from training data only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl
from features import features,labels
from ensemble_experiment import hard
from multi_gate_more import fit_predict
from gate_features import make

def main():
 p=Path(__file__).resolve().parents[2];root=p/'solution_v11';out=root/'runs';bundle=out/'final_bundle';bundle.mkdir(exist_ok=True);data=p/'DASE7506-MP2-main/code/data';b,a=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);hand=features(b,a['embeddings'])
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof']
 with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100']
 with np.load(out/'role_oof.npz') as z:role=z['train']
 with np.load(out/'critic_oof.npz') as z:critic=base+.25*128.0030667*z['0.1']
 with np.load(out/'expanded_oof.npz') as z:extra=base+.25*z['all_0.1']
 with np.load(p/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 models=[base,ind,lin,role,critic,extra];scales=np.asarray([np.std(s-s.mean(1,keepdims=True)).clip(.05) for s in models]);x,ii,eligible=make(models,b,hand,calc,scales);idx=np.arange(len(b));mm=[];ss=[];ww=[];bb=[]
 for seed in [201,202,203]:
  for ti,hi in StratifiedKFold(5,shuffle=True,random_state=seed).split(idx,h):
   _,sc,model=fit_predict(x,y,ii,eligible,h,ti,hi,.1);mm.append(sc.mean_);ss.append(sc.scale_);ww.append(model.coef_[0]);bb.append(model.intercept_[0])
 np.savez(bundle/'gate.npz',model_scales=scales,means=mm,scales=ss,weights=ww,bias=bb,threshold=.6)
 # The selected ridge expert uses five fold models; export exact coefficients.
 e=a['embeddings'];x=e-e.mean(1,keepdims=True);weights=[];stds=[]
 for ti,hi in StratifiedKFold(5,shuffle=True,random_state=20261101).split(idx,h):
  sd=x[ti].reshape(-1,x.shape[-1]).std(0).clip(.05);model=Ridge(alpha=100.,fit_intercept=False,solver='lsqr',tol=1e-4).fit((x[ti]/sd).reshape(-1,x.shape[-1]),(y[ti]-y[ti].mean(1,keepdims=True)).ravel());weights.append(model.coef_);stds.append(sd)
 np.savez(bundle/'linear.npz',weight=np.asarray(weights),sd=np.asarray(stds))
 shutil.copytree(p/'solution_v9/runs/model_bundle',bundle/'v9',dirs_exist_ok=True)
 shutil.copytree(p/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2',bundle/'independent',dirs_exist_ok=True)
 shutil.copy(p/'solution_v7/runs/model_bundle/quantity_model.joblib',bundle/'role_model.joblib');shutil.copy(p/'solution_v7/runs/model_bundle/relation_correction.npz',bundle/'role_correction.npz')
 shutil.copy(out/'critic_model.npz',bundle/'critic.npz');shutil.copy(out/'expanded_model.npz',bundle/'expanded.npz')
 size=sum(x.stat().st_size for x in bundle.rglob('*') if x.is_file());print(json.dumps({'asset_bytes':size,'asset_mib':size/2**20,'number_gate_models':len(mm)},indent=2));assert size<16*1024*1024
if __name__=='__main__':main()
