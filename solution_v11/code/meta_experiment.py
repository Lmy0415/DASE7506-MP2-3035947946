"""Crossfit a conservative interaction ranker from existing train-only OOF scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import features,labels
from calculation_evidence import evidence_features
from target_features import features as target_features
from ensemble_experiment import hard,stats
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'solution_v10/code'))
from rank_fit import fit

def feat(base,ind,hand,calc,role,target,mode):
 B=base-base.mean(1,keepdims=True);I=ind-ind.mean(1,keepdims=True)
 hand=hand[:,:,:54];inputs=[I[:,:,None],(I*B)[:,:,None],hand[:,:,1:3]*I[:,:,None],hand[:,:,1:3]*B[:,:,None]]
 if mode=='rich':inputs.extend([hand,calc[:,:,:14],role,target])
 return np.concatenate(inputs,axis=-1)
def flatten(x):return sparse.csr_matrix(x.reshape(-1,x.shape[-1]))
def metrics(s,y,h,base):return stats(s,y,h,base)

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs';project=root.parent;data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,ar=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(project/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vv=z['validation'].mean(0)
 with np.load(project/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 role,_=joblib.load(project/'solution_v7/runs/full_features.joblib');target=joblib.load(project/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37)
 hand=features(b,ar['embeddings']);vvbank,vva=load_inputs('validation',data);vy=labels(vvbank,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vvbank)
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 with np.load(project/'solution_v7/runs/validation_scores.npz') as z:vrole=z['features']
 vhand=features(vvbank,vva['embeddings']);vcalc=evidence_features(vvbank)[0];vtarget=target_features(vvbank,'numeric').toarray().reshape(-1,4,37)
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261031).split(np.arange(len(b)),h));results=[];preds={}
 for mode in ['gate','rich']:
  x=feat(base,ind,hand,calc,role,target,mode);vx=feat(vbase,vv,vhand,vcalc,vrole,vtarget,mode)
  for l2 in [.001,.01,.1,1.]:
   cv=np.zeros((len(b),4));deltas=np.zeros((len(b),4))
   for ti,hi in folds:
    std=x[ti].reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flatten(x[ti]/std),base[ti],y[ti],l2);deltas[hi]=(x[hi]/std)@w
   for strength in [.25,.5,1.]:
    s=base+strength*deltas;r={'mode':mode,'l2':l2,'strength':strength,**metrics(s,y,h,base)};results.append(r);preds[(mode,l2,strength)]=s;print(json.dumps(r),flush=True)
   print(json.dumps({'finished':mode,'l2':l2,'seconds':time.perf_counter()-tick}),flush=True)
 results.append({'mode':'none','l2':1.,'strength':0.,**metrics(base,y,h,base)})
 best=max(results,key=lambda z:(z['hard_correct'],-z['strength'],z['l2']));(out/'meta_search.json').write_text(json.dumps({'records':results,'selected':best,'baseline':metrics(base,y,h,base),'crossfit_nested_training_library':False},indent=2)+'\n')
 if best['mode']=='none':print(json.dumps({'selected':best}),flush=True);return
 x=feat(base,ind,hand,calc,role,target,best['mode']);vx=feat(vbase,vv,vhand,vcalc,vrole,vtarget,best['mode']);std=x.reshape(-1,x.shape[-1]).std(0).clip(.05);w=fit(flatten(x/std),base,y,best['l2']);s=vbase+best['strength']*((vx/std)@w)
 result={'train_cv':best,'validation_base':metrics(vbase,vy,vh,vbase),'validation_candidate':metrics(s,vy,vh,vbase),'features':x.shape[-1],'seconds':time.perf_counter()-tick};(out/'meta_validation.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out/'meta_validation_scores.npz',base=vbase,selected=s,weight=w,std=std);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
