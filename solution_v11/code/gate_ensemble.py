"""Aggregate cross-fitted selection gates instead of relying on one split."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import features,labels
from calculation_evidence import evidence_features
from target_features import features as target_features
from ensemble_experiment import hard
from multi_gate_more import make,fit_predict,metric

def main():
 p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';t=time.perf_counter();b,a=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);hand=features(b,a['embeddings'])
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vind=z['validation'].mean(0)
 with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100'];vlin=z['embedding_all_100_val']
 with np.load(out/'role_oof.npz') as z:ro=z['train'];vro=z['validation']
 with np.load(out/'critic_oof.npz') as z:cri=base+.25*128.0030667*z['0.1']
 with np.load(out/'expanded_oof.npz') as z:exp=base+.25*z['all_0.1']
 with np.load(out/'small_program_oof.npz') as z:small=base+2*z['10.0']
 with np.load(p/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 role,_=joblib.load(p/'solution_v7/runs/full_features.joblib');target=joblib.load(p/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37);models=[base,ind,lin,ro,cri,exp,small];scales=[np.std(s-s.mean(1,keepdims=True)).clip(.05) for s in models];idx=np.arange(len(b));results=[];models_cache={}
 vb,va=load_inputs('validation',data);vh=hard(vb);vhand=features(vb,va['embeddings']);vcalc=evidence_features(vb)[0];vtarget=target_features(vb,'numeric').toarray().reshape(-1,4,37)
 with np.load(p/'solution_v7/runs/validation_scores.npz') as z:vrole=z['features']
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 with np.load(out/'extra_validation_scores.npz') as z:vcri=vbase+.25*128.0030667*z['critic'];vsmall=vbase+2*z['small_program']
 with np.load(out/'expanded_validation_scores.npz') as z:vexp=vbase+.25*z['delta']
 vmodels=[vbase,vind,vbase+.25*7.9411391*vlin,vro,vcri,vexp,vsmall]
 for mode in ['basic','rich']:
  x,ii,eligible=make(models,b,hand,calc,role,target,mode,scales);vx,vi,valid=make(vmodels,vb,vhand,vcalc,vrole,vtarget,mode,scales)
  for C in [.01,.1,1.]:
   ps=[];pv=[]
   for seed in [201,202,203]:
    prob=np.zeros((len(b),4))
    for ti,hi in StratifiedKFold(5,shuffle=True,random_state=seed).split(idx,h):
     prob[hi],sc,model=fit_predict(x,y,ii,eligible,h,ti,hi,C)
     value=model.predict_proba(sc.transform(vx.reshape(-1,vx.shape[-1])))[:,1].reshape(len(vb),4);value[~valid]=0;pv.append(value)
    ps.append(prob)
   avg=np.mean(ps,0);avgval=np.mean(pv,0);models_cache[(mode,C)]=avgval
   for threshold in [.45,.5,.55,.6,.65,.7]:
    rr=metric(avg,threshold,ii,y,h);r={'mode':mode,'c':C,'threshold':threshold,**rr,'gain':rr['fixes']-rr['breaks']};results.append(r)
   print(json.dumps({'mode':mode,'c':C,'seconds':time.perf_counter()-t}),flush=True)
 results.append({'mode':'none','c':0,'threshold':1.,'gain':0});best=max(results,key=lambda r:(r['gain'],-r.get('switches',0),r['threshold']));(out/'gate_ensemble_search.json').write_text(json.dumps({'records':results,'selected':best,'per_question_oof':True,'fully_nested':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if best['mode']=='none':return
 vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));_,vi,valid=make(vmodels,vb,vhand,vcalc,vrole,vtarget,best['mode'],scales);pv=models_cache[(best['mode'],best['c'])];val=metric(pv,best['threshold'],vi,vy,vh);baseline=metric(np.zeros_like(pv),1.,vi,vy,vh);result={'selected':best,'validation_base':baseline,'validation_candidate':val,'seconds':time.perf_counter()-t};(out/'gate_ensemble_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
