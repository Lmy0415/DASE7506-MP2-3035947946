"""Learn when an independent specialist should replace the baseline answer."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import load_inputs,read_jsonl,parse_answer
from features import features,labels
from calculation_evidence import evidence_features
from target_features import features as target_features
from ensemble_experiment import hard

def make(models,bank,hand,calc,role,target,mode,scales):
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
  if mode=='rich':
   for f in [hand[:,:,14:54],calc[:,:,8:14],role,target]:block.extend((f[:,j]-f[ids,ii]).T)
  feats.append(np.stack(block,1))
 return np.stack(feats,1),ii,eligible

def fit_predict(x,y,base_choice,eligible,h,ti,hi,c):
 ids=np.arange(len(y));baseok=y[ids,base_choice].astype(bool);truth=y.astype(bool);valid=eligible&h[:,None]&(truth!=baseok[:,None]);tr=valid.copy();tr[hi]=False
 ii,jj=np.nonzero(tr);target=truth[ii,jj].astype(int);weights=1/np.maximum(1,valid.sum(1)[ii]);sc=StandardScaler().fit(x[ii,jj]);clf=LogisticRegression(C=c,max_iter=1000).fit(sc.transform(x[ii,jj]),target,sample_weight=weights)
 flat=x[hi].reshape(-1,x.shape[-1]);p=clf.predict_proba(sc.transform(flat))[:,1].reshape(len(hi),4);p[~eligible[hi]]=0
 return p,sc,clf

def metric(prob,threshold,base_choice,y,h):
 j=prob.argmax(1);active=prob.max(1)>=threshold;ids=np.arange(len(y));bc=y[ids,base_choice].astype(bool);alt=y[ids,j].astype(bool);c=np.where(active,alt,bc)
 return {'hard_correct':int((h&c).sum()),'fixes':int((h&active&~bc&alt).sum()),'breaks':int((h&active&bc&~alt).sum()),'switches':int((h&active).sum())}

def main():
 project=Path(__file__).resolve().parents[2];root=project/'solution_v11';out=root/'runs';data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,a=load_inputs('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);hand=features(b,a['embeddings'])
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(project/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vind=z['validation'].mean(0)
 with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100'];vlin=z['embedding_all_100_val']
 with np.load(out/'role_oof.npz') as z:ro=z['train'];vro=z['validation']
 with np.load(out/'critic_oof.npz') as z:cri=base+.25*128.0030667*z['0.1']
 with np.load(out/'expanded_oof.npz') as z:exp=base+.25*z['all_0.1']
 with np.load(out/'small_program_oof.npz') as z:small=base+2*z['10.0']
 with np.load(project/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 role,_=joblib.load(project/'solution_v7/runs/full_features.joblib');target=joblib.load(project/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37)
 # Critic OOF is available, but avoid using it as an inference dependency until
 # the selector proves useful; its probability can still be studied later.
 models=[base,ind,lin,ro,cri,exp,small];scales=[np.std(s-s.mean(1,keepdims=True)).clip(.05) for s in models];records=[];idx=np.arange(len(b))
 for mode in ['basic','rich']:
  x,ii,eligible=make(models,b,hand,calc,role,target,mode,scales)
  for c in [.001,.01,.1,1.]:
   pp=[]
   for seed in [201,202,203]:
    o=np.zeros((len(b),4))
    for ti,hi in StratifiedKFold(5,shuffle=True,random_state=seed).split(idx,h):o[hi]=fit_predict(x,y,ii,eligible,h,ti,hi,c)[0]
    pp.append(o)
   for threshold in [.5,.55,.6,.65,.7]:
    rr=[metric(p,threshold,ii,y,h) for p in pp];gains=[r['fixes']-r['breaks'] for r in rr];records.append({'mode':mode,'c':c,'threshold':threshold,'gains':gains,'mean_gain':float(np.mean(gains)),'worst_gain':min(gains),'switches':float(np.mean([z['switches'] for z in rr]))})
   print(json.dumps({'mode':mode,'c':c,'seconds':time.perf_counter()-tick}),flush=True)
 records.append({'mode':'none','c':0.,'threshold':1.,'gains':[0,0,0],'mean_gain':0.,'worst_gain':0,'switches':0.})
 best=max(records,key=lambda r:(r['mean_gain'],r['worst_gain'],-r['switches']));(out/'multigate_more_search.json').write_text(json.dumps({'records':records,'selected':best,'per_question_oof':True,'fully_nested':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if best['mode']=='none':return
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vhand=features(vb,va['embeddings']);vcalc=evidence_features(vb)[0];vtarget=target_features(vb,'numeric').toarray().reshape(-1,4,37)
 with np.load(project/'solution_v7/runs/validation_scores.npz') as z:vrole=z['features']
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 with np.load(out/'extra_validation_scores.npz') as z:vcri=vbase+.25*128.0030667*z['critic'];vsmall=vbase+2*z['small_program']
 with np.load(out/'expanded_validation_scores.npz') as z:vexp=vbase+.25*z['delta']
 vmodels=[vbase,vind,vbase+.25*7.9411391*vlin,vro,vcri,vexp,vsmall];vx,vi,valid=make(vmodels,vb,vhand,vcalc,vrole,vtarget,best['mode'],scales)
 x,ii,eligible=make(models,b,hand,calc,role,target,best['mode'],scales);p,sc,clf=fit_predict(x,y,ii,eligible,h,idx,np.array([],dtype=int),best['c']) if False else (None,None,None)
 bc=y[idx,ii].astype(bool);validtr=eligible&h[:,None]&(y.astype(bool)!=bc[:,None]);si,sj=np.nonzero(validtr);weights=1/validtr.sum(1)[si];sc=StandardScaler().fit(x[si,sj]);clf=LogisticRegression(C=best['c'],max_iter=1000).fit(sc.transform(x[si,sj]),y[si,sj].astype(int),sample_weight=weights)
 p=clf.predict_proba(sc.transform(vx.reshape(-1,vx.shape[-1])))[:,1].reshape(len(vb),4);p[~valid]=0
 val=metric(p,best['threshold'],vi,vy,vh);baseval=metric(np.zeros_like(p),1,vi,vy,vh);result={'selected':best,'validation_base':baseval,'validation_candidate':val,'seconds':time.perf_counter()-tick};(out/'multigate_more_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
