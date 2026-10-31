"""Crossfit a switch classifier on genuine disagreements between rankers."""
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

def make(base,other,bank,hand,calc,role,target,mode,scales):
 n=len(base);ii=base.argmax(1);jj=other.argmax(1);ids=np.arange(n);bs=base-base.mean(1,keepdims=True);os=other-other.mean(1,keepdims=True);bs=bs/scales[0];os=os/scales[1]
 topgap=lambda s:np.sort(s,axis=1)[:,-1]-np.sort(s,axis=1)[:,-2]
 diff=lambda a:a[ids,jj]-a[ids,ii]
 extra=np.stack([topgap(bs),topgap(os),bs[ids,jj]-bs[ids,ii],os[ids,jj]-os[ids,ii],np.std(bs,1),np.std(os,1)],1)
 out=[extra,diff(hand[:,:,:14]),diff(calc[:,:,:8])]
 if mode=='rich':out.extend([diff(hand[:,:,14:54]),diff(calc[:,:,8:14]),diff(role),diff(target)])
 answers=[[parse_answer(c['text']) for c in r['candidates']] for r in bank];disagree=np.array([a[i]!=a[j] for a,i,j in zip(answers,ii,jj)])
 return np.concatenate(out,1),ii,jj,disagree

def summarize(pred,i,j,y,h,disagree):
 idx=np.arange(len(y));bc=y[idx,i].astype(bool);oc=y[idx,j].astype(bool);active=h&disagree&pred
 return {'hard_correct':int((h&(bc|(active&oc))&~(active&~oc)).sum()),'hard_n':int(h.sum()),'fixes':int((active&~bc&oc).sum()),'breaks':int((active&bc&~oc).sum()),'switches':int(active.sum())}

def fit_predict(x,labels,info,ti,hi,c):
 train=ti[info[ti]]
 sc=StandardScaler().fit(x[train]);model=LogisticRegression(C=c,max_iter=1000).fit(sc.transform(x[train]),labels[train]);return model.predict_proba(sc.transform(x[hi]))[:,1]

def main():
 project=Path(__file__).resolve().parents[2];root=project/'solution_v11';out=root/'runs';data=project/'DASE7506-MP2-main/code/data';tick=time.perf_counter();bank,ar=load_inputs('train',data);y=labels(bank,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(bank);hand=features(bank,ar['embeddings'])
 with np.load(project/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(project/'kimi_mp2/DASE7506-MP2-main/runs/cv_math_cosine50_r2_scores.npz') as z:other=z['oof'];vother=z['validation'].mean(0)
 with np.load(project/'solution_v6/runs/train_evidence.npz') as z:calc=z['features']
 role,_=joblib.load(project/'solution_v7/runs/full_features.joblib');target=joblib.load(project/'solution_v8/runs/target_features_v2.joblib')[:,:37].toarray().reshape(-1,4,37)
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);vhand=features(vb,va['embeddings']);vcalc=evidence_features(vb)[0];vtarget=target_features(vb,'numeric').toarray().reshape(-1,4,37)
 with np.load(project/'solution_v7/runs/validation_scores.npz') as z:vrole=z['features']
 with np.load(project/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 idx=np.arange(len(bank));records=[];settings={}
 scales=(np.std(base-base.mean(1,keepdims=True)).clip(.05),np.std(other-other.mean(1,keepdims=True)).clip(.05))
 for mode in ['basic','rich']:
  x,i,j,d=make(base,other,bank,hand,calc,role,target,mode,scales);bc=y[idx,i].astype(bool);oc=y[idx,j].astype(bool);info=h&d&(bc!=oc);truth=oc.astype(int)
  print(json.dumps({'mode':mode,'informative':int(info.sum()),'base_wrong_alt_correct':int((info&oc).sum())}),flush=True)
  for c in [.001,.01,.1,1.]:
   ps=[]
   for seed in [101,102,103]:
    p=np.zeros(len(bank));fold=StratifiedKFold(5,shuffle=True,random_state=seed)
    for ti,hi in fold.split(idx,h):p[hi]=fit_predict(x,truth,info,ti,hi,c)
    ps.append(p)
   for threshold in [.55,.6,.65,.7,.75]:
    rr=[summarize(p>=threshold,i,j,y,h,d) for p in ps];gains=[r['fixes']-r['breaks'] for r in rr];record={'mode':mode,'c':c,'threshold':threshold,'gains':gains,'mean_gain':float(np.mean(gains)),'worst_gain':min(gains),'switches':float(np.mean([z['switches'] for z in rr]))};records.append(record);settings[(mode,c,threshold)]=rr
   print(json.dumps({'mode':mode,'c':c,'seconds':time.perf_counter()-tick}),flush=True)
 # A null model is an explicit option; choose by mean then worst split.
 records.append({'mode':'none','c':0.,'threshold':1.,'gains':[0,0,0],'mean_gain':0.,'worst_gain':0,'switches':0.})
 best=max(records,key=lambda r:(r['mean_gain'],r['worst_gain'],-r['switches']))
 (out/'gate_search.json').write_text(json.dumps({'records':records,'selected':best,'train_pairwise_reference_oof':True,'nested_outer_exclusion':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if best['mode']=='none':return
 x,i,j,d=make(base,other,bank,hand,calc,role,target,best['mode'],scales);bc=y[idx,i].astype(bool);oc=y[idx,j].astype(bool);info=h&d&(bc!=oc)
 vx,vi,vj,vd=make(vbase,vother,vb,vhand,vcalc,vrole,vtarget,best['mode'],scales);sc=StandardScaler().fit(x[info]);model=LogisticRegression(C=best['c'],max_iter=1000).fit(sc.transform(x[info]),oc[info].astype(int));p=model.predict_proba(sc.transform(vx))[:,1];val=summarize(p>=best['threshold'],vi,vj,vy,vh,vd)
 result={'selected':best,'validation_baseline':summarize(np.zeros(len(vb),bool),vi,vj,vy,vh,vd),'validation_candidate':val,'seconds':time.perf_counter()-tick};(out/'gate_validation.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out/'gate_model.npz',mean=sc.mean_,scale=sc.scale_,weight=model.coef_,intercept=model.intercept_,threshold=best['threshold']);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
