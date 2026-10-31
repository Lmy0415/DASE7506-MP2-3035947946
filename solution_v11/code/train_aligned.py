"""Nested held-out evaluation for reference-mined relation retrieval."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time,shutil
from pathlib import Path
import numpy as np,joblib
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from rank_fit import fit,rows,metrics,hard
from aligned_retrieval import build,features
NFEATURES=10
from aligned_submission import constrain,score_components

def view(x,name):
 if name=='strong':
  x=x.copy();active=np.maximum(x[:,:,2],x[:,:,3])>=.4;x*=active[:,:,None]
 return x

def paired(base,new,y):
 a=y[np.arange(len(y)),base.argmax(1)].astype(bool);b=y[np.arange(len(y)),new.argmax(1)].astype(bool)
 return {'fixes':int((~a&b).sum()),'regressions':int((a&~b).sum())}

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs_aligned';out.mkdir(exist_ok=True);data=root.parent/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b)
 with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids']
 _,ibase,_,hbase=joblib.load(root.parent/'solution_v7/runs/internal_features.joblib');_,fbase=joblib.load(root.parent/'solution_v7/runs/full_features.joblib')
 def subset(seq,ids):return [seq[int(i)] for i in ids]
 def cross_retrieve(ids,seed):
  xx=np.zeros((len(ids),4,NFEATURES))
  for k,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(ids,h[ids])):
   library=build(subset(b,ids[ti]),subset(refs,ids[ti]));xx[vi]=features(subset(b,ids[vi]),library)
   print(json.dumps({'retrieval_fold':k,'n':len(ids),'library_rows':len(library['programs']),'seconds':time.perf_counter()-tick}),flush=True)
  return xx
 cache=out/'aligned_crossfit.joblib'
 if cache.exists():ix,hx,fx=joblib.load(cache)
 else:
  ix=cross_retrieve(tr,20261024);library=build(subset(b,tr),subset(refs,tr));hx=features(subset(b,ho),library);fx=cross_retrieve(np.arange(len(b)),20261025);joblib.dump((ix,hx,fx),cache)
 # Recreate v9 honest target-correction scores without using holdout labels.
 tx=joblib.load(root.parent/'solution_v8/runs/target_features_v2.joblib')[:,:37]
 def target_oof(ids,base,seed):
  result=np.zeros_like(base)
  for ti,vi in StratifiedKFold(5,shuffle=True,random_state=seed).split(ids,h[ids]):
   w=fit(tx[rows(ids[ti])],base[ti],y[ids[ti]],.001);result[vi]=base[vi]+(tx[rows(ids[vi])]@w).reshape(-1,4)
  return constrain(subset(b,ids),result)
 w=fit(tx[rows(tr)],ibase,y[tr],.001);hv9=constrain(subset(b,ho),hbase+(tx[rows(ho)]@w).reshape(-1,4));iv9=target_oof(tr,ibase,20261026);fv9=target_oof(np.arange(len(b)),fbase,20261027)
 bases={'v6':(ibase,hbase,fbase),'v9':(iv9,hv9,fv9)};results=[]
 for base_name,(inner,outer,_) in bases.items():
  results.append({'base':base_name,'view':'all','l2':.1,'strength':0.,**metrics(outer,y[ho],h[ho]),**paired(outer,outer,y[ho])})
  for name in ['all','strong']:
   xx=view(ix,name);zz=view(hx,name)
   for l2 in [.001,.01,.1]:
    w=fit(sparse.csr_matrix(xx.reshape(-1,NFEATURES)),inner,y[tr],l2);delta=zz@w
    for strength in [.25,.5,1.]:
     s=outer+strength*delta;r={'base':base_name,'view':name,'l2':l2,'strength':strength,**metrics(s,y[ho],h[ho]),**paired(outer,s,y[ho])};results.append(r);print(json.dumps(r),flush=True)
 best=max(results,key=lambda r:(r['hard_correct'],r['correct'],-r['strength'],r['l2'],r['view']=='strong',r['base']=='v6'))
 (out/'internal_search.json').write_text(json.dumps(results,indent=2)+'\n');(out/'training_plan.json').write_text(json.dumps({'selected':best,'baseline':{k:metrics(v[1],y[ho],h[ho]) for k,v in bases.items()},'reference_library_outer_holdout_excluded':True,'every_training_retrieval_omits_own_fold':True,'test_used':False},indent=2)+'\n')
 library=build(b,refs);bundle=out/'model_bundle';bundle.mkdir(exist_ok=True);joblib.dump(library,bundle/'patterns.joblib',compress=0)
 source=root.parent/('solution_v6' if best['base']=='v6' else 'solution_v8')/'runs/model_bundle';shutil.copytree(source,bundle/'base',dirs_exist_ok=True)
 w=fit(sparse.csr_matrix(view(fx,best['view']).reshape(-1,NFEATURES)),bases[best['base']][2],y,best['l2'])
 np.savez(bundle/'correction.npz',weight=w,strength=np.array(best['strength']),base=np.array(best['base']),view=np.array(best['view']))
 vb,va=load_inputs('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb);s=score_components(vb,va,bundle)
 np.savez(out/'validation_scores.npz',**s)
 comparison={'selected_recipe':best,'library':{'reference_programs':len(library['programs'])},'internal':metrics(bases[best['base']][1]+best['strength']*(view(hx,best['view'])@fit(sparse.csr_matrix(view(ix,best['view']).reshape(-1,NFEATURES)),bases[best['base']][0],y[tr],best['l2'])),y[ho],h[ho]),'validation':{k:metrics(v,vy,vh) for k,v in s.items() if k!='delta'},'validation_changes':paired(s['base'],s['selected'],vy),'seconds':time.perf_counter()-tick,'test_used':False}
 changes=[]
 for i in np.flatnonzero(s['base'].argmax(1)!=s['selected'].argmax(1)):
  old=int(s['base'][i].argmax());new=int(s['selected'][i].argmax());changes.append({'row':int(i),'old':old,'new':new,'old_correct':bool(vy[i,old]),'new_correct':bool(vy[i,new])})
 (out/'validation_changes.json').write_text(json.dumps(changes,indent=2)+'\n');(out/'validation_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n');print(json.dumps(comparison,indent=2),flush=True)
 # Reference reliability and abstention diagnostics use train and validation only.
 with np.load(out/'validation_scores.npz') as z:assert np.isfinite(z['selected']).all()
 np.savez_compressed(out/'honest_scores.npz',inner_base=bases[best['base']][0],outer_base=bases[best['base']][1],full_base=bases[best['base']][2])
if __name__=='__main__':main()
