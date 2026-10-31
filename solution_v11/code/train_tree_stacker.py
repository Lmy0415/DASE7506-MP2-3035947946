"""Learn conditional corrections to baseline using grouped ranking trees."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mp2-mpl-cache')
import json,time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

def stats(s,y,h,base):
 idx=np.arange(len(y));c=y[idx,s.argmax(1)].astype(bool);b=y[idx,base.argmax(1)].astype(bool)
 return {'hard_correct':int((h&c).sum()),'hard_n':int(h.sum()),'fixes':int((h&~b&c).sum()),'breaks':int((h&b&~c).sum())}
def main():
 out=Path(__file__).resolve().parent.parent/'runs';tick=time.perf_counter()
 with np.load(out/'stacker_features.npz') as z:x=z['train'];base=z['base'];y=z['labels'];h=z['hard'];vx=z['validation'];vbase=z['vbase'];vy=z['vy'];vh=z['vhard']
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261109).split(np.arange(len(x)),h));results=[];cached={}
 for leaves in [3,7,15]:
  for lam in [10.,100.]:
   deltas={k:np.zeros_like(base) for k in [30,60,120]}
   for fold,(ti,hi) in enumerate(folds):
    params={'objective':'lambdarank','num_leaves':leaves,'learning_rate':.04,'min_data_in_leaf':60,'lambda_l2':lam,'num_threads':4,'seed':17,'verbosity':-1,'force_col_wise':True,'deterministic':True,'label_gain':[0,1],'lambdarank_truncation_level':4}
    train=lgb.Dataset(x[ti].reshape(-1,x.shape[-1]),label=y[ti].ravel().astype(int),group=np.full(len(ti),4),init_score=base[ti].ravel());model=lgb.train(params,train,num_boost_round=120)
    for it in deltas:deltas[it][hi]=model.predict(x[hi].reshape(-1,x.shape[-1]),num_iteration=it,num_threads=4).reshape(-1,4)
   for it,d in deltas.items():
    cached[(leaves,lam,it)]=d
    for a in [.25,.5,1.]:results.append({'leaves':leaves,'l2':lam,'iterations':it,'strength':a,**stats(base+a*d,y,h,base)})
   print(json.dumps({'leaves':leaves,'l2':lam,'seconds':time.perf_counter()-tick}),flush=True)
 results.append({'leaves':0,'l2':0.,'iterations':0,'strength':0.,**stats(base,y,h,base)});best=max(results,key=lambda r:(r['hard_correct'],-r['strength'],-r['iterations'],-r['leaves']));(out/'tree_stacker_search.json').write_text(json.dumps({'selected':best,'records':results,'per_question_oof':True,'fully_nested':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if not best['strength']:return
 params={'objective':'lambdarank','num_leaves':best['leaves'],'learning_rate':.04,'min_data_in_leaf':60,'lambda_l2':best['l2'],'num_threads':4,'seed':17,'verbosity':-1,'force_col_wise':True,'deterministic':True,'label_gain':[0,1],'lambdarank_truncation_level':4};train=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel().astype(int),group=np.full(len(x),4),init_score=base.ravel());model=lgb.train(params,train,num_boost_round=best['iterations']);delta=model.predict(vx.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4);res={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(vbase+best['strength']*delta,vy,vh,vbase),'seconds':time.perf_counter()-tick};(out/'tree_stacker_validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
