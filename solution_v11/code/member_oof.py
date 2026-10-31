"""Recreate original ensemble members' honest cross-fitted outputs separately."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import time,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from common import load_inputs,read_jsonl
from features import features,labels,lexical_texts
from ensemble_experiment import hard,fit

def main():
 torch.set_num_threads(4);p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';d=p/'DASE7506-MP2-main/code/data';tick=time.perf_counter();b,ar=load_inputs('train',d);y=labels(b,read_jsonl(d/'gsm8k_train.jsonl'));h=hard(b);e=ar['embeddings'];cx=np.concatenate([e,e-e.mean(1,keepdims=True),features(b,e)],-1);text=HashingVectorizer(n_features=32768,ngram_range=(1,2),alternate_sign=True,norm='l2').transform(lexical_texts(b,True));idx=lambda a:(a[:,None]*4+np.arange(4)).ravel();members=np.zeros((len(b),4,4))
 for k,(tr,ho) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261018).split(np.arange(len(b)),h)):
  ca=fit(cx[tr],y[tr],np.ones(len(tr),bool),cx[ho],17,.01,30,[10,30]);ea=fit(e[tr],y[tr],np.ones(len(tr),bool),e[ho],29,.1,40,[40]);lex=Ridge(alpha=10,solver='lsqr').fit(text[idx(tr)],y[tr].ravel()).predict(text[idx(ho)]).reshape(-1,4)
  members[ho]=np.stack([ca[10],ca[30],ea[40],lex],-1);print(json.dumps({'fold':k,'seconds':time.perf_counter()-tick}),flush=True)
 with np.load(p/'solution_v4/runs/oof/oof_scores.npz') as z:
  diff=float(np.max(abs(members.mean(-1)-z['scores'])));np.testing.assert_allclose(members.mean(-1),z['scores'],rtol=1e-5,atol=1e-5)
 np.savez_compressed(out/'member_oof.npz',members=members);print(json.dumps({'status':'passed','max_ensemble_difference':diff,'seconds':time.perf_counter()-tick}))
if __name__=='__main__':main()
