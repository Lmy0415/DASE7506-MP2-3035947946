"""Verify alignment invariance, fresh input support, and frozen-score consistency."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
import argparse,copy,json
from pathlib import Path
import numpy as np
from common import load_inputs
from submission import predict,member_scores
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);args=p.parse_args()
bank,arrays=load_inputs('validation',args.data_dir)
s=np.mean(member_scores(bank,arrays,args.checkpoint),axis=0)
with np.load(args.scores) as f: expected=f['selected_ensemble']
np.testing.assert_allclose(s,expected,rtol=2e-4,atol=2e-4)
chosen=predict(bank,arrays,args.checkpoint)
assert all(type(v) is int and 0<=v<4 for v in chosen)
assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},args.checkpoint)==[]
# Changed IDs cannot influence output; arbitrary bank lengths must work.
small=copy.deepcopy(bank[:17])
for i,row in enumerate(small):row['id']=f'unseen-{i}'
assert predict(small,{'embeddings':arrays['embeddings'][:17]},args.checkpoint)==chosen[:17]
perm=np.array([2,0,3,1]); changed=copy.deepcopy(small)
for row in changed:
    row['candidates']=[row['candidates'][int(k)] for k in perm]
    for j,c in enumerate(row['candidates']):c['index']=j
moved=np.mean(member_scores(changed,{'embeddings':arrays['embeddings'][:17,perm]},args.checkpoint),axis=0)
np.testing.assert_allclose(moved,s[:17,perm],rtol=2e-4,atol=2e-4)
print(json.dumps({'status':'passed','checks':['numpy/training agreement','Python integer outputs','empty bank','unseen IDs','arbitrary bank length','candidate-permutation score equivariance']}))
