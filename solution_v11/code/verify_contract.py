"""Check frozen inference against development output and input invariances."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mp2-mpl-cache')
import argparse,copy,json
from pathlib import Path
import numpy as np
from common import load_inputs
from submission import predict,score_components
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
bank,arrays=load_inputs('validation',a.data_dir);s=score_components(bank,arrays,a.checkpoint)['selected']
with np.load(a.scores) as z:expected=z['selected']
np.testing.assert_allclose(s,expected,rtol=4e-4,atol=4e-4)
chosen=predict(bank,arrays,a.checkpoint);assert all(type(i)is int and 0<=i<4 for i in chosen)
assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},a.checkpoint)==[]
small=copy.deepcopy(bank[:17]);small_arrays={'embeddings':arrays['embeddings'][:17]}
for i,r in enumerate(small):r['id']=f'new-input-{i}'
assert predict(small,small_arrays,a.checkpoint)==chosen[:17]
perm=np.array([2,0,3,1]);changed=copy.deepcopy(small)
for r in changed:
 r['candidates']=[r['candidates'][int(j)] for j in perm]
 for j,c in enumerate(r['candidates']):c['index']=j
moved=score_components(changed,{'embeddings':small_arrays['embeddings'][:,perm]},a.checkpoint)['selected']
np.testing.assert_allclose(moved,s[:17,perm],rtol=4e-4,atol=4e-4)
result={'status':'passed','checks':['saved training predictions match frozen inference','integer outputs','empty bank','new IDs','arbitrary length','candidate permutation score equivariance'],'max_saved_score_error':float(abs(s-expected).max())}
if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
