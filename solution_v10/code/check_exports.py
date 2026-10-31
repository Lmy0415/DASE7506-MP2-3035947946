"""Verify actual expert deltas even when model selection gives zero blend weight."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import copy,json
from pathlib import Path
import numpy as np
from common import load_inputs
import submission,local_submission
root=Path(__file__).resolve().parent.parent;b,arrays=load_inputs('validation',root.parent/'DASE7506-MP2-main/code/data');result={}
for name,folder,module in [('retrieval','runs',submission),('local_relations','runs_local',local_submission)]:
 checkpoint=root/folder/'model_bundle';s=module.score_components(b,arrays,checkpoint)
 with np.load(root/folder/'validation_scores.npz') as z:
  for k in ['base','delta','selected']:np.testing.assert_allclose(s[k],z[k],rtol=4e-4,atol=4e-4)
 assert module.predict([],{'embeddings':np.empty((0,4,1024),np.float32)},checkpoint)==[]
 small=copy.deepcopy(b[:17]);ar={'embeddings':arrays['embeddings'][:17]};perm=np.array([2,0,3,1])
 for i,r in enumerate(small):r['id']='new-'+str(i)
 np.testing.assert_allclose(module.score_components(small,ar,checkpoint)['delta'],s['delta'][:17])
 for r in small:
  r['candidates']=[r['candidates'][int(j)] for j in perm]
  for j,c in enumerate(r['candidates']):c['index']=j
 moved=module.score_components(small,{'embeddings':ar['embeddings'][:,perm]},checkpoint)
 for k in ['base','delta','selected']:np.testing.assert_allclose(moved[k],s[k][:17,perm],rtol=4e-4,atol=4e-4)
 result[name]={'status':'passed','checks':['exported baseline, actual nonzero expert delta, and selected scores','empty input','new IDs','batch length','candidate permutation of expert delta and selected scores']}
(root/'runs/export_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
