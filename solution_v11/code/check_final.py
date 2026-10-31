"""Check deployed expert outputs against independently saved development outputs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import copy,json
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl
from features import labels
from ensemble_experiment import hard,stats
from final_submission import score_components,predict
p=Path(__file__).resolve().parents[2];root=p/'solution_v11';out=root/'runs';data=p/'DASE7506-MP2-main/code/data';b,arrays=load_inputs('validation',data);y=labels(b,read_jsonl(data/'gsm8k_validation.jsonl'));h=hard(b);s=score_components(b,arrays,out/'final_bundle')
with np.load(p/'solution_v9/runs/validation_scores.npz') as z:base=z['selected']
with np.load(p/'training/independent_expert/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['validation'].mean(0)
with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100_val']
with np.load(out/'role_oof.npz') as z:role=z['validation']
with np.load(out/'extra_validation_scores.npz') as z:crit=base+.25*128.0030667*z['critic']
with np.load(out/'expanded_validation_scores.npz') as z:ex=base+.25*z['delta']
errors={}
for name,ref in [('base',base),('independent',ind),('linear',lin),('role',role),('critic',crit),('expanded',ex)]:
 errors[name]=float(np.max(abs(s[name]-ref)));np.testing.assert_allclose(s[name],ref,rtol=2e-4,atol=2e-4)
assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},out/'final_bundle')==[]
chosen=predict(b,arrays,out/'final_bundle');assert all(type(i) is int and 0<=i<4 for i in chosen)
small=copy.deepcopy(b[:17]);ar={'embeddings':arrays['embeddings'][:17]}
for i,r in enumerate(small):r['id']=f'unknown-{i}'
orig=score_components(small,ar,out/'final_bundle');np.testing.assert_allclose(orig['selected'],s['selected'][:17],rtol=2e-4,atol=2e-4)
perm=np.array([2,0,3,1])
for r in small:
 r['candidates']=[r['candidates'][int(j)] for j in perm]
 for j,c in enumerate(r['candidates']):c['index']=j
new=score_components(small,{'embeddings':ar['embeddings'][:,perm]},out/'final_bundle')
np.testing.assert_allclose(new['selected'],orig['selected'][:,perm],rtol=2e-4,atol=2e-4)
result={'status':'passed','expert_max_score_errors':errors,'validation_base':stats(s['base'],y,h,s['base']),'validation_selected':stats(s['selected'],y,h,s['base']),'checks':['exact development expert exports','empty input','integer choices','new IDs and arbitrary length','permuted candidate scores']}
(out/'final_contract.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'final_validation_scores.npz',base=s['base'],selected=s['selected']);print(json.dumps(result,indent=2))
