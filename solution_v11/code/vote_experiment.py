"""Compare calibrated cross-expert answer support using train-only predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,numpy as np,joblib
from pathlib import Path
from common import load_bank,read_jsonl,parse_answer
from features import labels
from ensemble_experiment import hard,stats

def votes(bank,models):
 result=np.zeros((len(bank),4))
 for row,s in zip(range(len(bank)),bank):
  ans=[parse_answer(c['text']) for c in s['candidates']]
  for m in models:
   selected=ans[m[row].argmax()]
   for j,a in enumerate(ans):result[row,j]+=float(a is not None and a==selected)
 return result

def main():
 p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';b=load_bank('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b)
 with np.load(p/'solution_v10/runs/honest_scores.npz') as z:base=z['full_base']
 with np.load(p/'training/independent_expert/runs/cv_math_cosine50_r2_scores.npz') as z:ind=z['oof'];vind=z['validation'].mean(0)
 with np.load(out/'linear_oof.npz') as z:lin=base+.25*7.9411391*z['embedding_all_100'];vlin=z['embedding_all_100_val']
 with np.load(out/'role_oof.npz') as z:ro=z['train'];vro=z['validation']
 with np.load(out/'critic_oof.npz') as z:cri=base+.25*128.0030667*z['0.1']
 with np.load(out/'expanded_oof.npz') as z:exp=base+.25*z['all_0.1']
 with np.load(out/'small_program_oof.npz') as z:small=base+2*z['10.0']
 with np.load(out/'neural_oof.npz') as z:neural=base+.5*.3699988662*z['equal'];vneural=z['equal_val']
 with np.load(out/'path_critic_oof.npz') as z:path=base+.1*7.43765021*z['0.001']
 choices={'independent':[ind],'diverse':[ind,lin,ro],'all':[ind,lin,ro,cri,exp,small,neural,path]};rr=[]
 for k,models in choices.items():
  v=votes(b,models)
  for alpha in [.025,.05,.1,.2,.4,.8]:rr.append({'view':k,'alpha':alpha,**stats(base+alpha*v,y,h,base)})
 rr.append({'view':'none','alpha':0.,**stats(base,y,h,base)});best=max(rr,key=lambda r:(r['hard_correct'],-r['alpha']));(out/'vote_search.json').write_text(json.dumps({'selected':best,'records':rr,'per_question_oof':True,'fully_nested':False},indent=2)+'\n');print(json.dumps({'selected':best}),flush=True)
 if best['view']=='none':return
 vb=load_bank('validation',data);vy=labels(vb,read_jsonl(data/'gsm8k_validation.jsonl'));vh=hard(vb)
 with np.load(p/'solution_v9/runs/validation_scores.npz') as z:vbase=z['selected']
 with np.load(out/'extra_validation_scores.npz') as z:vcri=vbase+.25*128.0030667*z['critic'];vsmall=vbase+2*z['small_program']
 with np.load(out/'expanded_validation_scores.npz') as z:vexp=vbase+.25*z['delta']
 # The path expert may be added to validation once its heldout score is exported.
 choices={'independent':[vind],'diverse':[vind,vbase+.25*7.9411391*vlin,vro]}
 if best['view'] not in choices:
  print(json.dumps({'status':'selected view needs validation path expert score'}));return
 pred=vbase+best['alpha']*votes(vb,choices[best['view']]);result={'selected':best,'validation_base':stats(vbase,vy,vh,vbase),'validation_candidate':stats(pred,vy,vh,vbase)};(out/'vote_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
