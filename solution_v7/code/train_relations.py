"""Train-only cross-fitting for required quantities, signed roles and goal operations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl
from features import labels
from quantity_relations import candidate_graph,candidate_features,reference_targets
from relation_models import fit_models,predictions
from train_evidence import fit_correction,apply,metrics,hard
from base_submission import score_components

def main():
 root=Path(__file__).resolve().parent.parent;out=root/'runs';data=root.parent/'DASE7506-MP2-main/code/data';start=time.perf_counter()
 b,ar=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');y=labels(b,refs);h=hard(b)
 cache=out/'graphs.joblib'
 if cache.exists():graphs=joblib.load(cache)
 else:
  graphs=[[candidate_graph(r,c['text']) for c in r['candidates']] for r in b];joblib.dump(graphs,cache,compress=0)
 gold_roles,gold_profiles=reference_targets(b,refs);stats={'reference_programs':sum(t is not None for t in gold_profiles),'role_labels':int(sum((t>=0).sum() for t in gold_roles if t is not None)),'candidate_graphs':sum(g['available'] for row in graphs for g in row),'candidate_count':len(b)*4};print(json.dumps({'coverage':stats}),flush=True)
 with np.load(root.parent/'solution_v6/runs/train_evidence.npz') as z:calc=z['features'];np.testing.assert_array_equal(y,z['labels'])
 with np.load(root.parent/'solution_v6/runs/nested_base.npz') as z:tr=z['train_ids'];ho=z['holdout_ids'];raw_inner=z['inner'];raw_outer=z['outer']
 plan=json.loads((root.parent/'solution_v6/runs/training_plan.json').read_text())['selected'];assert plan['gate']=='group'
 def corrected_base(x,s,t,ex,es):
  st=fit_correction(x,s,t,plan['l2'],plan['columns']);return apply(ex,es,st,plan['strength'])
 def cross_role(ids,seed):
  rr=[None]*len(ids);pp=np.zeros((len(ids),9));bb=np.zeros((len(ids),4));raw=raw_inner if len(ids)==len(tr) else full_raw
  for fold,(ti,hi) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(ids,h[ids])):
   model=fit_models([b[i] for i in ids[ti]],[refs[i] for i in ids[ti]]);rp,pr=predictions(model,[b[i] for i in ids[hi]])
   for k,r in zip(hi,rp):rr[k]=r
   pp[hi]=pr;bb[hi]=corrected_base(calc[ids[ti]],raw[ti],y[ids[ti]],calc[ids[hi]],raw[hi]);print(json.dumps({'crossfit_questions':len(ids),'fold':fold}),flush=True)
  xx,_=candidate_features([b[i] for i in ids],rr,pp,[graphs[i] for i in ids]);return xx,bb,rr,pp
 internal=out/'internal_features.joblib'
 if internal.exists():ix,ibase,hx,hbase=joblib.load(internal)
 else:
  ix,ibase,irp,ipp=cross_role(tr,20261020)
  m=fit_models([b[i] for i in tr],[refs[i] for i in tr]);hr,hp=predictions(m,[b[i] for i in ho]);hx,_=candidate_features([b[i] for i in ho],hr,hp,[graphs[i] for i in ho]);hbase=corrected_base(calc[tr],raw_inner,y[tr],calc[ho],raw_outer)
  joblib.dump((ix,ibase,hx,hbase),internal,compress=0)
 results=[];options=[];baseline=metrics(hbase,y[ho],h[ho]);print(json.dumps({'internal_v6_baseline':baseline}),flush=True)
 for view,columns in [('ledger',list(range(7))),('roles',[0,1,2,3,4,5,6,7,8,9,10,21,22]),('all',list(range(23)))]:
  for l2 in [.001,.01,.1]:
   st=fit_correction(ix,ibase,y[tr],l2,columns)
   for strength in [.25,.5,1.]:
    ss=apply(hx,hbase,st,strength);r={'view':view,'columns':columns,'l2':l2,'strength':strength,**metrics(ss,y[ho],h[ho])};results.append(r);options.append(r)
 best=max(options,key=lambda r:(r['hard_correct'],r['correct'],-len(r['columns']),r['l2'],-r['strength']));(out/'training_plan.json').write_text(json.dumps({'selected':best,'v6_baseline':baseline,'outer_holdout_labels_excluded_from_all_fits':True,'outer_split_source':'v6 seed 20261019','role_crossfit_seed':20261020,'reference_targets':'current question quantity sensitivity sign and operation profile'},indent=2)+'\n');(out/'internal_search.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'plan':best}),flush=True)
 with np.load(root.parent/'solution_v4/runs/oof/oof_scores.npz') as z:full_raw=z['scores'];np.testing.assert_array_equal(y,z['labels'])
 full_cache=out/'full_features.joblib'
 if full_cache.exists():fx,fbase=joblib.load(full_cache)
 else:
  fx,fbase,_,_=cross_role(np.arange(len(b)),20261021);joblib.dump((fx,fbase),full_cache,compress=0)
 state=fit_correction(fx,fbase,y,best['l2'],best['columns']);state['strength']=np.array(best['strength']);np.savez(out/'relation_correction.npz',**state)
 model=fit_models(b,refs);joblib.dump(model,out/'quantity_model.joblib',compress=0)
 bv,av=load_inputs('validation',data);yv=labels(bv,read_jsonl(data/'gsm8k_validation.jsonl'));hh=hard(bv);rv,pv=predictions(model,bv);vx,vg=candidate_features(bv,rv,pv);base=score_components(bv,av,root.parent/'solution_v6/runs/model_bundle')['selected'];ss=apply(vx,base,state,best['strength']);rows=[]
 # Predefined validation comparison, no test references or test error analysis.
 for scale in [0,.25,.5,1.]:
  pred=apply(vx,base,state,best['strength']*scale);rows.append({'relation_strength_scale':scale,**metrics(pred,yv,hh)})
 result={'v6_baseline':metrics(base,yv,hh),'relation':metrics(ss,yv,hh),'predefined_strengths':rows,'coverage':stats,'training_seconds':time.perf_counter()-start,'test_access':'none','model_training':{k:model[k] for k in ['fit_question_n','reference_program_n','role_example_n','role_class_counts']}}
 np.savez_compressed(out/'validation_scores.npz',baseline=base,relation=ss,features=vx,labels=yv,hard=hh);joblib.dump((rv,pv,vg),out/'validation_relations.joblib',compress=0);(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
