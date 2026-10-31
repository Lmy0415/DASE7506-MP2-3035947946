"""Validation-only diagnostics; test results are read solely from official score JSON."""
import argparse,json
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl,parse_answer
from features import labels
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
bank,arrays=load_inputs('validation',a.data_dir);refs=read_jsonl(a.data_dir/'gsm8k_validation.jsonl');y=labels(bank,refs)
with np.load(a.runs/'validation_scores.npz') as z:scores={k:z[k] for k in z.files}
with np.load(a.runs/'development_labels.npz') as z:hard=z['hard']
s=scores['selected_ensemble'];chosen=s.argmax(1);correct=y[np.arange(len(y)),chosen]
baseline=y[np.arange(len(y)),scores['consensus'].argmax(1)]
rng=np.random.default_rng(20261016);bootstrap=[]
for mask in [np.ones(len(y),dtype=bool),hard]:
 diff=(correct-baseline)[mask];sample=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(1)*100
 bootstrap.append({'n':len(diff),'gain_pp':float(diff.mean()*100),'paired_bootstrap_95pct_interval_pp':np.percentile(sample,[2.5,97.5]).tolist()})
# Margin only measures relative confidence; it is not a calibrated probability.
margin=np.sort(s,axis=1)[:,-1]-np.sort(s,axis=1)[:,-2]
bins=[]
for ids in np.array_split(np.argsort(margin[hard]),4):
 idx=np.where(hard)[0][ids];bins.append({'n':len(idx),'mean_margin':float(margin[idx].mean()),'accuracy':float(correct[idx].mean()*100)})
examples=[]
for idx in np.argsort(-margin):
 if hard[idx] and not correct[idx] and len(examples)<3:
  row=bank[idx];gold=parse_answer(refs[idx]['answer']);win=int(chosen[idx]);good=int(np.where(y[idx]>0)[0][0]);examples.append({'question':row['question'],'chosen_index':win,'chosen_answer':parse_answer(row['candidates'][win]['text']),'reference_answer':gold,'chosen_text':row['candidates'][win]['text'],'correct_candidate_text':row['candidates'][good]['text'],'margin':float(margin[idx])})
result={'bootstrap_seed':20261016,'bootstrap_replicates':10000,'bootstrap_vs_consensus':bootstrap,'hard_margin_quartiles':bins,'high_margin_validation_errors':examples,'validation_single_answer_n':int((~hard).sum()),'validation_hard_n':int(hard.sum()),'search_entries':len(json.loads((a.runs/'search.json').read_text()))-1,'caution':'Validation diagnostics reuse selection data and are optimistic; bootstrap intervals do not account for search selection.'}
(a.runs/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='high_margin_validation_errors'},indent=2))
