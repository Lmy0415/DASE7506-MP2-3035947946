import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json
import numpy as np
from common import load_inputs,read_jsonl,parse_answer
from features import labels
from legacy_submission import member_scores
root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data'
b,a=load_inputs('validation',data);refs=read_jsonl(data/'gsm8k_validation.jsonl');y=labels(b,refs);members=member_scores(b,a,root.parent/'solution/runs/model_bundle');s=np.mean(members,0);records=[]
for i,r in enumerate(b):
 j=int(s[i].argmax())
 if y[i,j]:continue
 k=max(np.flatnonzero(y[i]),key=lambda c:s[i,c]);answers=[parse_answer(c['text']) for c in r['candidates']]
 records.append({'error':len(records)+1,'row':i,'id':r['id'],'question':r['question'],'reference':refs[i]['answer'],'gold':parse_answer(refs[i]['answer']),'selected':j,'best_correct':int(k),'parsed_answers':answers,'scores':s[i].tolist(),'member_choices':[int(z[i].argmax()) for z in members],'candidates':r['candidates']})
(root/'runs/errors.json').write_text(json.dumps(records,indent=2)+'\n')
with (root/'runs/errors_readable.txt').open('w') as f:
 for r in records:
  f.write(f"\nERROR {r['error']} ROW {r['row']} SELECTED {r['selected']} CORRECT {r['best_correct']} ANSWERS {r['parsed_answers']} MEMBERS {r['member_choices']}\nQ: {r['question']}\nREF: {r['reference']}\nWRONG: {r['candidates'][r['selected']]['text']}\nRIGHT: {r['candidates'][r['best_correct']]['text']}\n")
print({'errors':len(records),'candidate_fields':list(b[0]['candidates'][0]),'code_example':b[0]['candidates'][0].get('code'),'readable_chars':(root/'runs/errors_readable.txt').stat().st_size})
