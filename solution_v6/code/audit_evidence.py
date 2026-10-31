import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json
from common import load_inputs,read_jsonl
from features import labels
from calculation_evidence import evidence_features
root=Path(__file__).resolve().parent.parent;data=root.parent/'DASE7506-MP2-main/code/data';errors=json.loads((root/'runs/errors.json').read_text());b,a=load_inputs('validation',data);x,d=evidence_features(b)
records=[]
for r in errors:
 row=d[r['row']];z=row[r['selected']];records.append({'error':r['error'],'row':r['row'],'gold':r['gold'],'selected_answer':z['answer'],'selected_evidence':z,'all_repaired':[v['repaired_answer'] for v in row],'all_errors':[len(v['errors']) for v in row]})
(root/'runs/validation_calculation_audit.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:
 if r['selected_evidence']['errors'] or r['selected_evidence']['box_mismatch']:print(json.dumps(r))
