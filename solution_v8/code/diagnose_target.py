"""Validation-only qualitative binding diagnostics, without test access."""
import json
from pathlib import Path
from common import load_inputs
from symbolic_binding import answer_binding
from provenance import used_quantities
root=Path(__file__).resolve().parent.parent;b,_=load_inputs('validation',root.parent/'DASE7506-MP2-main/code/data');changes=json.loads((root/'runs/validation_changes.json').read_text());cases=[];coverage={'candidates':len(b)*4,'bound':0,'with_provenance':0,'ambiguous_quantity':0,'with_resolved_symbols':0}
for row in b:
 for c in row['candidates']:
  z=answer_binding(row,c);u,a,_=used_quantities(row['question'],z)
  coverage['bound']+=bool(z['bound']);coverage['with_provenance']+=bool(u);coverage['ambiguous_quantity']+=bool(a);coverage['with_resolved_symbols']+=bool(z['resolved'])
for r in changes:
 if r['old_correct']==r['new_correct']:continue
 row=b[r['row']];record={**r,'question':row['question'],'bindings':[]}
 for k in [r['old'],r['new']]:
  z=answer_binding(row,row['candidates'][k]);u,a,slots=used_quantities(row['question'],z)
  record['bindings'].append({'candidate':k,'answer':z['answer'],'goal':z['goal'],'context':z['context'],'bound':z['bound'],'used_slots':sorted(u),'ambiguous_slots':sorted(a),'quantity_context':slots})
 cases.append(record)
result={'split':'validation','coverage':coverage,'fixes':sum(not r['old_correct'] and r['new_correct'] for r in changes),'regressions':sum(r['old_correct'] and not r['new_correct'] for r in changes),'changed_accuracy_cases':cases}
(root/'runs/diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='changed_accuracy_cases'}))
for c in cases:print(json.dumps({'row':c['row'],'question':c['question'],'fix':c['new_correct']}))
