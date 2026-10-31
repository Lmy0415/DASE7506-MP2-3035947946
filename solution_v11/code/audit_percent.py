import json,collections
from pathlib import Path
from common import read_jsonl,parse_answer
from percent_patterns import hypotheses
root=Path(__file__).resolve().parents[1];data=root.parent/'DASE7506-MP2-main/code/data';refs=read_jsonl(data/'gsm8k_train.jsonl');out=collections.defaultdict(lambda:{'covered':0,'correct':0,'wrong_examples':[]})
for r in refs:
 for kind,expected in hypotheses(r['question']):
  d=out[kind];d['covered']+=1;ok=expected==parse_answer(r['answer']);d['correct']+=ok
  if not ok:d['wrong_examples'].append({'question':r['question'],'expected':expected,'reference':parse_answer(r['answer'])})
(root/'runs/percent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
