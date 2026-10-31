"""Verify the frozen assets and documented training-path adjustments."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'evidence/frozen/FREEZE.json').read_text())['sha256']
changes=json.loads((root/'evidence/training_path_changes.json').read_text())['files']
adjusted={x['path']:x for x in changes};base=root/'solution_v11';missing=[];bad=[]
for name,expected in manifest.items():
 path=base/name
 if not path.is_file():missing.append(name)
 else:
  actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if name in adjusted:
   change=adjusted[name]
   if change['original_sha256']!=expected or actual!=change['current_sha256']:bad.append(name)
  elif actual!=expected:bad.append(name)
print(json.dumps({'checked':len(manifest),'documented_training_path_adjustments':len(adjusted),'missing':missing,'mismatched':bad},indent=2))
if missing or bad:raise SystemExit(1)
