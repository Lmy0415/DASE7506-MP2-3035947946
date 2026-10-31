"""Check the original v11 source and inference asset SHA-256 manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'evidence/frozen/FREEZE.json').read_text())['sha256']
base=root/'solution_v11';missing=[];bad=[]
for name,expected in manifest.items():
 path=base/name
 if not path.is_file():missing.append(name)
 elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(name)
print(json.dumps({'checked':len(manifest),'missing':missing,'mismatched':bad},indent=2))
if missing or bad:raise SystemExit(1)
