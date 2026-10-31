"""Hash-freeze source and assets, then time fresh official prediction/scoring processes."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],required=True);a=p.parse_args()
code=Path(__file__).resolve().parent;root=code.parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);bundle=a.checkpoint.resolve()
files=sorted([*code.glob('*.py'),*(p for p in bundle.rglob('*') if p.is_file())]);hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};path=out/'FREEZE.json'
if path.exists():assert json.loads(path.read_text())['sha256']==hashes,'Frozen method has changed.'
else:path.write_text(json.dumps({'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,'asset_bytes':sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),'test_used_for_selection':False},indent=2)+'\n')
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
env.setdefault('MPLCONFIGDIR','/private/tmp/mp2-mpl-cache')
commands=[[sys.executable,str(code/'predict.py'),'--method',str(code/'final_submission.py'),'--checkpoint',str(bundle),'--data-dir',str(a.data_dir.resolve()),'--split',a.split,'--output',str(out/f'{a.split}-predictions.json')],[sys.executable,str(code/'score_predictions.py'),'--data-dir',str(a.data_dir.resolve()),'--split',a.split,'--predictions',str(out/f'{a.split}-predictions.json'),'--output',str(out/f'{a.split}-score.json')]]
start=time.perf_counter();elapsed=[]
for command in commands:
 tick=time.perf_counter();subprocess.run(command,env=env,check=True);elapsed.append(time.perf_counter()-tick)
result={'split':a.split,'prediction_process_seconds':elapsed[0],'scoring_process_seconds':elapsed[1],'total_wall_seconds':time.perf_counter()-start,'threads_per_library':4,'processes':'sequential','platform':platform.platform(),'hardware':'Apple Silicon arm64; exact chip unavailable in sandbox','reference_xeon_runtime_measured':False,'commands':commands};(out/f'{a.split}-resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
