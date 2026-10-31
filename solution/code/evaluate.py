"""Freeze assets before first test score and time fresh official processes."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--checkpoint',type=Path,default=Path('../runs/model_bundle'));p.add_argument('--output',type=Path,default=Path('../runs'));a=p.parse_args()
code=Path(__file__).resolve().parent; out=a.output.resolve();out.mkdir(exist_ok=True,parents=True); checkpoint=a.checkpoint.resolve();data=a.data_dir.resolve()
files=sorted([code/'submission.py',code/'features.py',code/'train.py',*checkpoint.iterdir()])
hashes={str(f.relative_to(code.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
freeze=out/'FREEZE.json'
if freeze.exists():
 assert json.loads(freeze.read_text())['sha256']==hashes,'Frozen method changed; do not retune using test scores.'
else:freeze.write_text(json.dumps({'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,'asset_bytes':sum(f.stat().st_size for f in checkpoint.iterdir()),'selection':'train fitted; validation selected; no test reference access before freeze'},indent=2)+'\n')
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='4'
commands=[ [sys.executable,str(code/'predict.py'),'--method',str(code/'submission.py'),'--checkpoint',str(checkpoint),'--split',a.split,'--data-dir',str(data),'--output',str(out/f'{a.split}-predictions.json')], [sys.executable,str(code/'score_predictions.py'),'--split',a.split,'--data-dir',str(data),'--predictions',str(out/f'{a.split}-predictions.json'),'--output',str(out/f'{a.split}-score.json')]]
results=[];start=time.perf_counter()
for command in commands:
 t=time.perf_counter();subprocess.run(command,env=env,check=True);results.append(time.perf_counter()-t)
result={'split':a.split,'prediction_process_wall_seconds':results[0],'scoring_process_wall_seconds':results[1],'total_process_wall_seconds':time.perf_counter()-start,'platform':platform.platform(),'local_cpu':'Apple Silicon arm64 (exact chip unavailable in sandbox)','threads_per_library':4,'reference_cpu_eligibility':'Not measured on Xeon Platinum 8457C; local timing only','commands':commands}
(out/f'{a.split}-resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
