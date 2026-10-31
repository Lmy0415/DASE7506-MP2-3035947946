"""Recreate v7 reference-quantity expert as per-question OOF scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,joblib,numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
project=Path(__file__).resolve().parents[2];sys.path.append(str(project/'solution_v6/code'))
from common import load_bank,read_jsonl
from features import labels
from train_evidence import fit_correction,apply,hard
root=project/'solution_v11';data=project/'DASE7506-MP2-main/code/data';b=load_bank('train',data);y=labels(b,read_jsonl(data/'gsm8k_train.jsonl'));h=hard(b);x,base=joblib.load(project/'solution_v7/runs/full_features.joblib');s=np.zeros_like(base)
for ti,hi in StratifiedKFold(5,shuffle=True,random_state=20261103).split(np.arange(len(b)),h):
 st=fit_correction(x[ti],base[ti],y[ti],.01,list(range(7)));s[hi]=apply(x[hi],base[hi],st)
with np.load(project/'solution_v7/runs/validation_scores.npz') as z:v=z['relation']
np.savez_compressed(root/'runs/role_oof.npz',train=s,validation=v)
