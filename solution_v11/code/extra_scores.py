"""Compute validation scores from models selected on training OOF."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,joblib
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl
from program_critic import matrices
from train_small_program import train
from small_program_solver import score
p=Path(__file__).resolve().parents[2];out=p/'solution_v11/runs';data=p/'DASE7506-MP2-main/code/data';vb,_=load_inputs('validation',data)
_,_,vg=joblib.load(p/'solution_v7/runs/validation_relations.joblib');vx,_=matrices(vb,vg)
with np.load(out/'critic_model.npz') as z:vr=np.asarray(vx@z['weight']).reshape(-1,4)
available=(vx.getnnz(1)>0).reshape(-1,4);center=np.sum(vr*available,1,keepdims=True)/available.sum(1,keepdims=True).clip(1);vr=(vr-center)*available
b,_=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl');v,m=train(b,refs,np.arange(len(b)),10.);sp=score(vb,v,m)
np.savez_compressed(out/'extra_validation_scores.npz',critic=vr,small_program=sp)
