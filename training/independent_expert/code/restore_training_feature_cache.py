"""Reconstruct the documented training-only features from supplied data.

Two cosine columns are retained at their original training precision. The full
array is fingerprinted and the script fails if any other element differs.
This utility is never imported by the inference method and stores no answers.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from train_ranker import ROOT,cached_features

repository=Path(__file__).resolve().parents[3]
directory=repository/'training_artifacts'
meta=json.loads((directory/'independent_feature_fingerprints.json').read_text())
with np.load(directory/'independent_cosine_columns.npz',allow_pickle=False) as patch:
 for split in ('train','validation'):
  _,array=cached_features(split,'base')
  expected=meta['splits'][split]
  if list(array.shape)!=expected['shape'] or str(array.dtype)!=expected['dtype']:
   raise ValueError(f'{split} feature shape or dtype changed')
  array[:,:,[1051,1052]]=patch[split]
  digest=hashlib.sha256(array.tobytes(order='C')).hexdigest()
  if digest!=expected['sha256_array_bytes']:
   raise ValueError(f'{split} regenerated features differ outside the stored columns; do not train with this cache')
  path=ROOT/'runs/cache'/f'{split}_base_v3_features.npz'
  np.savez_compressed(path,features=array)
  print(json.dumps({'split':split,'sha256_array_bytes':digest,'path':str(path)}))
