"""v6 verifier plus a train-only target binding rank correction."""
from pathlib import Path
import numpy as np
from target_features import features
import base_submission

def score_components(bank,arrays,checkpoint=None):
 root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/model_bundle'
 base=base_submission.score_components(bank,arrays,root/'base')['selected']
 with np.load(root/'target.npz',allow_pickle=False) as z:
  x=features(bank,str(z['view']));delta=(x@z['weight']).reshape(-1,4);selected=base+float(z['strength'])*delta
 return {'base':base,'delta':delta,'selected':selected}

def predict(bank,arrays,checkpoint=None):
 if not bank:return []
 return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
