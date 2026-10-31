"""Frozen legacy verifier with a training-calibrated calculation evidence correction."""
from pathlib import Path
import numpy as np
from legacy_submission import member_scores
from calculation_evidence import evidence_features

def score_components(bank,arrays,checkpoint=None):
    root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/model_bundle'
    base=np.mean(member_scores(bank,arrays,root/'legacy'),0)
    with np.load(root/'correction.npz',allow_pickle=False) as z:
        x,_=evidence_features(bank,local_gate=bool(z['local_gate']))
        delta=x[:,:,z['columns']]/z['sd']@z['weight']
        selected=base*z['temperature']+delta*z['strength']
    return {'legacy':base,'delta':delta,'selected':selected}

def predict(bank,arrays,checkpoint=None):
    if not bank:return []
    return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
