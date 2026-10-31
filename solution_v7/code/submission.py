"""Frozen v6 verifier plus training-derived required-quantity evidence."""
from pathlib import Path
import numpy as np
import joblib
from base_submission import score_components as base_components
from relation_models import predictions
from quantity_relations import candidate_features

def score_components(bank,arrays,checkpoint=None):
    root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/model_bundle'
    base=base_components(bank,arrays,root/'base')['selected'];model=joblib.load(root/'quantity_model.joblib');rp,pp=predictions(model,bank);x,_=candidate_features(bank,rp,pp)
    with np.load(root/'relation_correction.npz',allow_pickle=False) as z:
        delta=x[:,:,z['columns']]/z['sd']@z['weight'];s=base*z['temperature']+z['strength']*delta
    return {'baseline':base,'quantity_delta':delta,'selected':s}

def predict(bank,arrays,checkpoint=None):
    if not bank:return []
    return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
