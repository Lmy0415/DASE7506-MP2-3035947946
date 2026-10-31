"""Compact trained verifier. Inference opens only the explicitly supplied assets."""
import json
from pathlib import Path
import numpy as np
from features import features,lexical_texts

def member_scores(bank,arrays,checkpoint):
    root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/model_bundle'
    config=json.loads((root/'config.json').read_text())
    e=np.asarray(arrays['embeddings'],dtype=np.float32);f=None;outputs=[]
    for meta in config['members']:
        with np.load(root/meta['file'],allow_pickle=False) as data:
            a={k:data[k] for k in data.files}
        if meta['kind']=='lexical':
            from sklearn.feature_extraction.text import HashingVectorizer
            hv=HashingVectorizer(n_features=len(a['weight']),ngram_range=(1,2),alternate_sign=True,norm='l2')
            s=(hv.transform(lexical_texts(bank,meta['full']))@a['weight']+a['bias'][0]).reshape(-1,4)
        else:
            mode=meta['mode']
            if mode!='embedding' and f is None:f=features(bank,e)
            x=e if mode=='embedding' else f if mode=='style' else np.concatenate([e,e-e.mean(1,keepdims=True),f],-1)
            x=(x-a['mu'])/a['sd']
            if meta['kind']=='linear': s=x@a['weight']+a['bias'][0]
            else:
                for k in range(3):
                    x=x@a[f'w{k}']+a[f'b{k}']
                    if k<2:x=np.maximum(x,0)
                s=x.squeeze(-1)
        outputs.append(s)
    return outputs

def predict(bank,arrays,checkpoint):
    if not bank:return []
    scores=np.mean(member_scores(bank,arrays,checkpoint),axis=0)
    return [int(i) for i in scores.argmax(1)]
