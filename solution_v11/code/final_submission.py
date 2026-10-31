"""Train-calibrated six-expert answer selector; supplied inputs and assets only."""
from pathlib import Path
import numpy as np,joblib
from scipy.special import expit
from common import parse_answer
from features import features
from calculation_evidence import evidence_features
from expanded_calculations import features as expanded_features
from independent_features import make_features
from independent_ranker import Ranker
from relation_models import predictions
from quantity_relations import candidate_features
from program_critic import matrices
from target_submission import score_components as target_components
from submission import constrain
from gate_features import make

def score_components(bank,arrays,checkpoint=None):
 root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent.parent/'runs/final_bundle'
 if not bank:return {'base':np.empty((0,4)),'selected':np.empty((0,4))}
 r=target_components(bank,arrays,root/'v9/target_base');v6=r['base'];base=constrain(bank,r['selected']);e=np.asarray(arrays['embeddings'],dtype=np.float32);hand=features(bank,e);calc=evidence_features(bank)[0]
 fi=make_features(bank,e,'base');ind=np.mean([Ranker.load(path).scores(fi) for path in sorted((root/'independent').glob('member_*.npz'))],axis=0)
 with np.load(root/'linear.npz',allow_pickle=False) as z:
  x=e-e.mean(1,keepdims=True);lin=np.mean([(x/z['sd'][i])@z['weight'][i] for i in range(len(z['weight']))],axis=0);lin=base+.25*7.9411391*lin
 rm=joblib.load(root/'role_model.joblib');rp,pp=predictions(rm,bank);fx,graphs=candidate_features(bank,rp,pp)
 with np.load(root/'role_correction.npz',allow_pickle=False) as z:role=v6*z['temperature']+z['strength']*(fx[:,:,z['columns']]/z['sd']@z['weight'])
 gx,_=matrices(bank,graphs)
 with np.load(root/'critic.npz',allow_pickle=False) as z:
  raw=np.asarray(gx@z['weight']).reshape(-1,4);available=(gx.getnnz(1)>0).reshape(-1,4);mean=(raw*available).sum(1,keepdims=True)/available.sum(1,keepdims=True).clip(1);critic=base+float(z['strength'])*float(z['scale'])*(raw-mean)*available
 ex=expanded_features(bank)[:,:,:12]
 with np.load(root/'expanded.npz',allow_pickle=False) as z:extra=base+float(z['strength'])*((ex/z['sd'])@z['weight'])
 models=[base,ind,lin,role,critic,extra]
 with np.load(root/'gate.npz',allow_pickle=False) as z:
  x,ii,eligible=make(models,bank,hand,calc,z['model_scales']);flat=x.reshape(-1,x.shape[-1]);prob=[]
  for mean,scale,weight,bias in zip(z['means'],z['scales'],z['weights'],z['bias']):prob.append(expit(((flat-mean)/scale)@weight+bias).reshape(-1,4))
  prob=np.mean(prob,axis=0);prob[~eligible]=0;best=prob.argmax(1);active=prob.max(1)>=float(z['threshold'])
 selected=base.copy()
 for i in np.flatnonzero(active):
  answer=parse_answer(bank[i]['candidates'][int(best[i])]['text'])
  for j,c in enumerate(bank[i]['candidates']):
   if answer is not None and parse_answer(c['text'])==answer:selected[i,j]=base[i].max()+1.
 return {'base':base,'selected':selected,'gate_probability':prob,'independent':ind,'linear':lin,'role':role,'critic':critic,'expanded':extra}

def predict(bank,arrays,checkpoint=None):
 if not bank:return []
 return [int(i) for i in score_components(bank,arrays,checkpoint)['selected'].argmax(1)]
