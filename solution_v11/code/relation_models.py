"""Sparse supervised quantity-role and operation-profile models, no pretrained assets."""
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,Ridge
from quantity_relations import slot_texts,question_text,reference_targets

def fit_models(bank,refs,binary=False):
 roles,profiles=reference_targets(bank,refs);texts=[];targets=[];qt=[];pt=[]
 for row,rr,pp in zip(bank,roles,profiles):
  if rr is not None:
   for t,v in zip(slot_texts(row),rr):
    if v>=0:texts.append(t);targets.append(v)
  if pp is not None:qt.append(question_text(row)[3]);pt.append(pp)
 v=TfidfVectorizer(max_features=8192,ngram_range=(1,2),min_df=2,sublinear_tf=True,dtype=np.float32);x=v.fit_transform(texts)
 m=LogisticRegression(C=4.,solver='lbfgs',max_iter=300,tol=1e-5).fit(x,(np.array(targets)>0).astype(int) if binary else np.array(targets))
 qv=TfidfVectorizer(max_features=8192,ngram_range=(1,2),min_df=2,sublinear_tf=True,dtype=np.float32);qx=qv.fit_transform(qt);qm=Ridge(alpha=10,solver='lsqr').fit(qx,np.array(pt,dtype=np.float32))
 return {'role_vectorizer':v,'role_model':m,'profile_vectorizer':qv,'profile_model':qm,'fit_question_n':len(bank),'reference_program_n':len(pt),'role_example_n':len(targets),'role_class_counts':np.bincount(targets,minlength=3).tolist(),'binary_required':binary}

def predictions(model,bank):
 texts=[t for row in bank for t in slot_texts(row)];probs=model['role_model'].predict_proba(model['role_vectorizer'].transform(texts)) if texts else np.empty((0,3))
 if model.get('binary_required',False):
  assert np.array_equal(model['role_model'].classes_,[0,1]);probs=np.c_[probs[:,0],probs[:,1]/2,probs[:,1]/2]
 else:assert np.array_equal(model['role_model'].classes_,[0,1,2])
 rp=[];i=0
 for row in bank:
  n=len(slot_texts(row));rp.append(probs[i:i+n]);i+=n
 if 'profile_model' in model:
  pp=model['profile_model'].predict(model['profile_vectorizer'].transform([question_text(row)[3] for row in bank]));pp[:,:5]=softmax(pp[:,:5]*4,axis=1);pp[:,5:]=pp[:,5:].clip(0,5)
 else:
  pp=np.zeros((len(bank),9),dtype=np.float32);pp[:,:5]=.2
 return rp,pp
