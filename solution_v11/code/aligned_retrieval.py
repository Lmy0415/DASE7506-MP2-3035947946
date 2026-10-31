"""Retrieve reference programs with learned quantity-context alignment."""
import re
from fractions import Fraction
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from programs import question_numbers,induce,evaluate,slots
from quantity_relations import slot_texts
from reference_patterns import text_view
from common import parse_answer

def build(bank,refs):
 texts=[];st=[];programs=[];slotrows=[]
 for row,ref in zip(bank,refs):
  q,_=question_numbers(row['question'])
  if len(q)<2 or len(q)>10 or len(q)!=len(set(q)):continue
  node=induce(row['question'],ref['answer'])
  if node is None:continue
  used=sorted(slots(node))
  if not used or len(used)>8:continue
  texts.append(text_view(row['question']));ss=slot_texts(row);slotrows.append(np.arange(len(st),len(st)+len(ss)));st.extend(ss);programs.append((node,used,len(q)))
 qvec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=14000,stop_words='english',sublinear_tf=True,dtype=np.float32)
 svec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=14000,sublinear_tf=True,dtype=np.float32)
 qm=qvec.fit_transform(texts);sm=svec.fit_transform(st)
 return {'qvec':qvec,'svec':svec,'qmatrix':qm,'smatrix':sm,'slotrows':slotrows,'programs':programs}

def features(bank,lib,top=18):
 if not bank:return np.empty((0,4,10))
 q=lib['qvec'].transform([text_view(r['question']) for r in bank]);allslots=[slot_texts(r) for r in bank];flat=[t for row in allslots for t in row];sm=lib['svec'].transform(flat) if flat else None;offs=np.cumsum([0]+[len(a) for a in allslots]);allout=[]
 for start in range(0,len(bank),128):
  qsim=(q[start:start+128]@lib['qmatrix'].T).toarray()
  for ii,row in enumerate(bank[start:start+128]):
   i=start+ii;vv,_=question_numbers(row['question']);n=len(vv);out=[]
   if n==0 or n>12:
    allout.append(np.zeros((4,10)));continue
   sims=qsim[ii];k=min(top,len(sims));ids=np.argpartition(sims,-k)[-k:];ids=ids[np.argsort(sims[ids])[::-1]];props=[];qs=sm[offs[i]:offs[i+1]]
   for j in ids:
    qscore=float(sims[j]);node,used,nn=lib['programs'][j]
    if qscore<.12 or len(used)>n:continue
    reference_rows=lib['slotrows'][j][used];similarities=(qs@lib['smatrix'][reference_rows].T).toarray()
    r,c=linear_sum_assignment(-similarities);mapping={used[col]:r0 for r0,col in zip(r,c)};pair=[similarities[mapping[k],i0] for i0,k in enumerate(used)];mean=float(np.mean(pair));minimum=float(np.min(pair))
    if minimum<.08:continue
    args=[Fraction(0)]*nn
    for old,new in mapping.items():args[old]=vv[new]
    try:a=str(evaluate(node,args))
    except (ValueError,IndexError,ZeroDivisionError,OverflowError,RecursionError):continue
    weight=(qscore**2)*(mean**2);props.append((a,weight,qscore,mean,minimum))
   answers=[parse_answer(c['text']) for c in row['candidates']];valid=[p for p in props if p[0] in answers]
   for a in answers:
    yes=[p for p in valid if p[0]==a and a is not None];no=[p for p in valid if p[0]!=a];ym=sum(z[1] for z in yes);nm=sum(z[1] for z in no)
    out.append([ym,nm,max((z[1] for z in yes),default=0),max((z[1] for z in no),default=0),ym/max(ym+nm,1e-8),max((z[2] for z in yes),default=0),max((z[3] for z in yes),default=0),max((z[4] for z in yes),default=0),len(yes)/top,len(no)/top])
   allout.append(out)
 return np.asarray(allout,dtype=np.float64).reshape(-1,4,10)
