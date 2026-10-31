"""Deterministic target/answer-context features; unknown bindings carry no penalty."""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from symbolic_binding import answer_binding,words
from provenance import used_quantities
CONCEPTS=['profit','revenue','cost','change','remaining','difference','per','total','time','age']
SIZE=16384

def features(bank,view='joint'):
 dictionaries=[];numeric=[]
 for row in bank:
  for candidate in row['candidates']:
   b=answer_binding(row,candidate);bound=b['bound'];g=b['goal_terms'];c=b['context_terms'];gc=b['goal_concepts'];ac=b['answer_concepts']
   n=[float(bound and bool(g&c)),len(g&c)/max(1,len(g)) if bound else 0,len(g&c)/max(1,len(c)) if bound else 0]
   n += [float(bound and k in gc and k in ac) for k in CONCEPTS]
   n += [float(bound and k in gc and bool(ac) and k not in ac) for k in CONCEPTS]
   n += [float(bound and k in ac and k not in gc) for k in CONCEPTS]
   used,ambiguous,slots=used_quantities(row['question'],b)
   relevant=[(i,set(words(t))) for i,t in slots]
   overlap=sum(bool(g&terms) for i,terms in relevant if i in used)
   n += [float(overlap>0),overlap/max(1,len(used)),len(used)/max(1,len(slots)),float(bool(used) and not ambiguous)]
   numeric.append(n)
   d={}
   if bound:
    gg=sorted(g)[:24];cc=sorted(c)[:40]
    for a in gg:
     for z in cc:d['cross:'+a+'|'+z]=1.
    for a in gc:
     for z in ac:d['concept:'+a+'|'+z]=1.
    for z in cc:d['context:'+z]=.3
   for i,terms in relevant:
    if i in used:
     for a in sorted(g)[:24]:
      for z in sorted(terms)[:24]:d['operand:'+a+'|'+z]=1.
   dictionaries.append(d)
 dense=sparse.csr_matrix(np.asarray(numeric,dtype=np.float64).reshape(-1,37))
 if view=='numeric':return dense
 hashed=FeatureHasher(n_features=SIZE,input_type='dict',alternate_sign=True).transform(dictionaries)
 norms=np.sqrt(np.asarray(hashed.multiply(hashed).sum(1)).ravel()).clip(1)
 hashed=hashed.multiply((1/norms)[:,None]).tocsr()
 return sparse.hstack([dense,hashed],format='csr')
