"""Check operation directions and repeated-quantity cumulative programs."""
import numpy as np
from local_relations import distribution,pair_texts,features
q0=('q',0);q1=('q',1)
assert distribution(('-',q0,q1),2).argmax()==3
assert distribution(('-',q1,q0),2).argmax()==4
assert distribution(('/',q0,q1),2).argmax()==5
assert distribution(('/',q1,q0),2).argmax()==6
p=('+',q0,('*',q0,q1));d=distribution(p,2)[0];np.testing.assert_allclose(d,[0,.5,.5,0,0,0,0])
assert distribution(('+',q0,('c','1')),2).argmax()==0
assert len(pair_texts('A box has 5 red balls and 7 blue balls. There are 3 boxes. How many balls?'))==3
class Dummy:
 classes_=np.arange(7)
 def predict_proba(self,x):return np.ones((x.shape[0],7))/7
state={'model':Dummy(),'bins':{2:{'programs':[('+',q0,q1)],'matrix':distribution(('+',q0,q1),2).ravel()[None,:]}}}
r={'question':'A has 23 books and buys 4 more. How many books?', 'candidates':[{'text':'#### '+str(v)} for v in [27,23,4,31]]}
x=features([r],state);assert x[0,0,0]==1 and np.all(x[0,1:,0]==0)
perm=[2,0,3,1];m=dict(r,candidates=[r['candidates'][i] for i in perm]);np.testing.assert_allclose(features([m],state),x[:,perm])
assert features([],state).shape==(0,4,12)
print('passed: subtract/divide direction; cumulative repeated-slot relations; unused quantities; transfer to new numbers; candidate permutation; empty input')
