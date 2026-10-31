import json
import numpy as np
from programs import question_numbers,induce,evaluate
from quantity_relations import reference_targets,candidate_graph,candidate_features
from fractions import Fraction
v,t=question_numbers('A .6 mile gap, two dogs and half a minute.');assert v==[Fraction(3,5),Fraction(2),Fraction(1,2)],(v,t)
row={'id':'fresh','question':'Two workers get 20 dollars for 8 hours.'}
ref={**row,'answer':'20*2=<<20*2=40>>40\n40*8=<<40*8=320>>320\n#### 320'}
node=induce(row['question'],ref['answer']);assert evaluate(node,question_numbers(row['question'])[0])==320
rr,pp=reference_targets([row],[ref]);assert rr[0].tolist()==[1,1,1]
good=candidate_graph(row,'20*2=40\n40*8=320\n#### 320');bad=candidate_graph(row,'20*8=160\n#### 160');assert good['available'] and bad['available'];assert good['roles'].tolist()==[1,1,1] and bad['roles'].tolist()==[0,1,1],(good,bad)
r={'id':'pct','question':'An item costs 80 dollars and its price rises 25%.','candidates':[]};ref={**r,'answer':'80*.25=<<80*.25=20>>20\n80+20=<<80+20=100>>100\n#### 100'}
node=induce(r['question'],ref['answer']);assert node is not None
roles,pr=reference_targets([r],[ref]);assert roles[0].tolist()==[1,1],(node,roles)
g=candidate_graph(r,'80*0.25=20\n80+20=100\n#### 100');assert g['available'] and g['roles'].tolist()==[1,1],g
print(json.dumps({'status':'passed','checks':['leading decimals and word quantities','reference quantity sensitivity','missing-factor dependency detection','percentage literal/decimal alias binding']}))
row={'id':'quantity-free','question':'Which solution fits?','candidates':[{'text':f'2+3=5\n#### {v}','index':i} for i,v in enumerate([5,6,7,8])]}
x,_=candidate_features([row],[np.empty((0,3))],np.zeros((1,9)));assert not x.any()
