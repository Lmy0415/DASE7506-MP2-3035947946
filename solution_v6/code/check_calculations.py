"""Regression checks for arithmetic safety, propagation, ambiguity and evidence gating."""
import json
from fractions import Fraction
import numpy as np
from calculation_evidence import analyze,equations,parse,evidence_features

r=analyze('Four groups contain 8 items each; remove 5 items.','4*8=31\n31-5=26\n#### 26')
assert r['repaired_answer']=='27',r
r=analyze('Use 1/2 of a batch.','1+2=0.5\n0.5*8=4\n#### 4');assert r['repaired_answer'] is None,r
r=analyze('There are six kinds of items.','2+5=6\n6*8=48\n#### 48');assert r['repaired_answer'] is None,r
r=analyze('Find the quantity.','0.3 / 9 = 0.0333\n0.0333*90=3\n#### 3');assert not r['errors'],r
r=analyze('Five, seven and ten minutes per week; two weeks.','- 5 minutes + 7 minutes + 10 minutes = 21 minutes\n21*2=42\n#### 42');assert r['repaired_answer']=='44',r
r=analyze('Find the difference.','2x=10\nx+24=36\n25% of $40 = 0.25*40=10\n#### 10');assert not r['errors'],r
for expr in ['__import__("os").system("true")','2**100','x+1','1/0']:
 try:parse(expr)
 except (ValueError,SyntaxError,ZeroDivisionError):pass
 else:raise AssertionError('Unsafe/unsupported expression accepted')
cs=[{'index':i,'text':f'2+3=5\n#### {a}'} for i,a in enumerate([5,6,7,8])]
b=[{'id':'new-id','question':'There are 2 and 3 objects.','candidates':cs}];x,_=evidence_features(b);assert not x.any(),x
cs[1]['text']='2+3=6\n#### 6';x,_=evidence_features(b);assert x.any();perm=[2,0,3,1];moved=[{**b[0],'candidates':[cs[i] for i in perm]}];xp,_=evidence_features(moved);np.testing.assert_allclose(xp,x[:,perm],atol=1e-7)
cs=[{'index':i,'text':t} for i,t in enumerate(['4*7=27\n#### 27','2+3=5\n#### 5','2+5=7\n#### 7','3+5=8\n#### 8'])]
xx,_=evidence_features([{'id':'another-id','question':'A task has 4 and 7 units.','candidates':cs}],local_gate=True);assert not xx[0,1:,:14].any(),xx
print(json.dumps({'status':'passed','checks':['safe restricted arithmetic','multi-step error propagation','fraction and English quantity ambiguity guards','rounding interval propagation','Markdown bullets and units','symbolic expressions not interpreted as numbers','no-evidence gate','candidate permutation equivariance']}))
