"""Check transfer of reference programs to unseen numbers and safe abstention."""
import numpy as np
from reference_patterns import build,features
from programs import evaluate,question_numbers
from common import parse_answer

def row(identifier,question,answers):
 return {'id':identifier,'question':question,'candidates':[{'index':i,'text':'#### '+str(a)} for i,a in enumerate(answers)]}
b=[row('a','Rae has 11 books and buys 7 more books. How many books now?',[18,11,7,4]),row('b','Lee has 17 books and buys 9 more books. How many books now?',[26,17,9,8])]
refs=[{'id':r['id'],'question':r['question'],'answer':f'<<{a}+{b}={a+b}>>\n#### {a+b}'} for r,(a,b) in zip(b,[(11,7),(17,9)])]
library=build(b,refs);assert len(library['programs'])==2
q=row('unseen','Kim has 23 books and buys 4 more books. How many books now?',[27,23,4,31]);x=features([q],library)
assert x[0,0,0]>0 and np.all(x[0,1:,0]==0),x
for program in library['programs']:assert evaluate(program,question_numbers(q['question'])[0])==27
perm=[2,0,3,1];m=dict(q,candidates=[q['candidates'][i] for i in perm]);np.testing.assert_allclose(features([m],library),x[:,perm])
unknown=row('unknown','Kim has 23 books and buys 4 more books. How many books now?',[99,98,97,96]);assert np.all(features([unknown],library)==0)
assert features([],library).shape==(0,4,12)
bad=row('ambiguous','Rae has 11 books and buys 11 books. How many books now?',[22,11,0,1]);badref={'id':'ambiguous','question':bad['question'],'answer':'<<11+11=22>>\n#### 22'}
filtered=build(b+[bad],refs+[badref]);assert len(filtered['programs'])==2
assert not any(k in library for k in ['ids','answers','references','questions'])
print('passed: symbolic transfer to new quantities; candidate permutation; no matching answer abstains; empty input; ambiguous source quantities filtered; no raw reference/answer export')
