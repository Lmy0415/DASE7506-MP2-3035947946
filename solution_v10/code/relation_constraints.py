"""Conservative question-side conservation and cumulative-stage constraints.
Only whitelisted patterns produce an answer hypothesis. Unrecognized is unknown.
"""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from programs import question_numbers
FAMILIES=['time_stages','hatching_stock','growth_by','stock_events']

def stem(w):
 return w[:-1] if len(w)>3 and w.endswith('s') else w

def hypotheses(question):
 values,q=question_numbers(question);q=re.sub(r'\s+',' ',q).strip();out=[]
 goal=re.split(r'\b(?:how many|how much|what is|calculate|find)\b',q)[-1]
 total=bool(re.search(r'\b(?:total|altogether|combined|in all|now|have|has)\b',goal))
 def add(f,value,components):
  if value>=0 and abs(value)<10**12:out.append({'family':f,'expected':str(value),'components':[str(x) for x in components]})
 # Time stages share the same unit. Additional explicit durations or rates abstain.
 if total and re.search(r'\b(?:time|minutes?|hours?)\b',goal) and 2<=len(values)<=3:
  base=re.search(r'NUM(\d+) (minutes?|hours?|seconds?)\b',q)
  factors=list(re.finditer(r'NUM(\d+) times (?:that long|that amount of time|as much(?: time)?|the amount of time|as long)',q))
  if base and len(factors)==len(values)-1 and not re.search(r'\b(?:per|every|daily|weekly|each day|a week|a day|less|more|longer|shorter)\b',q):
   indices=[int(base.group(1))]+[int(m.group(1)) for m in factors]
   if len(set(indices))==len(values):
    t=values[indices[0]];parts=[t];parts.append(t*values[indices[1]])
    if len(indices)==3:
     source=sum(parts) if re.search(r'combined (?:time|amount)|time.*combined',q[factors[1].start():]) else parts[-1]
     parts.append(source*values[indices[2]])
    add('time_stages',sum(parts),parts)
 # Existing adults survive while offspring are added. Birth-only questions abstain.
 if total and 'egg' in q and re.search(r'\bhatch',q) and not re.search(r'\b(?:die|died|dead|sold|sell|gave|give|lost|remain unhatched)\b',q):
  initial=re.search(r'(?:had|has|have|owns?|there (?:are|were)) (?:a total of )?NUM(\d+) (?:female |adult )?([a-z]+)',q)
  eggs=re.search(r'(?:laid|lays?|lay|produce[sd]?) (?:an average of )?NUM(\d+) eggs? (?:each|per)',q)
  hatch=re.search(r'NUM(\d+)\s*(%)? of (?:the |these )?eggs (?:hatch|hatched)',q)
  if initial and eggs and hatch and stem(initial.group(2)) in {stem(w) for w in re.findall('[a-z]+',goal)}:
   ids=[int(initial.group(1)),int(eggs.group(1)),int(hatch.group(1))]
   if len(set(ids))==3:
    n,k,p=(values[i] for i in ids);p=p/100 if hatch.group(2) else p
    if 0<=p<=1:add('hatching_stock',n+n*k*p,[n,n*k*p])
 # 'Increased by' describes the increment, not the resulting total.
 growth=re.search(r'increased? by NUM(\d+) (times|%)',q)
 initial=re.search(r'(?:were|was|had|has|have) NUM(\d+) [a-z]+ initially|initially[^.!?]*?NUM(\d+)',q)
 if total and growth and initial and len(values)==2:
  r=values[int(growth.group(1))];n=values[int(next(g for g in initial.groups() if g is not None))]
  r=r/100 if growth.group(2)=='%' else r
  add('growth_by',n*(1+r),[n,n*r])
 # Same-owner stock events with the same explicit object; no prices/rates/ratios.
 if (total or re.search(r'\bleft\b',goal)) and 2<=len(values)<=5 and not re.search(r'\b(?:each|per|times|percent|half|double|twice|cost|price|dollar|every|left to|than)\b|[%$]',q):
  clauses=re.split(r'[.!?;]|\band (?=(?:he |she |they )?(?:bought|buys|received|receives|gave|gives|lost|loses|ate|eats|sold|sells)\b)',q);ledger=[];owner=None;obj=None;valid=True
  for clause in clauses:
   if not re.search('NUM\\d+',clause):continue
   nums=list(re.finditer(r'NUM(\d+) ([a-z]+)',clause))
   if len(nums)!=1:valid=False;break
   m=nums[0];noun=stem(m.group(2));before=clause[:m.start()].strip()
   event=re.search(r'\b(has|had|have|owns|owned|buys|bought|gets|got|receives|received|finds|found|adds|added|loses|lost|sells|sold|gives|gave|eats|ate|uses|used)\b',before)
   if event is None:valid=False;break
   subject=before[:event.start()].strip();verb=event.group(1)
   subject=re.sub(r'^(?:then |after that |next )','',subject)
   if not ledger:
    if verb not in ['has','had','have','owns','owned'] or len(subject.split())>3:valid=False;break
    owner=subject;obj=noun;sign=1
   else:
    if subject not in [owner,'he','she','they',''] or noun!=obj:valid=False;break
    sign=-1 if verb in ['loses','lost','sells','sold','gives','gave','eats','ate','uses','used'] else 1
   ledger.append(sign*values[int(m.group(1))])
  if valid and len(ledger)==len(values) and obj in {stem(w) for w in re.findall('[a-z]+',goal)}:add('stock_events',sum(ledger),ledger)
 return out

def features(bank,view='joint'):
 from scipy import sparse
 x=[]
 for row in bank:
  hs=hypotheses(row['question'])
  for c in row['candidates']:
   a=parse_answer(c['text']);v=[]
   for family in FAMILIES:
    match=[h for h in hs if h['family']==family]
    if not match or a is None:v.extend([0.,0.,0.]);continue
    h=match[0];agree=a==h['expected'];omitted=a in h['components'] and not agree
    v.extend([float(agree),float(not agree),float(omitted)])
   x.append(v)
 z=np.asarray(x).reshape(-1,12)
 if view=='shared':z=z.reshape(-1,4,3).sum(1)
 elif view=='joint':z=np.c_[z,z.reshape(-1,4,3).sum(1)]
 return sparse.csr_matrix(z)
