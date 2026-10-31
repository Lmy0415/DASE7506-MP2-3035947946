"""Learn a compact family of two-quantity relationships from train references."""
import re
from fractions import Fraction
import numpy as np
from programs import question_numbers,induce,evaluate
from common import parse_answer

def template_values(question,override=None):
 values,text=question_numbers(question)
 if len(values) not in (2,3) or len(values)!=len(set(values)):return {}
 if override is not None:values=override
 answers={}
 def put(k,fn):
  try:
   v=fn()
   if abs(v)<10**12 and v.denominator<=10**9:answers[k]=str(v)
  except (ZeroDivisionError,ValueError,OverflowError):pass
 a,b=values[:2]
 put('sum2',lambda:a+b);put('sub_ab',lambda:a-b);put('sub_ba',lambda:b-a);put('mul2',lambda:a*b);put('div_ab',lambda:a/b);put('div_ba',lambda:b/a)
 if len(values)==2:
  for i,j in [(0,1),(1,0)]:
   u,v=values[i],values[j]
   for tag,fn in [('sumprod',lambda:u*(1+v)),('diffprod',lambda:u*(1-v)),('reverseplus',lambda:u/(1+v)),('reverseminus',lambda:u/(1-v)),('halfsum',lambda:(u+v)/2),('plus_half',lambda:u+v/2),('minus_half',lambda:u-v/2)]:put(tag+str(i),fn)
 else:
  a,b,c=values;put('sum3',lambda:a+b+c);put('mul3',lambda:a*b*c)
  for i in range(3):
   rest=[k for k in range(3) if k!=i];u,v,w=values[i],values[rest[0]],values[rest[1]]
   for tag,fn in [('summinus',lambda:v+w-u),('twosub',lambda:u-v-w),('prodplus',lambda:v*w+u),('prodminus',lambda:v*w-u),('minusprod',lambda:u-v*w),('sumprod',lambda:(v+w)*u),('sumdiv',lambda:(v+w)/u),('proddiv',lambda:v*w/u),('plusdiv',lambda:u+v/w),('minusdiv',lambda:u-v/w)]:put(tag+str(i),fn)
 rates=[int(m.group(1)) for m in re.finditer(r'NUM(\d+)\s*(?:%|percent)',text,re.I)]
 if len(rates)==1:
  pi=rates[0];rate=values[pi]/100;other=[i for i in range(len(values)) if i!=pi]
  for i in other:
   basis=values[i]
   for tag,fn in [('percent_of',lambda:basis*rate),('percent_up',lambda:basis*(1+rate)),('percent_down',lambda:basis*(1-rate)),('percent_divide',lambda:basis/rate),('reverse_up',lambda:basis/(1+rate)),('reverse_down',lambda:basis/(1-rate))]:put(tag+str(i),fn)
  if len(other)==2:
   x,z=values[other[0]],values[other[1]]
   for tag,fn in [('sum_percent',lambda:(x+z)*rate),('sum_up',lambda:(x+z)*(1+rate)),('sum_down',lambda:(x+z)*(1-rate)),('a_plus_bpercent',lambda:x+z*rate),('b_plus_apercent',lambda:z+x*rate),('a_minus_bpercent',lambda:x-z*rate),('b_minus_apercent',lambda:z-x*rate),('a_plus_bup',lambda:x+z*(1+rate)),('b_plus_aup',lambda:z+x*(1+rate))]:put(tag,fn)
 return {str(len(values))+'_'+k:v for k,v in answers.items()}

def text_view(q):
 _,q=question_numbers(q);goal=re.split(r'\b(?:how many|how much|what|calculate|find)\b',q)[-1];tokens=re.findall(r'NUM\d+|[a-z]+|%',q);tags=[]
 for i,t in enumerate(tokens):
  if re.fullmatch(r'NUM\d+',t):
   prefix=t.lower();tags.extend(prefix+'_'+w for w in tokens[max(0,i-6):i]+tokens[i+1:i+7] if re.fullmatch('[a-z]+',w))
 return q+' GOAL '+goal+' GOAL '+goal+' '+' '.join(tags)

def targets(row,ref):
 ans=parse_answer(ref['answer']);proposals=template_values(row['question']);matches=[k for k,v in proposals.items() if v==ans]
 if len(matches)>1:
  vv,_=question_numbers(row['question']);node=induce(row['question'],ref['answer'])
  if node is not None:
   stable=matches
   for deltas in [(3,5,7),(7,2,11)]:
    altered=[v+Fraction(deltas[i]) for i,v in enumerate(vv)];hyp=template_values(row['question'],altered)
    try:refval=str(evaluate(node,altered))
    except (ValueError,IndexError,ZeroDivisionError,RecursionError):break
    stable=[k for k in stable if hyp.get(k)==refval]
   if stable:matches=stable
 return matches

def score(bank,vec,model):
 if not bank:return np.zeros((0,4))
 p=model.predict_proba(vec.transform([text_view(r['question']) for r in bank]));classes=model.classes_;out=[]
 for row,pr in zip(bank,p):
  proposal=template_values(row['question']);mass={}
  for c,w in zip(classes,pr):
   if c in proposal:mass[proposal[c]]=mass.get(proposal[c],0)+w
  out.append([mass.get(parse_answer(c['text']),0.) for c in row['candidates']])
 return np.asarray(out)
