"""Learn a compact family of two-quantity relationships from train references."""
import re
from fractions import Fraction
import numpy as np
from programs import question_numbers
from common import parse_answer

def template_values(question):
 values,text=question_numbers(question)
 if len(values)!=2 or values[0]==values[1]:return {}
 a,b=values;answers={'sum':a+b,'subtract_ab':a-b,'subtract_ba':b-a,'product':a*b}
 if b:answers['divide_ab']=a/b
 if a:answers['divide_ba']=b/a
 rates=[int(m.group(1)) for m in re.finditer(r'NUM(\d+)\s*(?:%|percent)',text,re.I)]
 if len(rates)==1:
  rate=values[rates[0]]/100;basis=values[1-rates[0]]
  answers.update({'fraction_of':basis*rate,'percent_up':basis*(1+rate),'percent_down':basis*(1-rate)})
  if rate:answers['divide_rate']=basis/rate
  if 1+rate:answers['reverse_up']=basis/(1+rate)
  if 1-rate:answers['reverse_down']=basis/(1-rate)
 return {k:str(v) for k,v in answers.items() if abs(v)<10**12 and v.denominator<=10**9}

def text_view(q):
 _,q=question_numbers(q);goal=re.split(r'\b(?:how many|how much|what|calculate|find)\b',q)[-1];tokens=re.findall(r'NUM\d+|[a-z]+|%',q);tags=[]
 for i,t in enumerate(tokens):
  if re.fullmatch(r'NUM\d+',t):
   prefix=t.lower();tags.extend(prefix+'_'+w for w in tokens[max(0,i-6):i]+tokens[i+1:i+7] if re.fullmatch('[a-z]+',w))
 return q+' GOAL '+goal+' GOAL '+goal+' '+' '.join(tags)

def targets(row,ref):
 ans=parse_answer(ref['answer']);proposals=template_values(row['question']);return [k for k,v in proposals.items() if v==ans]

def score(bank,vec,model):
 if not bank:return np.zeros((0,4))
 p=model.predict_proba(vec.transform([text_view(r['question']) for r in bank]));classes=model.classes_;out=[]
 for row,pr in zip(bank,p):
  proposal=template_values(row['question']);mass={}
  for c,w in zip(classes,pr):
   if c in proposal:mass[proposal[c]]=mass.get(proposal[c],0)+w
  out.append([mass.get(parse_answer(c['text']),0.) for c in row['candidates']])
 return np.asarray(out)
