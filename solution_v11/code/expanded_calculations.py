"""Soft evidence from fully numeric equalities in LaTeX and prose.
Unsupported symbols, unit conversions, approximate statements and ambiguous
syntax are skipped.  A failure to parse is unknown rather than incorrect.
"""
import re,math
import numpy as np
from symbolic_binding import spans,affine,UNIT_WORDS
from common import parse_answer
from fractions import Fraction

def equation_features(text):
 results=[];seen=set();ans=parse_answer(text)
 for pos,block in spans(text):
  if '=' not in block or re.search(r'\\approx|≈|approximately|roughly|round(?:ed|ing)?\b',block,re.I):continue
  for line in re.split(r'[;\n]',block):
   if '=' not in line:continue
   # Avoid comparing unlike unit values and hide proof-free unit conversions.
   unit_text=re.sub(r'\\(?:times|cdot|div)\b',' ',line)
   units=re.findall(r'\b'+UNIT_WORDS+r'\b',unit_text,re.I)
   normalized_units={u.lower().rstrip('s') for u in units}
   if len(normalized_units)>1:continue
   pieces=line.split('=');parsed=[]
   for p in pieces:
    try:
     value,terms,norm=affine(p,{})
     parsed.append(None if terms else (value,norm))
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError,RecursionError):parsed.append(None)
   for left,right in zip(parsed,parsed[1:]):
    if left is None or right is None:continue
    x,xx=left;y,yy=right;key=(xx,yy)
    if key in seen:continue
    seen.add(key)
    rel=float(abs(x-y)/max(abs(x),abs(y),Fraction(1)))
    bad=rel>.015 and abs(x-y)>Fraction(1,200)
    is_answer=ans is not None and (x==Fraction(ans) or y==Fraction(ans))
    results.append((bad,min(rel,10),is_answer))
 total=len(results);errors=[r for r in results if r[0]];final=[r for r in results if r[2]];badfinal=[r for r in final if r[0]]
 return [float(total>0),np.log1p(total),np.log1p(len(errors)),len(errors)/max(1,total),max((r[1] for r in errors),default=0),float(bool(errors)),np.log1p(len(final)),float(bool(final)),np.log1p(len(badfinal)),float(bool(badfinal)),float(bool(final) and not badfinal),float(bool(errors) and len(errors)==total)]

def features(bank):
 if not bank:return np.empty((0,4,24))
 x=np.array([[equation_features(c['text']) for c in r['candidates']] for r in bank],dtype=np.float64)
 return np.concatenate([x,x-x.mean(1,keepdims=True)],-1)
