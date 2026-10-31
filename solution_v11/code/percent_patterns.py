"""Conservative numeric hypotheses for common percentage relations."""
import re
from programs import question_numbers

def hypotheses(question):
 vals,q=question_numbers(question);q=re.sub(r'\s+',' ',q).lower();goal=re.split(r'\b(?:how many|how much|what|calculate|find)\b',q)[-1];res=[]
 def add(f,v):
  if v>=0 and abs(v)<10**12:res.append((f,str(v)))
 # A single base dollar price and single rate; ignore ambiguous extra quantities,
 # sales tax, staged discounts, unit conversions and asking for the change itself.
 if len(vals)==2 and not re.search(r'\b(?:tax|tip|shipping|each day|each week|per month|per year|more than once|additional|combined|total of both)\b',q):
  per=list(re.finditer(r'num(\d+)\s*(?:%|percent)',q));money=list(re.finditer(r'\$\s*num(\d+)|num(\d+)\s*dollars?',q))
  if len(per)==1 and len(money)==1:
   pi=int(per[0].group(1));mi=int(next(g for g in money[0].groups() if g is not None));rate=vals[pi]/100;amount=vals[mi]
   if pi!=mi and 0<=rate<=2 and re.search(r'\b(?:pay|paid|cost|price|charge|spend|spent|would|will)\b',goal) and not re.search(r'\b(?:save|saved|savings|discount amount|increase|difference|more|less)\b',goal):
    discount=re.search(r'\b(?:discount|off|sale|reduced by|decreased by|decrease by|less expensive|less than)\b',q)
    increase=re.search(r'\b(?:more expensive|increased by|increase by|markup|mark-up|more than|price increase)\b',q)
    if discount and not increase:add('percent_decrease',amount*(1-rate))
    if increase and not discount:add('percent_increase',amount*(1+rate))
 # r% of an explicitly provided count. Final question must ask for the same
 # population subset; don't guess whether the remainder or sum is desired.
 if len(vals)==2 and not re.search(r'\b(?:left|remain|remaining|other|rest|more|less|increase|decrease|total|together|off|discount|save|difference|another|twice|double)\b',goal):
  m=re.search(r'num(\d+)\s*(?:%|percent)\s+of\s+(?:the\s+)?([a-z]+)',q)
  if m and re.search(r'\b(?:how many|what number)\b',question.lower()):
   pi=int(m.group(1));others=[i for i in range(len(vals)) if i!=pi];rate=vals[pi]/100
   # Numeric total immediately adjacent to the same noun.
   noun=m.group(2).rstrip('s');v=re.search(r'num(\d+)\s+([a-z]+)',q)
   if v and int(v.group(1))==others[0] and v.group(2).rstrip('s')==noun and noun in goal and 0<=rate<=1:add('percent_subset',vals[others[0]]*rate)
 return res
