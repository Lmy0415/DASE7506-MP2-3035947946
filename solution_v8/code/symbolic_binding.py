"""Restricted affine equation parsing, unit annotations and answer-target binding.
All evaluation uses a whitelist AST interpreter; candidate Python is never run.
"""
import ast,re
from fractions import Fraction
from collections import Counter
import numpy as np
from common import parse_answer
from programs import question_numbers

UNIT_WORDS=r'(?:square\s+|cubic\s+)?(?:ft|feet|foot|inches|inch|meters?|metres?|miles?|km|cm|ounces?|oz|liters?|litres?|gallons?|dollars?|euros?|cents?|pesetas?|minutes?|hours?|days?|weeks?|months?|years?|seconds?|balls?|sweaters?|balloons?|bottles?|bills?|pencils?|erasers?|apples?|students?|people|persons?|children|cookies?|snails?|slices?|pieces?|plates?|jars?|shares?|cars?|truckloads?|uses?|times|eggs?|pounds?|lbs?|kilograms?|grams?|g)' 
STOP={'the','a','an','of','to','in','for','and','is','are','be','how','many','much','does','do','will','what','that','this','his','her','their','it','he','she','they','can','with','at','from','as','by','on','each','per','total','number','amount','let','we','need','find','price','cost','money','have','has'}

def words(s):
 out=[]
 for w in re.findall(r'[a-z]+',s.lower()):
  if len(w)>3 and w.endswith('s') and not w.endswith('ss'):w=w[:-1]
  if w not in STOP and len(w)>1:out.append(w)
 return out

def goal_text(question):
 # The last interrogative clause is more reliable than splitting decimal dots.
 starts=list(re.finditer(r'\b(?:how\s+(?:many|much|long|old|far)|what|calculate|find)\b',question,re.I))
 return question[starts[-1].start():] if starts else question[-240:]

def concept(text):
 s=text.lower();out=set()
 patterns={'profit':r'\bprofit|\bnet\b|\bgain\b','revenue':r'revenue|income|earning|money from selling|money from sales|selling sweater|sales revenue','cost':r'\bcost|\bspen[dt]|\bexpense|\bpai[dy]|\bpay\b','change':r'\bchange\b','remaining':r'\bleft\b|remaining|remain\b','difference':r'difference|more.*than|less.*than','per':r'\beach\b|\bper\b|one .*cost|a single','total':r'\btotal\b|in all|altogether|combined','time':r'how long|minutes?|hours?|weeks?|days?|seconds?','age':r'\bage\b|how old|years old'}
 for k,p in patterns.items():
  if re.search(p,s):out.add(k)
 return out

def normalize(expr,labels):
 s=expr.replace('\\$','').replace(',','').replace('**','').replace('−','-').replace('×','*').replace('÷','/').replace('$','').replace('\\left','').replace('\\right','')
 # Unit annotations disappear arithmetically; named expressions remain symbols.
 def annotation(m):
  label=m.group(1).strip()
  if not label or re.match(r'^'+UNIT_WORDS+r'\b',label,re.I) or re.fullmatch(r'[\\/$\s^\d-]*'+UNIT_WORDS+r'(?:\s*(?:/|per)\s*'+UNIT_WORDS+r')*[\s^\d]*',label,re.I):return ' UNITANNOT '
  key='v_'+re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_')
  labels[key]=label;return ' '+key+' '+(m.group(2) or '')
 s=re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}(\s*\^\s*(?:\{[23]\}|[23]))?',annotation,s)
 s=re.sub(r'/\s*UNITANNOT',' ',s).replace('UNITANNOT',' ')
 for _ in range(7):
  old=s;s=re.sub(r'\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}',r'((\1)/(\2))',s);s=re.sub(r'\\boxed\s*\{([^{}]*)\}',r'\1',s)
  if old==s:break
 for op in ['\\times','\\cdot','\\div']:s=s.replace(op,'/' if op=='\\div' else '*')
 s=re.sub(r'\\[,;! ]',' ',s).replace('\\(',' ').replace('\\)',' ').replace('\\[',' ').replace('\\]',' ')
 s=s.replace('\\Rightarrow',';').replace('\\rightarrow',';').replace('\\approx','≈')
 s=re.sub(r'(?<=\d)\s*'+UNIT_WORDS+r'(?:\s*\^\s*[23])?(?:\s*(?:/|per)\s*'+UNIT_WORDS+r')*',' ',s,flags=re.I)
 s=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',s)
 s=s.replace('{','(').replace('}',')')
 # Conventional implicit multiplication, including 2x, 5(x+1), and x(2).
 s=re.sub(r'(?<=\d)\s*(?=[a-zA-Z_(])','*',s);s=re.sub(r'(?<=[a-zA-Z_)])\s*(?=\d|\()','*',s)
 return s.strip()

def affine(expr,labels):
 if len(expr)>320:raise ValueError('long expression')
 if re.search(r'\d\s*\*\*\s*[-+\d]',expr):raise ValueError('unsupported power')
 expr=normalize(expr,labels)
 tree=ast.parse(expr,mode='eval')
 def walk(n,depth=0):
  if depth>16:raise ValueError('depth')
  if isinstance(n,ast.Expression):return walk(n.body,depth+1)
  if isinstance(n,ast.Constant) and type(n.value) in (int,float):return Fraction(str(n.value)),{}
  if isinstance(n,ast.Name) and re.fullmatch(r'[a-zA-Z]|v_[a-z0-9_]{1,90}',n.id):return Fraction(0),{n.id:Fraction(1)}
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
   c,d=walk(n.operand,depth+1);sign=-1 if isinstance(n.op,ast.USub) else 1;return c*sign,{k:v*sign for k,v in d.items()}
  if isinstance(n,ast.BinOp):
   a,ad=walk(n.left,depth+1);b,bd=walk(n.right,depth+1)
   if isinstance(n.op,(ast.Add,ast.Sub)):
    sign=-1 if isinstance(n.op,ast.Sub) else 1;dd=dict(ad)
    for k,v in bd.items():dd[k]=dd.get(k,0)+v*sign
    return a+b*sign,{k:v for k,v in dd.items() if v}
   if isinstance(n.op,ast.Mult):
    if ad and bd:raise ValueError('nonlinear')
    return a*b,{k:v*(b if ad else a) for k,v in (ad or bd).items()}
   if isinstance(n.op,ast.Div):
    if bd or not b:raise ValueError('nonlinear/zero denominator')
    return a/b,{k:v/b for k,v in ad.items()}
  raise ValueError('unsupported syntax')
 c,d=walk(tree)
 if abs(c)>10**15 or any(abs(v)>10**12 for v in d.values()):raise ValueError('magnitude')
 return c,d,expr

def solve(equations):
 names=sorted({k for c,d in equations for k in d})
 if not names or len(names)>12:return {},False
 matrix=[[d.get(k,Fraction(0)) for k in names]+[-c] for c,d in equations];pivot=[];r=0
 for col in range(len(names)):
  idx=next((i for i in range(r,len(matrix)) if matrix[i][col]),None)
  if idx is None:continue
  matrix[r],matrix[idx]=matrix[idx],matrix[r];scale=matrix[r][col];matrix[r]=[x/scale for x in matrix[r]]
  for i in range(len(matrix)):
   if i!=r and matrix[i][col]:
    factor=matrix[i][col];matrix[i]=[x-factor*y for x,y in zip(matrix[i],matrix[r])]
  pivot.append((r,col));r+=1
  if r==len(matrix):break
 if any(not any(row[:-1]) and row[-1] for row in matrix):return {},True
 result={}
 for r,col in pivot:
  if all(not matrix[r][k] for k in range(len(names)) if k!=col):result[names[col]]=matrix[r][-1]
 return result,False

def spans(text):
 # Math blocks plus ordinary lines cover both LaTeX and prose equations.
 blocks=[]
 for m in re.finditer(r'\$\$(.*?)\$\$|(?<![\\$])\$(?!\$)(.*?)\$(?!\$)|\\\[(.*?)\\\]|\\\((.*?)\\\)',text,re.S):
  blocks.append((m.start(),next(g for g in m.groups() if g is not None)))
 at=0
 for line in text.splitlines(True):
  if '=' in line and len(line)<650:blocks.append((at,re.sub(r'^\s*[-*]\s+','',line)))
  at+=len(line)
 seen=set();out=[]
 for pos,b in sorted(blocks):
  key=(pos,b.strip())
  if key not in seen:seen.add(key);out.append((pos,b))
 return out

def extract(text):
 labels={};equations=[];records=[];attempted=0;parsed=0
 for m in re.finditer(r'\blet\s*\$?\s*([a-z])\s*\$?\s*(?:be|represent|denote|=)\s*([^\n.]{3,130})',text,re.I):labels[m.group(1)]=m.group(2)
 for pos,block in spans(text):
  if '=' not in block:continue
  block=re.sub(r'\n\s*\n',';',block)
  for line in re.split(r'[;\n]',block):
   if '=' not in line:continue
   chunks=line.split('=');parts=[]
   for chunk in chunks:
    attempted+=1
    try:parts.append(affine(chunk,labels));parsed+=1
    except (ValueError,SyntaxError,ZeroDivisionError,RecursionError,OverflowError):parts.append(None)
   for left,right in zip(parts,parts[1:]):
    if left is None or right is None:continue
    a,ad,aa=left;b,bd,bb=right;dd=dict(ad)
    for k,v in bd.items():dd[k]=dd.get(k,0)-v
    dd={k:v for k,v in dd.items() if v};cc=a-b
    if dd:equations.append((cc,dd))
    if not ad and not bd and a==b:
     prefix=text[max(0,pos-180):pos].strip()
     # Retain the closest prose label; older steps often name another object.
     prefix=re.split(r'\n\s*\n',prefix)[-1]
     prefix=prefix.split('\n')[-1]
     context=prefix[-120:]+' '+line[:100]
     records.append({'value':a,'expr':aa,'context':context,'variables':[]})
   # A named left side with a numeric branch in a chain binds to that branch.
   if parts and parts[0] is not None:
    c,d,_=parts[0]
    if c==0 and len(d)==1 and next(iter(d.values()))==1:
     name=next(iter(d));numeric=next((p[0] for p in reversed(parts[1:]) if p is not None and not p[1]),None)
     if numeric is not None:records.append({'value':numeric,'expr':line,'context':labels.get(name,name),'variables':[name]})
  if len(equations)>40:break
 resolved,inconsistent=solve(equations[:40])
 for name,v in resolved.items():records.append({'value':v,'expr':name,'context':labels.get(name,name),'variables':[name]})
 return {'labels':labels,'resolved':resolved,'inconsistent':inconsistent,'records':records,'parsed':parsed,'attempted':attempted}

def answer_binding(row,candidate,parsed=None):
 text=candidate['text'];p=extract(text) if parsed is None else parsed;a=parse_answer(text);av=Fraction(a) if a is not None else None;goal=goal_text(row['question']);target=set(words(goal));matches=[r for r in p['records'] if av is not None and r['value']==av]
 # Prefer a label sharing the actual requested object over unrelated equal numbers.
 ranked=sorted(matches,key=lambda r:(len(target&set(words(r['context']))),bool(r['variables'])),reverse=True)
 context=' '.join(r['context'] for r in ranked[:2])
 if not context:
  # Missing bindings are unknown; local answer language remains a weak text signal.
  context='UNBOUND '
 final=text[-500:];final=re.sub(r'\d+(?:[.,/]\d+)*',' NUMBER ',final)
 aliases=' '.join(k+' '+v for k,v in p['labels'].items() if any(k in r['variables'] for r in ranked[:2]))
 return {'answer':a,'goal':goal,'context':context,'final_text':final,'aliases':aliases,'parsed':p['parsed'],'attempted':p['attempted'],'bound':bool(matches),'inconsistent':p['inconsistent'],'resolved':len(p['resolved']),'records':p['records'],'goal_terms':target,'context_terms':set(words(context)),'goal_concepts':concept(goal),'answer_concepts':concept(context)}
