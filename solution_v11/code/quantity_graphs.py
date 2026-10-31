"""Question quantity roles and candidate numerical dependency graphs. Input-only."""
import ast,re,collections
from fractions import Fraction
import numpy as np
from common import parse_answer
from programs import question_numbers,induce,evaluate,slots
from calculation_evidence import equations,value

OPS=['+','-','*','/','identity']

def counts(node):
 out=np.zeros(4,dtype=np.float32)
 def walk(n):
  if n[0] in OPS[:4]:out[OPS.index(n[0])]+=1;walk(n[1]);walk(n[2])
 walk(node);return out

def role(node,values):
 base=evaluate(node,values);out=[]
 for i,v in enumerate(values):
  changed=list(values);changed[i]+=Fraction(1,100)
  try:d=evaluate(node,changed)-base;out.append(0 if d==0 else 1 if d>0 else 2)
  except (ValueError,ZeroDivisionError):out.append(-1)
 return np.array(out,dtype=np.int64)

def question_text(row):
 values,text=question_numbers(row['question']);goal=re.split(r'[.!?]',text)[-2:] if text.endswith('?') else re.split(r'[.!?]',text)[-1:]
 goal=' '.join(goal);generic=re.sub(r'NUM\d+','NUMBER',text)
 return values,text,goal,generic+' GOAL '+goal+' GOAL '+goal

def slot_texts(row):
 values,text,goal,full=question_text(row);words=re.findall(r'NUM\d+|[a-z]+|[%/]',text);out=[]
 for i,v in enumerate(values):
  at=words.index('NUM'+str(i));left=words[max(0,at-9):at];right=words[at+1:at+10]
  generic=lambda zz:[re.sub(r'NUM\d+','NUMBER',t) for t in zz]
  goalwords=set(re.findall(r'[a-z]+',goal));local=set(left+right);overlap=sorted(goalwords&local-{'the','a','of','is','to','and','how','many','much','does'})
  # Anchored prefixes allow roles to differ between a quantity's left/right context.
  out.append(' '.join(['L_'+w for w in generic(left)]+['QUANTITY']+['R_'+w for w in generic(right)])+' GOAL '+re.sub(r'NUM\d+','NUMBER',goal)+' OVERLAP '+' '.join(overlap)+f' POSITION_{min(i,6)} COUNT_{min(len(values),8)}')
 return out

def reference_targets(bank,refs):
 targets=[];meta=[]
 for row,ref in zip(bank,refs):
  assert row['id']==ref['id'] and row['question']==ref['question']
  vv,_=question_numbers(row['question']);node=induce(row['question'],ref['answer'])
  if node is None:targets.append(None);meta.append(None);continue
  try:
   rr=role(node,vv);cnt=counts(node);root=OPS.index(node[0]) if node[0] in OPS else 4
   # Identical-valued question slots cannot be distinguished by numeric binding.
   for i,v in enumerate(vv):
    if vv.count(v)>1:rr[i]=-1
   targets.append(rr);meta.append(np.r_[np.eye(5,dtype=np.float32)[root],np.log1p(cnt)])
  except (ValueError,ZeroDivisionError,IndexError,RecursionError):targets.append(None);meta.append(None)
 return targets,meta

def candidate_graph(row,text):
 vv,normalized_question=question_numbers(row['question']);initial={v:('q',i) for i,v in enumerate(vv) if vv.count(v)==1}
 percentage_slots={int(i) for i in re.findall(r'NUM(\d+)\s*%',normalized_question)}
 aliases={vv[i]/100:i for i in percentage_slots if vv.count(vv[i])==1 and vv[i]/100 not in initial}
 for v,i in aliases.items():initial[v]=('/',('q',i),('c','100'))
 known={};ambiguous=set();used=set();checked=0
 def build(n,depth=0):
  if depth>14:raise ValueError('depth')
  if isinstance(n,ast.Expression):return build(n.body,depth+1)
  if isinstance(n,ast.Constant):
   v=Fraction(str(n.value))
   if v in known:
    if v in initial and known[v]!=initial[v]:raise ValueError('ambiguous binding')
    return known[v]
   if v in initial:return initial[v]
   if vv.count(v)>1:raise ValueError('duplicate question quantity')
   return ('c',str(v))
  if isinstance(n,ast.UnaryOp):
   node=build(n.operand,depth+1);return ('*',('c','-1'),node) if isinstance(n.op,ast.USub) else node
  if isinstance(n,ast.BinOp):
   tag={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(n.op)]
   return (tag,build(n.left,depth+1),build(n.right,depth+1))
  raise ValueError('unsupported')
 eqs=equations(text)
 for eq in eqs:
  for n in ast.walk(eq['left']):
   if isinstance(n,ast.Constant) and type(n.value) in (int,float):
    v=Fraction(str(n.value));used.update(i for i,x in enumerate(vv) if x==v)
    if v in aliases:used.add(aliases[v])
  if not eq['valid']:continue
  try:
   node=build(eq['left']);rv=eq['rv'];checked+=1
   if rv in known and known[rv]!=node:
    # Algebraically equivalent derivations may differ; retain the newest valid one.
    pass
   known[rv]=node
  except (ValueError,KeyError,ZeroDivisionError,RecursionError):ambiguous.add(eq['rv'])
 answer=parse_answer(text);node=None
 if answer is not None:
  av=Fraction(answer)
  if av in known:node=known[av]
  elif av in initial:node=initial[av]
 result={'used':np.array([i in used for i in range(len(vv))]),'available':False,'ambiguous':len(ambiguous)>0,'checked':checked,'roles':np.full(len(vv),-1),'counts':np.zeros(4),'root':-1,'final_slots':set(),'program':None}
 if node is not None:
  try:
   if evaluate(node,vv)!=Fraction(answer):raise ValueError('inconsistent final graph')
   result.update(available=True,roles=role(node,vv),counts=counts(node),root=OPS.index(node[0]) if node[0] in OPS else 4,final_slots=slots(node),program=node)
  except (ValueError,ZeroDivisionError,IndexError,RecursionError):pass
 return result

def candidate_features(bank,role_probs,profiles,graphs=None):
 rows=[]
 if graphs is None:graphs=[[candidate_graph(r,c['text']) for c in r['candidates']] for r in bank]
 for row,pp,profile,gg in zip(bank,role_probs,profiles,graphs):
  vv,_=question_numbers(row['question']);unique=np.array([vv.count(v)==1 for v in vv]);need=1-pp[:,0] if len(vv) else np.zeros(0);features=[]
  for g in gg:
   if not vv:features.append([0.]*23);continue
   missing=(~g['used'])&unique;den=max(1,unique.sum());required=need[unique].sum()
   f=[g['available'],g['ambiguous'],np.log1p(g['checked']),float(missing.sum()/den),float(need[missing].sum()/den),float(need[missing].max(initial=0)),float(required/den)]
   if g['available']:
    valid=(g['roles']>=0)&unique;match=pp[np.arange(len(vv)),g['roles'].clip(0)] if len(vv) else np.zeros(0);zero=(g['roles']==0)&unique;active=(g['roles']>0)&unique;cc=np.log1p(g['counts']);diff=cc-profile[5:]
    f += [float(match[valid].mean()) if valid.any() else 0,float(match[valid].min(initial=1)),float(need[zero].sum()/den),float(pp[active,0].sum()/den),float(profile[g['root']]),*diff.tolist(),*np.abs(diff).tolist(),float(np.abs(diff).sum()),float(active.sum()/den),float(need[active].sum()/max(1,required))]
   else:f += [0.]*16
   features.append(f)
  rows.append(features)
 out=np.array(rows,dtype=np.float32)
 if out.size and out.shape[-1]!=23:raise ValueError(out.shape)
 return out,graphs
