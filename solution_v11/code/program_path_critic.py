"""Learn question-to-candidate arithmetic structure using references and negatives."""
import re,collections
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize
from programs import question_numbers,induce,slots
from quantity_relations import role,counts,OPS,slot_texts
H=FeatureHasher(n_features=32768,input_type='dict',alternate_sign=True,dtype=np.float64)
STOP=set('the a an is are was were how many much what does do did will can to of at on in for from that this and or but his her their it they she he if as by be after before with all find calculate'.split())

def graph_ref(row,ref):
 values,_=question_numbers(row['question'])
 if len(values)!=len(set(values)):return None
 node=induce(row['question'],ref['answer'])
 if node is None:return None
 try:return {'available':True,'roles':role(node,values),'counts':counts(node),'root':OPS.index(node[0]) if node[0] in OPS else 4,'final_slots':slots(node),'program':node}
 except (ValueError,ZeroDivisionError,IndexError,RecursionError):return None

def representation(row,g):
 if g is None or not g['available']:return {}
 values,text=question_numbers(row['question']);parts=re.split(r'\b(?:how many|how much|what|calculate|find)\b',text);goal=parts[-1];words=[w for w in re.findall(r'[a-z]+',text) if len(w)>1 and w not in STOP and w!='num'];gw=[w for w in re.findall(r'[a-z]+',goal) if len(w)>1 and w not in STOP and w!='num'];tokens=set(words+['goal_'+w for w in gw]);prog=['root_'+OPS[g['root']],'slots_'+str(min(len(g['final_slots']),8))]
 for k,count in enumerate(g['counts']):
  prog.append('op_'+OPS[k]+'_'+str(min(int(count),3)))
  if count:prog.append('has_'+OPS[k])
 out={}
 for p in prog:
  out['P:'+p]=1
  for w in tokens:out['T:'+w+'|'+p]=1
 st=slot_texts(row)
 for i,r in enumerate(g['roles']):
  if r<0 or i>=len(st):continue
  # Skip equal-valued quantities to avoid binding to the wrong object.
  if values.count(values[i])>1:continue
  local=set(re.findall(r'[a-z_]+',st[i].lower()))
  for w in local:
   if len(w)>2 and w not in STOP:out['S:'+w+'|role_'+str(r)]=1
 node=g.get('program')
 if node is not None:
  paths=[]
  def walk(z,path=()):
   if z[0] in ('q','c'):
    if z[0]=='q':
     i=int(z[1]);leaf='slot_'+str(i);local=set(re.findall(r'[a-z_]+',st[i].lower())) if i<len(st) else set()
    else:
     k=str(z[1]);leaf='const_'+(k if k in ('0','1','-1','2','10','12','24','60','100','1000','1/2','1/4') else 'other');local=set()
    structure='/'.join(path[-3:]);paths.append('leaf:'+leaf+'|'+structure)
    for w in local:
     if len(w)>2 and w not in STOP:out['PATHLOCAL:'+w+'|'+structure]=1
    return
   tag=z[0];walk(z[1],path+(tag if tag in ('+','*') else tag+'L',));walk(z[2],path+(tag if tag in ('+','*') else tag+'R',))
  walk(node)
  for p in set(paths):
   out['PATH:'+p]=1
   for w in tokens:out['PATHQ:'+w+'|'+p]=1
 return out

def matrices(bank,graphs,refs=None):
 dictionaries=[representation(row,g) for row,group in zip(bank,graphs) for g in group]
 x=normalize(H.transform(dictionaries),norm='l2',copy=False)
 if refs is None:return x,None
 refgraphs=[graph_ref(row,ref) for row,ref in zip(bank,refs)]
 xr=normalize(H.transform(representation(row,g) for row,g in zip(bank,refgraphs)),norm='l2',copy=False)
 return x,xr
