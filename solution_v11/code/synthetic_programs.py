"""Train-only structured negatives with provably different numerical answers."""
from fractions import Fraction
from programs import question_numbers,induce,evaluate,program_key,slots
from quantity_relations import role,counts,OPS

def mutations(node,depth=0):
 if depth>10 or node[0] in ('q','c'):return []
 tag,left,right=node;out=[left,right]
 for op in ['+','-','*','/']:
  if op!=tag:out.append((op,left,right))
 if tag in ('-','/'):out.append((tag,right,left))
 for child in mutations(left,depth+1):out.append((tag,child,right))
 for child in mutations(right,depth+1):out.append((tag,left,child))
 return out

def structure(node,values):
 try:
  if not slots(node):return None
  return {'available':True,'program':node,'roles':role(node,values),'counts':counts(node),'root':OPS.index(node[0]) if node[0] in OPS else 4,'final_slots':slots(node)}
 except (ValueError,ZeroDivisionError,OverflowError,IndexError,RecursionError):return None

def generate(row,ref,max_examples=8):
 values,_=question_numbers(row['question'])
 if len(values)>12 or len(values)!=len(set(values)):return None,[]
 node=induce(row['question'],ref['answer'])
 if node is None:return None,[]
 try:gold=evaluate(node,values)
 except (ValueError,ZeroDivisionError,OverflowError,IndexError,RecursionError):return None,[]
 refgraph=structure(node,values)
 if refgraph is None:return None,[]
 unique=set();out=[]
 for candidate in mutations(node):
  key=program_key(candidate)
  if key in unique:continue
  unique.add(key)
  try:
   result=evaluate(candidate,values)
   if result==gold or abs(result)>10**12:continue
  except (ValueError,ZeroDivisionError,OverflowError,IndexError,RecursionError):continue
  graph=structure(candidate,values)
  if graph is not None:out.append(graph)
  if len(out)==max_examples:break
 return refgraph,out
