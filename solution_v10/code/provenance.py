"""Backward trace verified numeric equations to unique question quantity slots.
Equal-valued question quantities remain ambiguous instead of selecting a person.
"""
import ast,re
from fractions import Fraction
from programs import question_numbers

def used_quantities(question,binding):
 values,normalized=question_numbers(question);nodes={};result=set();ambiguous=set()
 for record in binding['records']:
  try:
   tree=ast.parse(record['expr'],mode='eval')
   constants={Fraction(str(n.value)) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float)}
  except (SyntaxError,ValueError,ZeroDivisionError):continue
  if len(constants)>1 or any(isinstance(n,ast.BinOp) for n in ast.walk(tree)):
   # Only records with constant arithmetic and no symbols are provenance edges.
   if not any(isinstance(n,ast.Name) for n in ast.walk(tree)):nodes.setdefault(record['value'],[]).append(constants)
 def trace(value,seen):
  if value in seen:return
  ids=[i for i,v in enumerate(values) if v==value]
  if len(ids)==1:result.add(ids[0]);return
  if len(ids)>1:ambiguous.update(ids);return
  branches=nodes.get(value,[])
  # Diverging derivations cannot support a unique dependency assertion.
  if len({tuple(sorted(z)) for z in branches})!=1:return
  for v in branches[0]:trace(v,seen|{value})
 if binding['answer'] is not None and binding['bound']:trace(Fraction(binding['answer']),set())
 slots=[]
 for m in re.finditer(r'NUM(\d+)',normalized):
  i=int(m.group(1));following=normalized[m.end():];following=re.split(r'NUM\d+|[.!?;]',following)[0][:90]
  preceding=re.split(r'NUM\d+|[.!?;]',normalized[:m.start()])[-1][-55:]
  slots.append((i,preceding+' '+following))
 return result,ambiguous,slots
