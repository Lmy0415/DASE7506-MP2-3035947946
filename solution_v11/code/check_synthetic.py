from synthetic_programs import mutations
from programs import evaluate
from fractions import Fraction
a=('q',0);b=('q',1);root=('*',a,('+',b,('c','1')));values=[Fraction(8),Fraction(3)]
mut=mutations(root);assert ('*',a,b) in mut
assert evaluate(root,values)==32 and evaluate(('*',a,b),values)==24
print('passed: missing original amount generates a distinct answer')
