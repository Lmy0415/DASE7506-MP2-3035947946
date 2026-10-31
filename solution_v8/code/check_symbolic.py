import json
from fractions import Fraction
from symbolic_binding import affine,solve,extract,answer_binding
p=extract('Let $p$ be the price of a pencil.\nLet $e$ be the price of an eraser.\n$$20p+40e=80$$\n$$e=p/2$$')
assert p['resolved']['p']==2 and p['resolved']['e']==1,p
c,d,_=affine(r'100 \text{ balloons}\times 3 \text{ ounces per balloon}',{});assert c==300 and not d
c,d,_=affine(r'\frac{40,000 \text{ft}^2}{800 \text{ft}^2}',{});assert c==50 and not d,(c,d)
p=extract(r'Let $x$ be the price of an eraser. $x=80/40/2=1$');assert p['resolved']['x']==1,p
for expr in ['__import__("os").system("true")','x.__class__','2**100','x*y','x/y','1/0']:
 try:affine(expr,{})
 except (ValueError,SyntaxError,ZeroDivisionError):pass
 else:raise AssertionError(expr)
p=extract('$$x+y=10$$\n$$2x+y=14$$');assert p['resolved']=={'x':4,'y':6},p
row={'question':'How much profit is made from selling the shirts?'};cand={'text':'Let $p$ be the profit from shirts.\n$$p=100-60=40$$\n#### 40'}
b=answer_binding(row,cand);assert b['bound'] and 'profit' in b['context'],b
print(json.dumps({'status':'passed','checks':['two-variable affine solve','units and unit exponents inside fractions','symbol declaration to target labels','implicit multiplication','restricted AST rejects code and nonlinear expressions']}))
from provenance import used_quantities
row={'question':'Mary has 12 apples and buys 3 apples. John has 7 pears. How many apples does Mary have?'}
cand={'text':'$$12+3=15$$\n$$7*2=14$$\n#### 15'}
b=answer_binding(row,cand);used,ambiguous,_=used_quantities(row['question'],b);assert used=={0,1} and not ambiguous,(used,ambiguous)
row={'question':'Mary has 5 apples and John has 5 pears. How many apples after buying 3?'}
cand={'text':'$$5+3=8$$\n#### 8'}
b=answer_binding(row,cand);used,ambiguous,_=used_quantities(row['question'],b);assert used=={2} and ambiguous=={0,1},(used,ambiguous)
print(json.dumps({'status':'passed','checks':['trace final answer dependencies only','equal quantities with different owners remain ambiguous']}))
