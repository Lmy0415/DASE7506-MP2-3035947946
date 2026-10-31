"""Induce question-number programs from training-reference calculator annotations.
Programs contain question-number slots and a fixed small set of general constants,
never stored question IDs, training answers or reference text at inference.
"""
import ast,re,json,math
from fractions import Fraction
from collections import Counter
import numpy as np
from common import parse_answer

SMALL={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'half':Fraction(1,2),'quarter':Fraction(1,4),'twice':2,'double':2,'triple':3}
CONSTANTS={Fraction(x) for x in [0,1,2,3,4,5,6,7,10,12,24,60,100,1000]}|{Fraction(1,2),Fraction(1,4),Fraction(1,100)}
TOKEN=re.compile(r'(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?|[a-z]+(?:-[a-z]+)?',re.I)

def question_numbers(question):
    # Keep numeric slots in textual order; normalize digit and simple word numbers alike.
    q=question.lower().replace(',','');tokens=[];numbers=[];matches=list(TOKEN.finditer(q));i=0;last=0
    while i<len(matches):
        m=matches[i];word=m.group();value=None;end=m.end()
        if word[0].isdigit() or word[0]=='.':
            try:value=Fraction(word)
            except (ValueError,ZeroDivisionError):pass
        else:
            parts=word.split('-')
            if all(p in SMALL for p in parts):
                value=sum(Fraction(SMALL[p]) for p in parts)
                if len(parts)==1 and SMALL[word]>=20 and i+1<len(matches) and matches[i+1].group() in SMALL and SMALL[matches[i+1].group()]<10 and not q[m.end():matches[i+1].start()].strip():
                    i+=1;value+=SMALL[matches[i].group()];end=matches[i].end()
        tokens.append(q[last:m.start()])
        if value is None:tokens.append(q[m.start():end])
        else:
            tokens.append(' NUM'+str(len(numbers))+' ');numbers.append(value)
        last=end;i+=1
    tokens.append(q[last:])
    return numbers,''.join(tokens)

def constant(v):return ('c',str(v))
def opnode(op,left,right):
    if left[0]=='c' and right[0]=='c':
        try:return constant(evaluate((op,left,right),[]))
        except (ZeroDivisionError,ValueError):pass
    if op in ('+','*') and repr(left)>repr(right):left,right=right,left
    return (op,left,right)

def evaluate(node,values,depth=0):
    if depth>18:raise ValueError('deep program')
    tag=node[0]
    if tag=='q':return values[int(node[1])]
    if tag=='c':return Fraction(node[1])
    a=evaluate(node[1],values,depth+1);b=evaluate(node[2],values,depth+1)
    v=a+b if tag=='+' else a-b if tag=='-' else a*b if tag=='*' else a/b if tag=='/' else None
    if v is None or abs(v)>10**15 or v.denominator>10**12:raise ValueError('unbounded')
    return v

def slots(node):
    if node[0]=='q':return {int(node[1])}
    if node[0]=='c':return set()
    return slots(node[1])|slots(node[2])

def induce(question,reference):
    values,normalized_question=question_numbers(question);known={};valid=[]
    percentage_slots={int(i) for i in re.findall(r'NUM(\d+)\s*%',normalized_question)}
    if len(values)>14:return None
    def expression(s):
        tree=ast.parse(s.strip().replace(',','').replace('^','**'),mode='eval').body
        def walk(n,depth=0):
            if depth>12:raise ValueError('deep')
            if isinstance(n,ast.Constant) and type(n.value) in (int,float):
                v=Fraction(str(n.value));indices=[i for i,x in enumerate(values) if x==v]
                if indices:return ('q',indices[0])
                pct=[i for i in percentage_slots if values[i]/100==v]
                if len(pct)==1:return ('/',('q',pct[0]),constant(100))
                if v in known:return known[v]
                if v in CONSTANTS:return constant(v)
                raise ValueError('unsupported constant')
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
                node=walk(n.operand,depth+1);return node if isinstance(n.op,ast.UAdd) else opnode('*',constant(-1),node)
            if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(n.op)];return opnode(op,walk(n.left,depth+1),walk(n.right,depth+1))
            raise ValueError('unsupported expression')
        return walk(tree)
    for calc in re.findall(r'<<([^<>]+)>>',reference):
        if '=' not in calc:continue
        lhs,rhs=calc.rsplit('=',1)
        try:
            result=Fraction(rhs.replace(',','').strip());node=expression(lhs)
            if evaluate(node,values)!=result:continue
            known[result]=node
            if slots(node) and len(repr(node))<700:valid.append((result,node))
        except (SyntaxError,ValueError,ZeroDivisionError,IndexError,OverflowError,RecursionError):continue
    gold=parse_answer(reference)
    if gold is None:return None
    matches=[n for v,n in valid if v==Fraction(gold)]
    return matches[-1] if matches else None

def program_key(node):return json.dumps(node,separators=(',',':')) if node is not None else 'OTHER'

def evidence(bank,templates,probabilities):
    out=[];mass=[]
    for row,probs in zip(bank,probabilities):
        values,_=question_numbers(row['question']);answers=[parse_answer(c['text']) for c in row['candidates']];acc={};matched=0
        for key,p in zip(templates,probs):
            if key=='OTHER':continue
            try:answer=str(evaluate(json.loads(key),values))
            except (ValueError,IndexError,ZeroDivisionError,OverflowError,RecursionError):continue
            if answer in answers:acc[answer]=acc.get(answer,0)+float(p);matched+=float(p)
        backoff=max(1-matched,1e-6)/4
        out.append([math.log(backoff+acc.get(ans,0)+1e-8) for ans in answers]);mass.append(matched)
    return np.asarray(out),np.asarray(mass)
