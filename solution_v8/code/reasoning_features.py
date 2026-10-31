"""Bounded numeric equation checks and input-only answer-group comparisons.
No code from candidate texts is executed. Expressions use a restricted AST walker.
"""
import ast,operator,re
from fractions import Fraction
import numpy as np
from common import parse_answer
from features import features
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
NUM=r'[-+]?\d+(?:\.\d+)?'
EXPR=re.compile(r'(?<![\w.])([() ]*[+-]?\d[\d. ()+*/-]{0,180}?)\s*=\s*('+NUM+r')(?![\w.])')

def numeric_value(expr):
    if len(expr)>180:raise ValueError('long')
    def walk(n,depth=0):
        if depth>12:raise ValueError('deep')
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return float(n.value)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return walk(n.operand,depth+1)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return OPS[type(n.op)](walk(n.left,depth+1),walk(n.right,depth+1))
        raise ValueError('non-arithmetic')
    value=walk(ast.parse(expr.strip(),mode='eval').body)
    if not np.isfinite(value) or abs(value)>1e15:raise ValueError('large')
    return value

def normalize_math(text):
    s=text.replace(',','').replace('**','').replace('$','').replace('\\left','').replace('\\right','')
    s=re.sub(r'\\(?:text|mathrm|mathbf)\{[^{}]*\}',' ',s)
    for _ in range(5):
        s=re.sub(r'\\(?:d?frac)\{([^{}]*)\}\{([^{}]*)\}',r'((\1)/(\2))',s)
        s=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',s)
    for word in ['\\times','\\cdot','×']:s=s.replace(word,'*')
    s=s.replace('÷','/').replace('−','-').replace('\\div','/')
    s=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',s)
    return s

def arithmetic_features(text,answer):
    s=normalize_math(text);passed=[];errors=[];endpoints=[]
    for m in EXPR.finditer(s):
        expr=m.group(1).strip()
        if not re.search(r'[+*/-]',expr) or len(re.findall(NUM,expr))<2:continue
        try:
            lhs=numeric_value(expr);rhs=float(m.group(2));error=abs(lhs-rhs)/max(1,abs(lhs),abs(rhs))
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):continue
        passed.append(error<1e-5);errors.append(min(error,1));endpoints.append(rhs)
    # The scorer's #### precedence may disagree with the final visible box.
    boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
    box=parse_answer('\\boxed{'+boxes[-1]+'}') if boxes else None
    nums=re.findall(NUM,s)
    v=float(Fraction(answer)) if answer is not None else 0
    equal_last=bool(nums and abs(float(nums[-1])-v)<1e-6)
    endpoint_match=sum(abs(x-v)<1e-6 for x in endpoints)
    return [np.log1p(len(passed)),np.log1p(sum(passed)),np.log1p(len(passed)-sum(passed)),
        np.mean(passed) if passed else .5,max(errors,default=0),np.mean(errors) if errors else 0,
        bool(passed) and not passed[-1], endpoint_match/max(1,len(endpoints)),
        box is not None,answer is not None and box==answer,box is not None and box!=answer,
        equal_last,'####' in text,bool(re.search(r'####\s*\*?\*?\s*(step|job|part|case|option)',text,re.I)),
        text.count('\\boxed'),text.count('####')]

def enhanced_features(bank,embeddings):
    base=features(bank,embeddings);extra=[];e=np.asarray(embeddings,dtype=np.float32)
    norm=e/(np.linalg.norm(e,axis=-1,keepdims=True)+1e-8)
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']];group=[]
        for j,c in enumerate(row['candidates']):
            same=np.array([a is not None and a==answers[j] and k!=j for k,a in enumerate(answers)])
            other=np.array([a is not None and a!=answers[j] for a in answers]);sim=norm[i]@norm[i,j]
            group.append(arithmetic_features(c['text'],answers[j])+[
                float(sim[same].mean()) if same.any() else 0,float(sim[other].mean()) if other.any() else 0,
                float(sim[same].max()) if same.any() else 0,float(sim[other].max()) if other.any() else 0,
                float(np.linalg.norm(e[i,j]-e[i,same].mean(0))) if same.any() else 0,
                float(np.linalg.norm(e[i,j]-e[i,other].mean(0))) if other.any() else 0])
        extra.append(group)
    x=np.array(extra,dtype=np.float32)
    return np.concatenate([base,x,x-x.mean(1,keepdims=True)],axis=-1)

def hard_mask(bank):
    return np.array([len({a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None})>1 for r in bank])
