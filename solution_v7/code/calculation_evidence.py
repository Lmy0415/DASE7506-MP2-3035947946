"""Input-only exact arithmetic checks and conservative calculation-chain repair.
No execution of candidate programs. No question IDs, references, or gold answers.
"""
import ast,re
from fractions import Fraction
import numpy as np
from common import parse_answer
from reasoning_features import normalize_math

ALLOWED=re.compile(r'[0-9.()+*/\s-]+')

def value(node,changes=None,depth=0):
    if depth>16:raise ValueError('depth')
    if isinstance(node,ast.Expression):return value(node.body,changes,depth+1)
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):v=Fraction(str(node.value))
    elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        v=value(node.operand,changes,depth+1)*(-1 if isinstance(node.op,ast.USub) else 1)
    elif isinstance(node,ast.BinOp):
        a=value(node.left,changes,depth+1);b=value(node.right,changes,depth+1)
        if isinstance(node.op,ast.Add):v=a+b
        elif isinstance(node.op,ast.Sub):v=a-b
        elif isinstance(node.op,ast.Mult):v=a*b
        elif isinstance(node.op,ast.Div):v=a/b
        else:raise ValueError('operator')
    else:raise ValueError('syntax')
    # A complete earlier result can occur as a literal or fraction subtree.
    if changes:
        original=value(node,None,depth)
        if original in changes:return changes[original]
    if abs(v)>10**15 or v.denominator>10**12:raise ValueError('magnitude')
    return v

def parse(expr):
    if len(expr)>240 or len(re.findall(r'\d+',expr))>40:raise ValueError('long')
    expr=re.sub(r'(?<=\d)\s*\(', '*(',expr.strip())
    tree=ast.parse(expr,mode='eval');v=value(tree)
    return tree,v

def side(raw,left):
    # Only remove surrounding parentheses that belong to an outer equation.
    raw=(re.split(r'\n\s*\n',raw)[-1] if left else re.split(r'\n\s*\n',raw)[0]).strip()
    if left:
        while raw.startswith('(') and raw.count('(')>raw.count(')'):raw=raw[1:].lstrip()
    else:
        while raw.endswith(')') and raw.count(')')>raw.count('('):raw=raw[:-1].rstrip()
    try:return raw,*parse(raw)
    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    # Adjacent Markdown lines can be separate equations, never join their digits.
    lines=raw.splitlines();options=[]
    for n in range(1,min(len(lines),5)+1):
        candidate=' '.join(lines[-n:] if left else lines[:n]).strip()
        try:options.append((candidate,*parse(candidate)))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    return max(options,key=lambda z:len(z[0])) if options else None

def equations(text):
    lines=[];inside_math=False
    for line in text.splitlines():
        if not inside_math:line=re.sub(r'^\s*[-*]\s+','',line)
        lines.append(line)
        if line.count('$$')%2:inside_math=not inside_math
    text='\n'.join(lines)
    s=normalize_math(text)
    s=re.sub(r'\([^()\n]*[A-Za-z]{2,}[^()\n]*\)',' ',s)
    s=re.sub(r'(?<=\d)\s*(?:dollars?|minutes?|hours?|weeks?|days?|years?|meters?|metres?|miles?|liters?|litres?|ounces?|pesetas?|euros?|students?|people|cookies?|snails?|apples?|candies|slices?|pieces?|plates?|jars?|gallons?|shares?|cars?)\b',' ',s,flags=re.I)
    s=re.sub(r'\\[,;! ]',' ',s).replace('\\(',' ').replace('\\)',' ').replace('\\[',' ').replace('\\]',' ')
    s=s.replace('{','(').replace('}',')').replace('\\approx','≈').replace('\\Rightarrow','→').replace('\\rightarrow','→')
    output=[]
    for m in re.finditer('=',s):
        ls=re.search(r'[0-9.()+*/\s-]+$',s[:m.start()]);rs=ALLOWED.match(s,m.end())
        if not ls or not rs:continue
        # Do not interpret 3x or x3 as a standalone numeric operand.
        if ls.start()>0 and s[ls.start()-1].isalnum() and not ls.group()[0].isspace():continue
        if rs.end()<len(s) and s[rs.end()].isalnum() and not rs.group()[-1].isspace():continue
        left=side(ls.group(),True);right=side(rs.group(),False)
        if left is None or right is None:continue
        l,lt,lv=left;r,rt,rv=right
        if not any(isinstance(n,ast.BinOp) for n in ast.walk(lt)):continue
        # Pure arithmetic expressions can span whitespace, but not new paragraphs.
        context=s[max(0,m.start()-90):min(len(s),m.end()+100)].lower()
        diff=abs(lv-rv)
        rounded=False
        if re.fullmatch(r'-?\d*\.\d+',r):
            places=len(r.split('.')[-1]);rounded=diff<=Fraction(1,2*10**places)
        if any(w in context for w in ['round','nearest','approximately','whole','ceil','floor']) and diff<=Fraction(1,2):rounded=True
        output.append({'lhs':l,'rhs':r,'left':lt,'right':rt,'lv':lv,'rv':rv,'valid':lv==rv or rounded,'rounded':rounded,'position':m.start()})
    return output

def interval(node,bounds,depth=0):
    if depth>16:raise ValueError('depth')
    original=value(node)
    if original in bounds:return bounds[original]
    if isinstance(node,ast.Expression):return interval(node.body,bounds,depth+1)
    if isinstance(node,ast.Constant):return original,original
    if isinstance(node,ast.UnaryOp):
        a,b=interval(node.operand,bounds,depth+1)
        return (-b,-a) if isinstance(node.op,ast.USub) else (a,b)
    if isinstance(node,ast.BinOp):
        a,b=interval(node.left,bounds,depth+1);c,d=interval(node.right,bounds,depth+1)
        if isinstance(node.op,ast.Add):return a+c,b+d
        if isinstance(node.op,ast.Sub):return a-d,b-c
        if isinstance(node.op,ast.Mult):v=[a*c,a*d,b*c,b*d];return min(v),max(v)
        if isinstance(node.op,ast.Div):
            if c<=0<=d:raise ValueError('uncertain zero denominator')
            v=[a/c,a/d,b/c,b/d];return min(v),max(v)
    raise ValueError('syntax')

def analyze(question,text):
    rows=equations(text);bounds={}
    for row in rows:
        try:
            lo,hi=interval(row['left'],bounds)
            if lo<=row['rv']<=hi and lo!=hi:
                row['valid']=True;row['rounded']=True;bounds[row['rv']]=(lo,hi)
            elif row['rounded'] and row['lv']!=row['rv']:
                places=len(row['rhs'].split('.')[-1]) if re.fullmatch(r'-?\d*\.\d+',row['rhs']) else 0
                step=Fraction(1,2*10**places);bounds[row['rv']]=(row['rv']-step,row['rv']+step)
        except (ValueError,ZeroDivisionError):pass
    answer=parse_answer(text);a=Fraction(answer) if answer is not None else None
    # An earlier intermediate result that equals an original question quantity is
    # ambiguous; do not substitute it through unrelated calculations.
    qnums={Fraction(n.replace(',','')) for n in re.findall(r'(?<![\w.])-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)',question)}
    for numerator,denominator in re.findall(r'(\d+)\s*/\s*(\d+)',question):
        if int(denominator):qnums.add(Fraction(int(numerator),int(denominator)))
    small=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(21)))
    small.update({'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'twice':2,'double':2,'triple':3,'half':Fraction(1,2),'quarter':Fraction(1,4),'dozen':12})
    words=re.findall(r'[a-z]+',question.lower())
    qnums.update(Fraction(small[w]) for w in words if w in small)
    for word_left,word_right in zip(words,words[1:]):
        if word_left in small and word_right in small and 20<=small[word_left]<=90 and 0<small[word_right]<10:qnums.add(Fraction(small[word_left]+small[word_right]))
    # Standard conversion constants can legitimately recur independently.
    qnums.update(Fraction(n) for n in [0,1,2,10,12,24,60,100,1000])
    changes={};ambiguous=set();errors=[];repair_steps=[]
    for row in rows:
        if not row['valid']:errors.append({'lhs':row['lhs'],'rhs':row['rhs'],'expected':str(row['lv']),'claimed':str(row['rv'])})
        try:corrected=value(row['left'],changes)
        except (ValueError,ZeroDivisionError):continue
        claimed=row['rv']
        if row['rounded'] or corrected==claimed:continue
        if claimed in qnums or claimed in ambiguous:continue
        if claimed in changes and changes[claimed]!=corrected:
            changes.pop(claimed,None);ambiguous.add(claimed);continue
        changes[claimed]=corrected
        repair_steps.append({'claimed':str(claimed),'corrected':str(corrected),'lhs':row['lhs']})
    repaired=changes.get(a) if a is not None else None
    if not errors:repaired=None
    boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
    box=parse_answer('\\boxed{'+boxes[-1]+'}') if boxes else None
    return {'answer':answer,'box':box,'box_mismatch':box is not None and answer!=box,'checked':len(rows),'errors':errors,'repaired_answer':str(repaired) if repaired is not None and repaired!=a else None,'repair_steps':repair_steps,'ambiguous_results':len(ambiguous),'valid':sum(r['valid'] for r in rows)}

def evidence_features(bank,local_gate=False):
    outputs=[];details=[]
    for row in bank:
        aa=[analyze(row['question'],c['text']) for c in row['candidates']];details.append(aa);features=[]
        for j,a in enumerate(aa):
            answer=a['answer'];incoming=[z for k,z in enumerate(aa) if k!=j and z['repaired_answer']==answer and answer is not None];outmatch=any(z['answer']==a['repaired_answer'] for k,z in enumerate(aa) if k!=j and a['repaired_answer'] is not None)
            boxsupport=sum(z['box']==answer and z['box_mismatch'] for z in aa if answer is not None)
            relative=[float(abs(Fraction(e['expected'])-Fraction(e['claimed']))/max(1,abs(Fraction(e['expected'])))) for e in a['errors']]
            f=[a['box_mismatch'],boxsupport,a['box'] is not None, np.log1p(a['checked']),np.log1p(len(a['errors'])),len(a['errors'])/max(1,a['checked']),max(relative,default=0),a['repaired_answer'] is not None,outmatch,len(incoming),len({z['answer'] for z in incoming}),len(incoming)/max(1,sum(z['answer']==answer for z in aa)),a['valid']/max(1,a['checked']),a['ambiguous_results']>0]
            relevant=bool(a['errors'] or a['box_mismatch'] or incoming or boxsupport)
            features.append(f if (relevant or not local_gate) else [0.]*len(f))
        # Arithmetic evidence does not establish semantic correctness. If there is no
        # arithmetic/format anomaly anywhere in the group, leave legacy rankings intact.
        trigger=any(a['errors'] or a['box_mismatch'] for a in aa)
        outputs.append(features if trigger else np.zeros_like(features))
    x=np.asarray(outputs,dtype=np.float32)
    if not len(bank):return np.empty((0,4,28),np.float32),details
    x[:,:,6]=np.minimum(x[:,:,6],10)
    return np.concatenate([x,x-x.mean(1,keepdims=True)],-1),details
