"""Deterministic, candidate-permutation-equivariant input features; no references."""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer

WORDS = ['therefore','however','wait','alternatively','assume','approximately','round','integer','remaining','total','each','per','percent','more','less','not','check','correct','mistake','but','so','if','let','answer','step']

def features(bank, embeddings):
    rows=[]
    for row in bank:
        texts=[c['text'] for c in row['candidates']]
        answers=[parse_answer(t) for t in texts]
        qnums=set(re.findall(r'\d+(?:\.\d+)?',row['question']))
        nums=[float(Fraction(a)) if a is not None else 0. for a in answers]
        group=[]
        for j,(t,a,v) in enumerate(zip(texts,answers,nums)):
            lower=t.lower(); ns=re.findall(r'\d+(?:\.\d+)?',t)
            agree=sum(a is not None and a==b for b in answers)
            group.append([a is not None,agree/4, len(set(x for x in answers if x is not None))/4,
                np.log1p(len(t)),np.log1p(len(t.split())),np.log1p(len(ns)),
                np.log1p(abs(v)),np.sign(v),v==int(v),v==0,
                sum(v>x for x in nums)/4, sum(v==x for x in nums)/4,
                len(qnums&set(ns))/max(1,len(qnums)),len(set(ns)-qnums)/max(1,len(ns)),
                *[np.log1p(t.count(s)) for s in ['\n','=', '+','-','\\times','\\frac','\\boxed','####','**','?','!','%', '$']],
                *[np.log1p(lower.count(w)) for w in WORDS]])
        rows.append(group)
    f=np.asarray(rows,dtype=np.float32)
    e=np.asarray(embeddings,dtype=np.float32)
    # Group-relative features cancel much of the shared question representation.
    norm=e/(np.linalg.norm(e,axis=-1,keepdims=True)+1e-8)
    sims=norm@norm.transpose(0,2,1)
    simstats=np.stack([sims.mean(-1),np.sort(sims,axis=-1)[...,-2],np.sort(sims,axis=-1)[...,0]],-1)
    return np.concatenate([f,f-f.mean(1,keepdims=True),simstats],-1)


def labels(bank, references):
    assert len(bank)==len(references)
    result=[]
    for row,ref in zip(bank,references):
        assert row['id']==ref['id'] and row['question']==ref['question']
        gold=parse_answer(ref['answer'])
        result.append([float(parse_answer(c['text'])==gold and gold is not None) for c in row['candidates']])
    y=np.asarray(result,dtype=np.float32)
    assert (y.sum(1)>0).all()
    return y


def lexical_texts(bank, reasoning=True):
    for row in bank:
        for c in row['candidates']:
            text=c['text'] if reasoning else str(parse_answer(c['text']))
            # Remove absolute numeric identities; retain magnitude/format features separately.
            yield re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',row['question']+' SOLUTION '+text)
