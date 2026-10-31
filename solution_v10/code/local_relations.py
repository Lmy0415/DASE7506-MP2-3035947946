"""Learn reusable pairwise operation relations from reference expression trees."""
import re,collections,itertools
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from reference_patterns import build,text_view
from programs import slots,question_numbers,evaluate,program_key
from common import parse_answer
LABELS=['unused','+','*','-left','-right','/left','/right']
H=HashingVectorizer(n_features=16384,alternate_sign=False,ngram_range=(1,2),norm='l2',dtype=np.float64)

def distribution(node,n):
 pairs=list(itertools.combinations(range(n),2));active=slots(node);out=[]
 for i,j in pairs:
  count=collections.Counter()
  def visit(t):
   if t[0] in ['q','c']:return
   left,right=slots(t[1]),slots(t[2]);tag=t[0]
   if i in left and j in right:count[tag if tag in ['+','*'] else tag+'left']+=1
   if j in left and i in right:count[tag if tag in ['+','*'] else tag+'right']+=1
   visit(t[1]);visit(t[2])
  if i not in active or j not in active:count['unused']=1
  else:visit(node)
  total=sum(count.values());out.append([count[k]/max(1,total) for k in LABELS])
 return np.asarray(out,dtype=np.float32).reshape(-1,7)

def pair_texts(question):
 values,q=question_numbers(question);pairs=list(itertools.combinations(range(len(values)),2));ctx=[]
 for m in re.finditer(r'NUM\d+',q):
  s=q[max(0,m.start()-55):m.end()+65];s=re.sub(r'NUM\d+','quantity',s)
  ctx.append(re.findall(r'[a-z]+',s))
 goal=re.split(r'\b(?:how many|how much|what|calculate|find)\b',q)[-1]
 texts=[]
 for i,j in pairs:
  # Position-marked neighborhoods preserve object and operation cues.
  texts.append(' '.join('left_'+w for w in ctx[i])+' '+' '.join('right_'+w for w in ctx[j])+' target '+goal+' '+text_view(question))
 return texts

def fit(bank,refs):
 library=build(bank,refs);texts=[];labels=[];weights=[];templates={};accepted=0
 # Recover nodes once through the vetted unique-quantity library.
 from programs import induce
 for row,ref in zip(bank,refs):
  vv,_=question_numbers(row['question'])
  if len(vv)!=len(set(vv)) or not 2<=len(vv)<=8:continue
  node=induce(row['question'],ref['answer'])
  if node is None:continue
  accepted+=1;templates.setdefault(len(vv),{})[program_key(node)]=node
  pp=pair_texts(row['question']);dd=distribution(node,len(vv))
  for text,probs in zip(pp,dd):
   for k in np.flatnonzero(probs):texts.append(text);labels.append(int(k));weights.append(float(probs[k])/max(1,len(pp)))
 model=LogisticRegression(C=3,solver='lbfgs',max_iter=220,tol=.001).fit(H.transform(texts),labels,sample_weight=weights)
 bins={}
 for n,mapping in templates.items():
  nodes=list(mapping.values());matrix=np.stack([distribution(t,n).ravel() for t in nodes]);bins[n]={'programs':nodes,'matrix':matrix}
 return {'model':model,'bins':bins,'metadata':{'training_questions':len(bank),'accepted_reference_programs':accepted,'pair_label_rows':len(labels),'programs':sum(len(v['programs']) for v in bins.values()),'labels':LABELS,'reference_text_exported':False}}

def features(bank,state):
 out=[];model=state['model']
 # Batch pair text vectorization/prediction; avoid one sklearn call per question.
 texts=[];lengths=[];numbers=[]
 for row in bank:
  vv,_=question_numbers(row['question']);numbers.append(vv);tt=pair_texts(row['question']) if len(vv) in state['bins'] else [];texts.extend(tt);lengths.append(len(tt))
 probabilities=np.zeros((len(texts),7))
 if texts:probabilities[:,model.classes_]=model.predict_proba(H.transform(texts))
 at=0
 for row,vv,length in zip(bank,numbers,lengths):
  if not length:out.append(np.zeros((4,12)));continue
  pp=probabilities[at:at+length];at+=length;group=state['bins'][len(vv)];logp=np.log(pp.clip(1e-6));scores=group['matrix']@logp.ravel()/length
  top=min(32,len(scores));ids=np.argpartition(scores,-top)[-top:];probs=softmax((scores[ids]-scores[ids].max())*4)
  proposals=[]
  for j,p in zip(ids,probs):
   try:a=str(evaluate(group['programs'][int(j)],vv))
   except (ValueError,IndexError,ZeroDivisionError,OverflowError,RecursionError):continue
   proposals.append((a,float(p),float(scores[j])))
  answers=[parse_answer(c['text']) for c in row['candidates']];valid=[z for z in proposals if z[0] in answers];features_row=[]
  confidence=float(pp.max(1).mean())
  for a in answers:
   yes=[z for z in valid if z[0]==a and a is not None];no=[z for z in valid if z[0]!=a];sm=sum(z[1] for z in yes);om=sum(z[1] for z in no);mx=max((z[1] for z in yes),default=0);ox=max((z[1] for z in no),default=0)
   features_row.append([sm,om,mx,ox,mx-ox,sm*confidence,sm/max(sm+om,1e-8),len(yes)/32,sm*float(confidence>=.5),om*float(confidence>=.5),float(bool(yes and max(yes,key=lambda z:z[2])[2]>=scores.max()-1e-6)),confidence*sm])
  out.append(features_row)
 return np.asarray(out).reshape(-1,4,12)
