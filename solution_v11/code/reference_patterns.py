"""Retrieve symbolic relation programs mined from train-reference calculator steps.
Only number-slot programs, TF-IDF vectors and vocabulary survive export: no
training answers, references, question IDs, or nearest-neighbor numeric answers.
"""
import re,collections
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from programs import question_numbers,induce,evaluate,program_key,slots
from common import parse_answer

def text_view(question):
 _,q=question_numbers(question)
 last=list(re.finditer(r'\b(?:how many|how much|what|calculate|find)\b',q))
 goal=q[last[-1].start():] if last else q[-180:]
 return q+' goal '+goal+' goal '+goal

def build(bank,refs):
 rows=[];programs=[];counts=[];rejected=collections.Counter()
 for row,ref in zip(bank,refs):
  assert row['id']==ref['id'] and row['question']==ref['question']
  vv,_=question_numbers(row['question'])
  if len(vv)!=len(set(vv)):
   rejected['ambiguous_repeated_quantity']+=1;continue
  node=induce(row['question'],ref['answer'])
  if node is None:rejected['unrecovered']+=1;continue
  # Partial reference calculations are accepted only when the resulting program
  # exactly reconstructs the reference final answer from original number slots.
  if not slots(node) or str(evaluate(node,vv))!=parse_answer(ref['answer']):continue
  rows.append(text_view(row['question']));programs.append(node);counts.append(len(vv))
 vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=16000,sublinear_tf=True,stop_words='english',dtype=np.float32)
 matrix=vectorizer.fit_transform(rows)
 return {'vectorizer':vectorizer,'matrix':matrix,'programs':programs,'counts':np.asarray(counts),'metadata':{'training_questions':len(bank),'library_rows':len(rows),'distinct_programs':len({program_key(n) for n in programs}),'rejected':dict(rejected),'raw_reference_text_exported':False,'training_answer_exported':False}}

NFEATURES=12
NAMES=['support_mass','opposition_mass','max_support_similarity','max_opposition_similarity','similarity_margin','top_neighbor_support','support_fraction','supporting_neighbors','strong_support_mass','strong_opposition_mass','same_program_support','distinct_support_programs']
def features(bank,library,top=12):
 if not bank:return np.zeros((0,4,NFEATURES))
 query=library['vectorizer'].transform([text_view(row['question']) for row in bank]);result=[];meta=[]
 for start in range(0,len(bank),128):
  similarity=(query[start:start+128]@library['matrix'].T).toarray()
  for offset,row in enumerate(bank[start:start+128]):
   vv,_=question_numbers(row['question']);ss=similarity[offset];ss[library['counts']!=len(vv)]=-1
   k=min(top,len(ss));ids=np.argpartition(ss,-k)[-k:];ids=ids[np.argsort(ss[ids])[::-1]]
   proposals=[]
   for j in ids:
    sim=float(ss[j])
    if sim<.15:continue
    try:answer=str(evaluate(library['programs'][j],vv))
    except (ValueError,IndexError,ZeroDivisionError,OverflowError,RecursionError):continue
    proposals.append((answer,sim,program_key(library['programs'][j])))
   answers=[parse_answer(c['text']) for c in row['candidates']];valid=[p for p in proposals if p[0] in answers]
   features_row=[]
   for answer in answers:
    support=[p for p in valid if answer is not None and p[0]==answer];opp=[p for p in valid if p[0]!=answer]
    # Nonmatching programs remain uncertainty, not negative votes.
    weight=lambda p:p[1]**4
    sm=sum(weight(p) for p in support);om=sum(weight(p) for p in opp);mx=max((p[1] for p in support),default=0);ox=max((p[1] for p in opp),default=0)
    pp=collections.Counter(p[2] for p in support)
    features_row.append([sm,om,mx,ox,mx-ox,float(bool(valid and valid[0][0]==answer))*valid[0][1] if valid else 0,sm/max(sm+om,1e-8),len(support)/top,sum(weight(p) for p in support if p[1]>=.4),sum(weight(p) for p in opp if p[1]>=.4),max(pp.values(),default=0)/top,len(pp)/top])
   result.append(features_row);meta.append({'proposals':len(proposals),'candidate_matching':len(valid),'max_similarity':max((p[1] for p in proposals),default=0)})
 return np.asarray(result,dtype=np.float64)
