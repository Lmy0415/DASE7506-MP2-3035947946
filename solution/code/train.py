"""Train on train references, select on validation; never opens test references."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import HashingVectorizer
from common import load_inputs,read_jsonl
from features import features, labels,lexical_texts

def metrics(scores,y,hard):
    correct=y[np.arange(len(y)),scores.argmax(1)]
    return {'accuracy':float(100*correct.mean()),'hard_accuracy':float(100*correct[hard].mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('../runs')); args=p.parse_args()
    out=args.output;out.mkdir(exist_ok=True,parents=True);tick=time.perf_counter();torch.set_num_threads(4)
    banks=[];es=[];fs=[];ys=[]
    for split in ['train','validation']:
        bank,arr=load_inputs(split,args.data_dir);banks.append(bank);es.append(arr['embeddings']);fs.append(features(bank,es[-1]));ys.append(labels(bank,read_jsonl(args.data_dir/f'gsm8k_{split}.jsonl')))
    from common import parse_answer
    hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'] if parse_answer(c['text']) is not None))>1 for row in banks[1]])
    records=[];scores={};assets={}
    def record(name,s,asset=None,extra=None):
        scores[name]=s; r={'name':name,**metrics(s,ys[1],hard),**(extra or {})};records.append(r); print(json.dumps(r),flush=True)
        if asset is not None: assets[name]=asset
        (out/'search.json').write_text(json.dumps(records,indent=2))
    record('first',np.tile([1,0,0,0],(len(hard),1)))
    record('consensus',fs[1][...,1]+fs[1][...,0]*.001)
    np.savez_compressed(out/'development_labels.npz',validation=ys[1],hard=hard)
    # Logistic baselines and a compact lexical ridge model.
    for mode in ['style','embedding','context']:
        xx=fs if mode=='style' else es if mode=='embedding' else [np.concatenate([e,e-e.mean(1,keepdims=True),f],-1) for e,f in zip(es,fs)]
        mu=xx[0].reshape(-1,xx[0].shape[-1]).mean(0);sd=xx[0].reshape(-1,xx[0].shape[-1]).std(0).clip(.05)
        zz=[((x-mu)/sd).reshape(-1,len(mu)) for x in xx]
        for c in [.001,.01,.1,1]:
            model=LogisticRegression(C=c,max_iter=400,solver='lbfgs').fit(zz[0],ys[0].ravel())
            s=model.decision_function(zz[1]).reshape(-1,4)
            name=f'lr_{mode}_{c}';record(name,s,{'kind':'linear','mode':mode,'mu':mu,'sd':sd,'weight':model.coef_[0].astype('float32'),'bias':model.intercept_.astype('float32')})
    for full in [False,True]:
        hv=HashingVectorizer(n_features=32768,ngram_range=(1,2),alternate_sign=True,norm='l2')
        xx=[hv.transform(lexical_texts(b,full)) for b in banks]
        for alpha in [1,10,100]:
            m=Ridge(alpha=alpha,solver='lsqr').fit(xx[0],ys[0].ravel())
            record(f'lexical_{full}_{alpha}',m.predict(xx[1]).reshape(-1,4),{'kind':'lexical','full':full,'weight':m.coef_.astype('float32'),'bias':np.array([m.intercept_],dtype='float32')})
    # Small neural candidate scorer with a shared function for all candidate positions.
    for mode in ['embedding','context']:
        xx=es if mode=='embedding' else [np.concatenate([e,e-e.mean(1,keepdims=True),f],-1) for e,f in zip(es,fs)]
        mu=xx[0].reshape(-1,xx[0].shape[-1]).mean(0);sd=xx[0].reshape(-1,xx[0].shape[-1]).std(0).clip(.05)
        zz=[torch.tensor((x-mu)/sd) for x in xx]; y=torch.tensor(ys[0])
        for seed in [17,29,43]:
            for wd in [.01,.1]:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=torch.nn.Sequential(torch.nn.Linear(len(mu),96),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,1))
                opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=wd)
                for epoch in range(1,41):
                    model.train()
                    for idx in np.array_split(rng.permutation(len(y)),20):
                        s=model(zz[0][idx]).squeeze(-1);yy=y[idx]
                        loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(yy==0,-1e9),1)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(s,yy)
                        opt.zero_grad();loss.backward();opt.step()
                    if epoch in [5,10,20,30,40]:
                        model.eval()
                        with torch.no_grad(): s=model(zz[1]).squeeze(-1).numpy()
                        asset={'kind':'mlp','mode':mode,'mu':mu,'sd':sd}
                        for k,layer in enumerate([model[0],model[3],model[6]]):
                            asset[f'w{k}']=layer.weight.detach().numpy().T.copy();asset[f'b{k}']=layer.bias.detach().numpy().copy()
                        record(f'mlp_{mode}_s{seed}_wd{wd}_e{epoch}',s,asset)
    # One fixed validation criterion, favor the nontrivial groups matching test construction.
    ranked=sorted(assets,key=lambda n:(metrics(scores[n],ys[1],hard)['hard_accuracy'],metrics(scores[n],ys[1],hard)['accuracy']),reverse=True)
    selected=[ranked[0]];best=scores[selected[0]]; best_metric=metrics(best,ys[1],hard)['hard_accuracy']
    # Greedy equal-weight ensemble, at most five members, strict improvement only.
    for _ in range(4):
        options=[]
        for n in ranked:
            if n in selected:continue
            avg=sum(scores[k] for k in selected+[n])/ (len(selected)+1)
            options.append((metrics(avg,ys[1],hard)['hard_accuracy'],n,avg))
        value,n,avg=max(options,key=lambda t:t[0])
        if value<=best_metric: break
        selected.append(n);best=avg;best_metric=value
    bundle=out/'model_bundle';bundle.mkdir(exist_ok=True)
    config={'members':[],'criterion':'validation hard-group accuracy; overall accuracy breaks single-model ties','selected':selected}
    for i,n in enumerate(selected):
        asset=assets[n]; meta={k:v for k,v in asset.items() if not isinstance(v,np.ndarray)}
        np.savez(bundle/f'member{i}.npz',**{k:v for k,v in asset.items() if isinstance(v,np.ndarray)})
        config['members'].append({**meta,'file':f'member{i}.npz','name':n})
    (bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    record('selected_ensemble',best,extra={'members':selected})
    np.savez_compressed(out/'validation_scores.npz',**scores)
    # Preserve all trained candidates for reproducibility and validation-only inspection.
    archive=out/'development_models';archive.mkdir(exist_ok=True)
    for n,a in assets.items():
        np.savez(archive/f'{n}.npz',**{k:v for k,v in a.items() if isinstance(v,np.ndarray)})
    summary={'training_seconds':time.perf_counter()-tick,'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'numpy':np.__version__,'hard_validation_n':int(hard.sum()),'selected':selected,'metrics':metrics(best,ys[1],hard),'asset_bytes':sum(p.stat().st_size for p in bundle.iterdir()),'train_questions':len(banks[0]),'validation_questions':len(banks[1])}
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
