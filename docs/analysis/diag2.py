import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import json, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
def load(s): return [json.loads(l) for l in open(f"data/processed/{s}.jsonl")]
D={s:load(s) for s in ["train","val","test"]}
def arr(rows,key): return np.array([[r[key][m] for m in M] for r in rows])
Y={s:arr(D[s],"pass_rate") for s in D}; C={s:arr(D[s],"cost") for s in D}
cbar=C["train"].mean(0)
def frontier(P, s, lams=np.concatenate([[0],np.logspace(-3,2,300)])):
    # pick argmax p_k - lam*cbar_k per task
    pts=[]
    for lam in lams:
        pick=np.argmax(P-lam*cbar[None,:],1)
        pts.append((C[s][np.arange(len(pick)),pick].mean(), Y[s][np.arange(len(pick)),pick].mean()))
    return np.array(pts)
def summarize(P,s):
    pts=frontier(P,s)
    flash=(C[s][:,0].mean(),Y[s][:,0].mean()); top=(C[s][:,2].mean(),Y[s][:,2].mean())
    # gain over mixing line, averaged over cost grid between flash and top
    grid=np.linspace(flash[0],top[0],50)
    order=np.argsort(pts[:,0]); xs,ys=pts[order,0],pts[order,1]
    ymax=np.maximum.accumulate(ys)  # best solve achievable at <= cost
    best=np.interp(grid,xs,ymax,left=np.nan)
    line=flash[1]+(top[1]-flash[1])*(grid-flash[0])/(top[0]-flash[0])
    gain=np.nanmean(best-line)
    # cost to match top solve
    ok=xs[ymax>=top[1]-1e-9]
    return gain,(ok.min() if len(ok) else None), np.interp(0.1,xs,ymax) 
def aucs(P,s):
    out=[]
    for k in range(3):
        b=(Y[s][:,k]>=0.5).astype(int); out.append(roc_auc_score(b,P[:,k]) if 0<b.mean()<1 else np.nan)
    anys=(Y[s].max(1)>=0.5).astype(int); out.append(roc_auc_score(anys,P.max(1)))
    return np.round(out,3)
texts={s:[r["state"] for r in D[s]] for s in D}
def feats(t):
    return [np.log1p(len(t)), t.count("```")/2, int("Traceback" in t), int("Error" in t), t.count("\n"), len(re.findall(r"\bdef |\bclass ",t))]
F={s:np.array([feats(t) for t in texts[s]]) for s in D}
def fit_predict(Xtr,Xs,C_=1.0):
    P={s:np.zeros((len(Xs[s]) if not hasattr(Xs[s],"shape") else Xs[s].shape[0],3)) for s in Xs}
    for k in range(3):
        b=(Y["train"][:,k]>=0.5).astype(int)
        clf=LogisticRegression(C=C_,max_iter=2000).fit(Xtr,b)
        for s in Xs: P[s][:,k]=clf.predict_proba(Xs[s])[:,1]
    return P
from scipy.sparse import hstack, csr_matrix
res={}
tf=TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2),sublinear_tf=True).fit(texts["train"])
X={s:tf.transform(texts[s]) for s in D}
for name,Xs in [("tfidf",X)]:
    for C_ in [0.1,1,10]:
        P=fit_predict(Xs["train"],{s:Xs[s] for s in ["val","test"]},C_)
        for s in ["val","test"]:
            print(f"{name} C={C_} {s}: AUC[flash,mini,5.2,any]={aucs(P[s],s)} gain/cost_to_match5.2/solve@$0.1={summarize(P[s],s)}")
