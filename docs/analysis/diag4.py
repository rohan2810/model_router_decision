import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import json, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
def load(s): return [json.loads(l) for l in open(f"data/processed/{s}.jsonl")]
D={s:load(s) for s in ["train","val","test"]}
arr=lambda rows,k: np.array([[r[k][m] for m in M] for r in rows])
Y={s:arr(D[s],"pass_rate") for s in D}; C={s:arr(D[s],"cost") for s in D}
# 1. oracle policies
print("== oracle policies (cheapest solver; none -> X)")
for s in ["val","test"]:
    y=Y[s]>=0.5; pick=np.where(y.any(1),y.argmax(1),-1)
    for name,fb in [("none->5.2",2),("none->flash",0)]:
        p=np.where(pick>=0,pick,fb); i=np.arange(len(p))
        print(f" {s} {name}: solve {Y[s][i,p].mean():.3f} cost {C[s][i,p].mean():.4f}   (always 5.2: {Y[s][:,2].mean():.3f} @ {C[s][:,2].mean():.4f})")
# 2. conditional structure
print("== conditional signal")
for s in ["train","val","test"]:
    b=(Y[s]>=0.5)
    print(f" {s}: P(5.2 solves | flash fails)={b[~b[:,0],2].mean():.3f}  P(flash solves | 5.2 fails)={b[~b[:,2],0].mean():.3f}  P(any solves)={b.any(1).mean():.3f}")
# 3. embeddings (MiniLM, CPU)
from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); enc=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
def embed(texts,bs=64):
    out=[]
    with torch.no_grad():
        for i in range(0,len(texts),bs):
            t=tok(texts[i:i+bs],padding=True,truncation=True,max_length=256,return_tensors="pt")
            h=enc(**t).last_hidden_state; m=t["attention_mask"].unsqueeze(-1).float()
            e=(h*m).sum(1)/m.sum(1); out.append(torch.nn.functional.normalize(e,dim=-1))
    return torch.cat(out).numpy()
E={s:embed([r["state"] for r in D[s]]) for s in D}
np.savez("/home/rohan/.claude/jobs/18e1bdd8/tmp/emb.npz",**E)
tf=TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2),sublinear_tf=True).fit([r["state"] for r in D["train"]])
X={s:tf.transform([r["state"] for r in D[s]]) for s in D}
def aucs(P,s):
    out=[]
    for k in range(3):
        b=(Y[s][:,k]>=0.5).astype(int); out.append(roc_auc_score(b,P[:,k]))
    out.append(roc_auc_score((Y[s].max(1)>=0.5).astype(int),P.max(1)))
    return np.round(out,3)
def fitP(Xtr,Ytr,Xs,C_):
    P={s:np.zeros((Xs[s].shape[0],3)) for s in Xs}
    for k in range(3):
        clf=LogisticRegression(C=C_,max_iter=3000).fit(Xtr,(Ytr[:,k]>=0.5).astype(int))
        for s in Xs: P[s][:,k]=clf.predict_proba(Xs[s])[:,1]
    return P
print("== embeddings LR AUC [flash,mini,5.2,any]")
for C_ in [0.3,1,10]:
    P=fitP(E["train"],Y["train"],{s:E[s] for s in ["val","test"]},C_)
    print(f" MiniLM C={C_} val {aucs(P['val'],'val')} test {aucs(P['test'],'test')}")
# 4. in-distribution Verified data: leave-one-repo-out over Verified, train=smith vs smith+other-Verified
print("== LORO on Verified: smith-only vs smith+other Verified repos (MiniLM LR C=1; mean AUC of 'any solves'/flash pooled)")
repos=np.array([r["repo"] for r in D["test"]]); ts=np.unique(repos)
Xs_=np.vstack([E["train"],E["val"]]); Ys_=np.vstack([Y["train"],Y["val"]])
pa=np.zeros((len(repos),3)); pb=np.zeros((len(repos),3))
for rp in ts:
    te=repos==rp
    pa[te]=fitP(Xs_,Ys_,{"t":E["test"][te]},1.0)["t"]
    pb[te]=fitP(np.vstack([Xs_,E["test"][~te]]),np.vstack([Ys_,Y["test"][~te]]),{"t":E["test"][te]},1.0)["t"]
print(" smith-only     :",aucs(pa,"test")); print(" smith+Verified :",aucs(pb,"test"))
# verified-only
pc=np.zeros((len(repos),3))
for rp in ts:
    te=repos==rp
    pc[te]=fitP(E["test"][~te],Y["test"][~te],{"t":E["test"][te]},1.0)["t"]
print(" Verified-only  :",aucs(pc,"test"))
