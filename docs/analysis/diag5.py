import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
R="/home/rohan/ws/2026/jev/.claude/worktrees/train-eval-scripts/"
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
load=lambda p:[json.loads(l) for l in open(p)]
D={s:load(f"/home/rohan/ws/2026/jev/data/processed/{s}.jsonl") for s in ["train","val","test"]}
arr=lambda rows,k: np.array([[r[k][m] for m in M] for r in rows])
Y={s:arr(D[s],"pass_rate") for s in D}; C={s:arr(D[s],"cost") for s in D}; cbar=C["train"].mean(0)
def out(s,pick):
    i=np.arange(len(pick)); return Y[s][i,pick].mean(), C[s][i,pick].mean()
def frontier(S,s):
    return np.array([out(s,np.argmax(S-lam*cbar[None],1))[::-1] for lam in np.concatenate([[0],np.logspace(-3,2,400)])])
def gain(pts,s):
    f=(C[s][:,0].mean(),Y[s][:,0].mean()); t=(C[s][:,2].mean(),Y[s][:,2].mean())
    o=np.argsort(pts[:,0]); xs=pts[o,0]; ys=np.maximum.accumulate(pts[o,1])
    g=np.linspace(f[0],t[0],60); line=f[1]+(t[1]-f[1])*(g-f[0])/(t[0]-f[0])
    best=np.interp(g,xs,ys,left=f[1]); ok=ys>=t[1]-1e-9
    return np.mean(best-line), np.interp(0.1,xs,ys), (xs[ok].min() if ok.any() else np.nan)
for s in ["val","test"]:
    P=load(R+f"outputs/preds/exp1c-{s}.jsonl"); pm={p["id"]:p["probabilities"] for p in P}
    Q=np.array([[pm[r["id"]][k] for k in M+["none"]] for r in D[s]])
    cum=np.cumsum(Q[:,:3],1)
    print(f"\n== Exp1c {s}")
    am=Q.argmax(1)
    for nm,fb in [("none->5.2",2),("none->flash",0)]:
        pick=np.where(am==3,fb,am); print(" argmax, %s: solve %.3f cost %.4f"%((nm,)+out(s,pick)))
    print(" always flash %.3f %.4f | always 5.2 %.3f %.4f"%(out(s,np.zeros(len(am),int))+out(s,np.full(len(am),2))))
    print(" cumulative-prob utility frontier: gain over mixing line %.4f | solve@$0.10 %.3f | cost to match 5.2 %.3f"%gain(frontier(cum,s),s))
    print(" AUC any-solve %.3f flash-solve %.3f"%(roc_auc_score((Y[s].max(1)>=.5),1-Q[:,3]), roc_auc_score(Y[s][:,0]>=.5,cum[:,0])))
    mask=Y[s][:,0]<0.5
    print(" AUC 5.2-solves among flash-fail tasks (1c: P(mini)+P(5.2)) %.3f  (n=%d)"%(roc_auc_score(Y[s][mask,2]>=.5,Q[mask,1]+Q[mask,2]),mask.sum()))
E=np.load("/home/rohan/.claude/jobs/18e1bdd8/tmp/emb.npz")
tf=TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2),sublinear_tf=True).fit([r["state"] for r in D["train"]])
X={s:tf.transform([r["state"] for r in D[s]]) for s in D}
for nm,Xs in [("tfidf",X),("minilm",{s:E[s] for s in D})]:
    for C_ in [0.1,1]:
        P={s:np.zeros((len(D[s]),3)) for s in ["val","test"]}
        for k in range(3):
            clf=LogisticRegression(C=C_,max_iter=3000).fit(Xs["train"],(Y["train"][:,k]>=.5).astype(int))
            for s in P: P[s][:,k]=clf.predict_proba(Xs[s])[:,1]
        for s in P:
            g=gain(frontier(P[s],s),s); mask=Y[s][:,0]<.5
            print("%s C=%s %s: frontier gain %.4f solve@$0.10 %.3f cost-to-match %.3f | uplift AUC (5.2 solves | flash fails) %.3f"%((nm,C_,s)+g+(roc_auc_score(Y[s][mask,2]>=.5,P[s][mask,2]),)))
