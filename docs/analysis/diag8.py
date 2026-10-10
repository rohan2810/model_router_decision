import json, numpy as np
from scipy.stats import norm
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
load=lambda p:[json.loads(l) for l in open(p)]
D={s:load(f"/home/rohan/ws/2026/jev/data/processed/{s}.jsonl") for s in ["train","val","test"]}
arr=lambda rows,k: np.array([[r[k][m] for m in M] for r in rows])
Y={s:arr(D[s],"pass_rate") for s in D}; C={s:arr(D[s],"cost") for s in D}; cbar=C["train"].mean(0)
rng=np.random.default_rng(0)
def out(s,pick):
    i=np.arange(len(pick)); return C[s][i,pick].mean(), Y[s][i,pick].mean()
def gain(S,s):
    pts=np.array([out(s,np.argmax(S-lam*cbar[None],1)) for lam in np.concatenate([[0],np.logspace(-3,2,300)])])
    f=(C[s][:,0].mean(),Y[s][:,0].mean()); t=(C[s][:,2].mean(),Y[s][:,2].mean())
    o=np.argsort(pts[:,0]); xs=pts[o,0]; ys=np.maximum.accumulate(pts[o,1])
    g=np.linspace(f[0],t[0],60); line=f[1]+(t[1]-f[1])*(g-f[0])/(t[0]-f[0])
    return np.mean(np.interp(g,xs,ys,left=f[1])-line), np.interp(0.05,xs,ys)-(f[1]+(t[1]-f[1])*(0.05-f[0])/(t[0]-f[0]))
print("simulated router with per-model AUC=a (noise shared across models rho=0.5); gain = mean solve above flash<->5.2 mixing line")
for s in ["val","test"]:
    y=(Y[s]>=0.5).astype(float); n=len(y)
    for a in [0.55,0.60,0.65,0.70,0.75,0.80,0.90]:
        d=np.sqrt(2)*norm.ppf(a); gs=[];g5=[]
        for _ in range(30):
            shared=rng.normal(size=(n,1)); e=rng.normal(size=(n,3))
            z=np.sqrt(.5)*shared+np.sqrt(.5)*e
            score=y*d+z
            base=np.log(y.mean(0)/(1-y.mean(0)))
            p=1/(1+np.exp(-(base+d*(score-d/2))))
            gg=gain(p,s); gs.append(gg[0]); g5.append(gg[1])
        print(f" {s} AUC {a:.2f}: frontier gain {np.mean(gs):+.3f} | gain at $0.05/task {np.mean(g5):+.3f}")
    # oracle
    print(f" {s} oracle (perfect y): frontier gain %+.3f | gain at $0.05 %+.3f"%gain(y,s))
