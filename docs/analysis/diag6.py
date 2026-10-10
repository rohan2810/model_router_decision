import json, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
load=lambda p:[json.loads(l) for l in open(p)]
D={s:load(f"/home/rohan/ws/2026/jev/data/processed/{s}.jsonl") for s in ["train","val","test"]}
def mut(i):
    m=re.sub(r"__[a-z0-9]{8}$","",i.split(".")[-1])
    if m.startswith("pr_"): return "pr_mirror"
    if m.startswith("func_pm_"): return "procedural"
    return m
rows=[]
for s in ["train","val"]:
    for r in D[s]:
        rows.append(dict(split=s,mut=mut(r["id"]),repo=r["repo"],**{m:r["pass_rate"][m] for m in M}))
df=pd.DataFrame(rows); df["any"]=(df[M].max(axis=1)>=.5).astype(float)
print(df.groupby(["split","mut"]).agg(n=("any","size"),flash=(M[0],"mean"),mini=(M[1],"mean"),g52=(M[2],"mean"),anysolve=("any","mean")).round(3).to_string())
# mutation-type-only predictor: train mean rate per mut, AUC on val
tr=df[df.split=="train"]; va=df[df.split=="val"]
rate=tr.groupby("mut")[M[0]].mean(); va=va.assign(p=va.mut.map(rate).fillna(tr[M[0]].mean()))
print("mutation-type-only AUC on val, flash solves: %.3f"%roc_auc_score(va[M[0]]>=.5,va.p))
rate=tr.groupby("mut")["any"].mean(); va=va.assign(p=va.mut.map(rate).fillna(tr["any"].mean()))
print("mutation-type-only AUC on val, any solves: %.3f"%roc_auc_score(va["any"]>=.5,va.p))
# repo+commit grouping in train: how many tasks share a repo commit
tr_ids=[r["id"] for r in D["train"]]
rc=pd.Series([".".join(i.split(".")[:2]) for i in tr_ids]).value_counts()
print("train: distinct repo@commit",len(rc),"median tasks per repo@commit",rc.median(),"max",rc.max())
# test: per-repo rates
t=pd.DataFrame([dict(repo=r["repo"],**{m:r["pass_rate"][m] for m in M}) for r in D["test"]])
print(t.groupby("repo").agg(n=(M[0],"size"),flash=(M[0],"mean"),mini=(M[1],"mean"),g52=(M[2],"mean")).round(2).sort_values("n",ascending=False).to_string())
