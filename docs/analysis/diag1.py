import json, numpy as np, pandas as pd
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
def load(s): return [json.loads(l) for l in open(f"data/processed/{s}.jsonl")]
D={s:load(s) for s in ["train","val","test"]}
for s,rows in D.items():
    pr=np.array([[r["pass_rate"][m] for m in M] for r in rows]); n=np.array([[r["n_runs"][m] for m in M] for r in rows])
    cost=np.array([[r["cost"][m] for m in M] for r in rows])
    print(f"\n== {s}: {len(rows)} tasks")
    print(" mean pass rate",pr.mean(0).round(3)," mean cost",cost.mean(0).round(4))
    print(" n_runs mean",n.mean(0).round(2)," frac fractional pass_rate",((pr>0)&(pr<1)).mean(0).round(3))
    b=(pr>=0.5).astype(int)
    print(" phi corr of binary outcomes:\n",np.corrcoef(b.T).round(3))
    # pattern counts
    from collections import Counter
    pat=Counter(map(tuple,b)); print(" outcome patterns (flash,mini,5.2):",dict(sorted(pat.items(),key=lambda x:-x[1])))
    # complementarity: tasks where weaker solves but 5.2 fails
    print(" flash solves & 5.2 fails:",((b[:,0]==1)&(b[:,2]==0)).mean().round(3)," 5.2 solves & flash fails:",((b[:,0]==0)&(b[:,2]==1)).mean().round(3))
    L=np.array([len(r["state"]) for r in rows]); print(" statement chars median",np.median(L),"p90",np.percentile(L,90))
    # cost variance across tasks
    print(" cost cv per model",(cost.std(0)/cost.mean(0)).round(2), " cost ratio 5.2/flash median", np.median(cost[:,2]/cost[:,0]).round(1))
    # repo concentration
    rc=pd.Series([r["repo"] for r in rows]).value_counts(); print(" repos",len(rc)," top5",rc.head(5).to_dict())
# per-repo pass rates on train: how much does repo explain?
rows=D["train"]; df=pd.DataFrame({"repo":[r["repo"] for r in rows],**{m:[r["pass_rate"][m] for r in rows] for m in M}})
g=df.groupby("repo")[M].agg(["mean","count"])
for m in M:
    tot=df[m].var(); within=(df.groupby("repo")[m].transform(lambda x:x-x.mean())**2).mean()
    print(f"{m}: variance share explained by repo (train, in-sample) {1-within/df[m].var(ddof=0):.3f}")
