import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
base="/home/rohan/ws/2026/jev/data/"
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
rows={s:[json.loads(l) for l in open(f"{base}processed/{s}.jsonl")] for s in ["train","val"]}
tab={}
for s in rows:
    tab[s]=pd.DataFrame([{ "id":r["id"],**{m:r["pass_rate"][m] for m in M}} for r in rows[s]]).set_index("id")
for name in ["deepseek-v3.2","gemini-2.5-pro","gemini-3-pro"]:
    fs=sorted(glob.glob(base+f"extra/v3-2k-traj-{name}/*.parquet"))
    df=pd.concat([pd.read_parquet(f,columns=["instance_id","resolved"]) for f in fs]).dropna()
    g=df.groupby("instance_id").resolved.mean()
    for s in tab: tab[s][name]=g.reindex(tab[s].index)
for s in tab:
    t=tab[s].dropna(); print(s,"tasks with all 6 models:",len(t))
    cols=list(t.columns)
    for target in M:
        others=[c for c in cols if c!=target]
        y=(t[target]>=.5).astype(int)
        print(f"  {s}: AUC of mean(other 5 models' pass rate) -> {target} solves: {roc_auc_score(y,t[others].mean(axis=1)):.3f} | single best other: "+
              ", ".join(f"{c}={roc_auc_score(y,t[c]):.2f}" for c in others))
    print("  solve rates:",t.mean().round(3).to_dict())
