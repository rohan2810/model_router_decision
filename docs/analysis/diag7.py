import glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
for fam in ["swebench-verified","v3-2k-traj"]:
    for m in ["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]:
        fs=sorted(glob.glob(f"/home/rohan/ws/2026/jev/data/raw/{fam}-{m}/*.parquet"))
        df=pd.concat([pd.read_parquet(f,columns=["instance_id","resolved","instance_cost","api_calls","step_cost_list"]) for f in fs]).dropna(subset=["resolved"])
        sc=df.step_cost_list.map(lambda x: np.array(x,float))
        out={f"first{K}":np.mean([s[:K].sum() for s in sc]) for K in (2,4,8)}
        print(f"{fam:18s} {m:18s} n={len(df):5d} mean cost {df.instance_cost.mean():.4f} | "+" ".join(f"{k}=${v:.4f}" for k,v in out.items())+
              f" | mean api_calls {df.api_calls.mean():.1f} | AUC(-api_calls -> resolved) {roc_auc_score(df.resolved.astype(int),-df.api_calls):.3f}")
