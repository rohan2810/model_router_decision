import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import json, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
M=["deepseek-v4-flash","gpt-5-mini","gpt-5.2"]
# split-half reliability of run outcomes, per model & family
for fam in ["v3-2k-traj","swebench-verified"]:
    for m in M:
        fs=sorted(glob.glob(f"data/raw/{fam}-{m}/*.parquet"))
        df=pd.concat([pd.read_parquet(f,columns=["instance_id","resolved"]).assign(split=os.path.basename(f).split("-")[0]) for f in fs]).dropna()
        g=df.groupby("instance_id").resolved.apply(list)
        multi=g[g.map(len)>=2]
        if len(multi)<20: print(fam,m,"tasks with >=2 runs:",len(multi)); continue
        a=np.array([x[0] for x in multi]);b=np.array([x[1] for x in multi])
        # ceiling: if true p known perfectly, expected AUC vs one noisy run; approx by run1 as predictor of run2
        print(f"{fam:18s} {m:18s} tasks>=2 runs {len(multi):5d} | run1->run2 AUC {roc_auc_score(b,a):.3f} agreement {np.mean(a==b):.3f} | solve rate {df.resolved.mean():.3f}")
