import glob, json, os, numpy as np, pandas as pd
base="/home/rohan/ws/2026/jev/data/"
ids={s:set(json.loads(l)["id"] for l in open(f"{base}processed/{s}.jsonl")) for s in ["train","val","test"]}
res={}
for d in sorted(glob.glob(base+"extra/*")):
    name=os.path.basename(d).replace("v3-2k-traj-","")
    fs=sorted(glob.glob(d+"/*.parquet"))
    df=pd.concat([pd.read_parquet(f,columns=["instance_id","resolved","instance_cost"]) for f in fs]).dropna(subset=["resolved"])
    g=df.groupby("instance_id").agg(r=("resolved","mean"),c=("instance_cost","mean"))
    res[name]=g
    print(f"{name:20s} runs {len(df):5d} tasks {len(g):5d} | in our train {len(set(g.index)&ids['train']):5d} val {len(set(g.index)&ids['val']):4d} test {len(set(g.index)&ids['test']):3d} | solve {g.r.mean():.3f} cost {g.c.mean():.3f}")
