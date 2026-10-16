import subprocess, hashlib, yaml, pandas as pd
def sh(*a): return subprocess.run(list(a), capture_output=True, text=True)
commits = sh("git","log","--format=%h","--","data/dataset.csv.dvc").stdout.split()[::-1]
for c in commits:
    sh("git","checkout",c,"--","data/dataset.csv.dvc")
    sh("dvc","pull","-f","data/dataset.csv.dvc")
    want = yaml.safe_load(open("data/dataset.csv.dvc"))["outs"][0]["md5"]
    got = hashlib.md5(open("data/dataset.csv","rb").read()).hexdigest()
    r, cols = pd.read_csv("data/dataset.csv").shape
    print(f"{c}: rows={r} cols={cols} md5(file)={got} md5(.dvc)={want} -> {'VERIFIED' if got==want else 'MISMATCH'}")