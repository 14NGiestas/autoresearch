#!/usr/bin/env python3
"""
Hoard Valve demos via HLTV (bypasses cstracker Cloudflare) -> player_say
Usage: python3 scripts/hoard_valve_demos.py --limit 10 --out data_cs_demo
"""
import os, re, sys, time, bz2, glob, argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

HLTV_RESULTS = "https://www.hltv.org/results"
HEADERS = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

def get_match_ids(limit=10):
    print(f"[hltv] fetching {HLTV_RESULTS}")
    r = requests.get(HLTV_RESULTS, headers=HEADERS, timeout=15)
    if r.status_code!=200:
        print(f" hltv failed {r.status_code}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    ids=[]
    for a in soup.find_all("a", href=re.compile(r"/matches/\d+/")):
        m=re.search(r"/matches/(\d+)/", a["href"])
        if m: ids.append(m.group(1))
        if len(ids)>=limit*2: break
    ids=list(dict.fromkeys(ids))[:limit]
    print(f" found {len(ids)} match ids: {ids[:5]}")
    return ids

def get_demo_url(match_id):
    url=f"https://www.hltv.org/matches/{match_id}/x"
    r=requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code!=200: return None
    # HLTV demo link is /download/demo/<id> or gotv
    m=re.search(r'href="(/download/demo/\d+)"', r.text)
    if m: return "https://www.hltv.org"+m.group(1)
    m=re.search(r'(https?://[^"]+\.dem\.bz2)', r.text)
    if m: return m.group(1)
    return None

def download_demo(demo_url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = demo_url.split("/")[-1].split("?")[0] or "demo.dem.bz2"
    p = out_dir / fname
    if p.exists(): return p
    print(f"  dl {demo_url[:80]}")
    try:
        with requests.get(demo_url, headers=HEADERS, stream=True, timeout=30) as r:
            if r.status_code!=200:
                print(f"   fail {r.status_code}")
                return None
            with open(p, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return p
    except Exception as e:
        print(f"  dl err {e}")
        return None

def extract_chats(dem_bz2):
    # try demoparser2, else strings fallback
    dem = str(dem_bz2).replace(".bz2","")
    if not Path(dem).exists():
        try:
            with bz2.open(dem_bz2, 'rb') as f: open(dem,'wb').write(f.read())
        except: return []
    try:
        from demoparser2 import DemoParser
        parser = DemoParser(dem)
        df = parser.parse_ticks(wants=["player_say"])
        # demoparser2 returns dataframe with player_say
        chats=[]
        for _, row in df.iterrows():
            msg=str(row.get("player_say","")).strip()
            if 2 < len(msg) < 120:
                chats.append(msg)
        return chats[:200]
    except Exception as e:
        print(f"   demoparser2 fail {e}")
        # strings fallback
        import subprocess
        try:
            out=subprocess.check_output(["strings", dem], text=True, timeout=10)
            chats=re.findall(r'player_say[^\n]{0,20}([^\n]{3,100})', out)
            return [c.strip() for c in chats[:50] if c.strip()]
        except: return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="/home/pauli/.cache/autoresearch/data_cs_demo")
    a=ap.parse_args()
    out=Path(a.out)
    tmp=Path("/tmp/cs_demos")
    tmp.mkdir(exist_ok=True)
    ids=get_match_ids(a.limit)
    all_chats=[]
    for mid in ids:
        demo_url=get_demo_url(mid)
        if not demo_url:
            print(f" no demo for {mid}")
            continue
        p=download_demo(demo_url, tmp)
        if not p: continue
        chats=extract_chats(p)
        print(f" {mid}: {len(chats)} chats")
        for c in chats:
            all_chats.append(f"### USER\nEvent: match {mid}\n\n### ASSISTANT\n{c}")
        time.sleep(1)
        if len(all_chats)>5000: break

    if not all_chats:
        print(" no chats, writing synthetic fallback 12k")
        # reuse earlier synthetic
        import pyarrow as pa, pyarrow.parquet as pq, random, math
        chats=["ez","nt","nice shot","wp","gg","lag","lucky","ns","go","rush"]
        all_chats=[f"### USER\nEvent: Killed target\n\n### ASSISTANT\n{random.choice(chats)}" for _ in range(12000)]

    # write parquet for 15M trainer
    import pyarrow as pa, pyarrow.parquet as pq, random, math
    random.seed(1); random.shuffle(all_chats)
    n_val=max(1, math.floor(len(all_chats)*0.1))
    val, train = all_chats[:n_val], all_chats[n_val:]
    out.mkdir(parents=True, exist_ok=True)
    def write(name, arr): pq.write_table(pa.table({"text": pa.array(arr)}), str(out/name))
    write("shard_06542.parquet", val)
    chunk=max(1,(len(train)+3)//4)
    for i in range(4):
        part=train[i*chunk:(i+1)*chunk]
        if part: write(f"shard_{i:05d}.parquet", part)
    print(f"done {len(all_chats)} -> {out}")

if __name__=="__main__":
    main()
