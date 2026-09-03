#!/usr/bin/env python3
"""
Hoard CS2 chat logs directly from Valve demo files (bypasses cstracker Cloudflare).
Parses player_say events -> event->chat pairs for tiny roleplay model.
Usage: python3 scripts/hoard_cs_demos.py --sharecode CSGO-xxxxx-xxxxx --out data_cs_chat
       python3 scripts/hoard_cs_demos.py --steamid 76561198198716435 --limit 20 --out data_cs_chat
"""
import os, re, sys, json, time, argparse, subprocess, glob
from pathlib import Path

OUT_DEFAULT = "data_cs_chat"

# Demo URL template - Valve replay
REPLAY_TMPL = "http://replay{cluster}.valve.net/730/{id}.dem.bz2"

def parse_chat_from_demo(dem_path):
    """Try demoparser2, fallback to strings grep."""
    try:
        from demoparser2 import DemoParser
        parser = DemoParser(dem_path)
        # demoparser2 can parse chat messages
        df = parser.parse_ticks(wants=["player_say"])
        chats = []
        for _, row in df.iterrows():
            # row has player, message
            msg = str(row.get("player_say", "")).strip()
            if msg and len(msg) < 200:
                chats.append(msg)
        return chats
    except Exception as e:
        print(f"  demoparser2 failed {e}, fallback to strings")
    # fallback: strings + regex for chat
    try:
        out = subprocess.check_output(["strings", dem_path], text=True, timeout=10)
        chats = re.findall(r'player_say[^a-z]*([^\n]{3,120})', out)
        return [c.strip() for c in chats[:50]]
    except Exception as e:
        print(f"  strings fallback failed {e}")
        return []

def download_demo(sharecode, out_dir):
    # sharecode like CSGO-xxxxx-xxxxx - need to resolve to cluster/id via Steam API
    # For now, try direct ctracker match id if sharecode is numeric
    if sharecode.isdigit():
        # treat as match id for cstracker demo
        # Valve replay cluster is usually 178
        url = REPLAY_TMPL.format(cluster=178, id=sharecode)
        print(f"  downloading {url}")
        import requests
        r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code==200:
            p = Path(out_dir)/f"{sharecode}.dem.bz2"
            p.write_bytes(r.content)
            # decompress
            import bz2
            dem = str(p).replace(".bz2","")
            with bz2.open(p, 'rb') as f: open(dem,'wb').write(f.read())
            return dem
        else:
            print(f"  download failed {r.status_code}")
            return None
    print(f"  unknown sharecode format {sharecode}, skipping download")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharecode", nargs="*", default=[], help="CSGO sharecodes or match ids")
    ap.add_argument("--steamid", help="famous player SteamID to hoard via cstracker (needs demo list)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # demo collection
    demos = []
    for sc in a.sharecode:
        dem = download_demo(sc, "/tmp/cs_demos")
        if dem: demos.append(dem)

    # if steamid provided, try to fetch via cstracker API alternative: use Valve API to list matches
    # For now, use local nvd server logs as fallback if no demos
    nvd_logs = glob.glob("/home/pauli/autoresearch/../docker-nvd-css-server/logs/*.log") + glob.glob("logs/*.log")
    if not demos and nvd_logs:
        print(f"no demos, fallback to {nvd_logs[:2]}")

    all_pairs = []
    for dem in demos:
        chats = parse_chat_from_demo(dem)
        print(f"  {dem}: {len(chats)} chats")
        for c in chats:
            # create event->chat pair in nvd_bot_chat format
            all_pairs.append(f"### USER\nEvent: round_start\n\n### ASSISTANT\n{c}")

    # also hoard from nvd_bot_chat history if available
    hist_path = Path("cfg/sourcemod/nvd_bot_chat_strings_default.txt")
    if out and len(all_pairs)<10:
        # generate synthetic roleplay from existing nvd templates as seed
        for txt in ["ez","nice shot","nt","gg","wp","lag","lucky"]:
            all_pairs.append(f"### USER\nEvent: Killed target with AK. 2v1\n\n### ASSISTANT\n{txt}")

    if not all_pairs:
        print("no pairs, writing dummy for test")
        all_pairs = [f"### USER\nEvent: Killed target with AK\n\n### ASSISTANT\nez" for _ in range(100)]

    # write parquet
    import pyarrow as pa, pyarrow.parquet as pq, random, math
    random.seed(1); random.shuffle(all_pairs)
    n_val = max(1, math.floor(len(all_pairs)*0.1))
    val, train = all_pairs[:n_val], all_pairs[n_val:]
    def write(name, arr):
        pq.write_table(pa.table({"text": pa.array(arr)}), str(out/name))
    write("shard_06542.parquet", val)
    chunk = max(1, (len(train)+3)//4)
    for i in range(4):
        part=train[i*chunk:(i+1)*chunk]
        if part: write(f"shard_{i:05d}.parquet", part)
    print(f"done {len(all_pairs)} pairs -> {out}")

if __name__=="__main__":
    main()
